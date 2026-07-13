// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dit_fused_distributed_groupnorm_program_factory.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <unordered_map>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/operations/normalization/groupnorm/device/groupnorm_program_utils.hpp"

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::experimental::prim {

namespace {

uint32_t float_to_u32(float v) {
    uint32_t out;
    std::memcpy(&out, &v, sizeof(float));
    return out;
}

}  // namespace

// Single worker core per device runs the stock welford GroupNorm kernels (reader / writer /
// compute). The only additions vs. ttnn::group_norm are (1) the reader publishes each device's
// per-group (mean, var) stick over the fabric ring, gathers every device's stick, and Chan-merges
// them before feeding the unchanged compute POST; (2) a forwarder core drives the ring all-gather.
DitFusedDistributedGroupnormMeshWorkloadFactory::cached_program_t
DitFusedDistributedGroupnormMeshWorkloadFactory::create_at(
    const DitFusedDistributedGroupnormParams& args,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const DitFusedDistributedGroupnormInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    Tensor& output_tensor = tensor_return_value.at(0);
    Tensor* stats_dram_tensor = tensor_return_value.size() > 1 ? &tensor_return_value[1] : nullptr;
    const auto& input_tensor = tensor_args.input;
    const auto& gamma = tensor_args.gamma;
    const auto& beta = tensor_args.beta;
    const auto& input_mask = tensor_args.input_mask;

    Program program = CreateProgram();

    IDevice* device = input_tensor.device();
    auto* mesh_device = input_tensor.device();

    const bool is_local = (args.ring_size <= 1);
    const bool use_mux = !is_local;

    TT_FATAL(input_mask.has_value(), "dit_fused_distributed_groupnorm requires an input_mask (welford GroupNorm)");

    // ------------------------------------------------------------------------
    // Shapes / single-core GroupNorm sizing (num_cores == 1, no mcast topology)
    // ------------------------------------------------------------------------
    const uint32_t tile_h = TILE_HEIGHT;
    const uint32_t tile_w = TILE_WIDTH;
    const uint32_t datum_size_bytes = 2;  // bfloat16

    const auto& shape = input_tensor.padded_shape();
    const uint32_t num_batches = shape[0];
    const uint32_t H = shape[1] * shape[2] * num_batches;  // folded spatial
    const uint32_t Ht = H / tile_h;
    const uint32_t W = shape[3];  // channels
    const uint32_t Wt = W / tile_w;
    const uint32_t num_groups = args.num_groups;

    TT_FATAL(num_batches == 1, "dit_fused_distributed_groupnorm currently supports N==1 (got {})", num_batches);
    TT_FATAL(W % tile_w == 0, "channels ({}) must be divisible by {}", W, tile_w);
    TT_FATAL(W % num_groups == 0, "channels ({}) must be divisible by num_groups ({})", W, num_groups);
    TT_FATAL(H % tile_h == 0, "folded HW ({}) must be divisible by {}", H, tile_h);

    const uint32_t num_channels_per_group = W / num_groups;
    const uint32_t num_channels_per_group_mod_tile_w =
        (num_channels_per_group % tile_w == 0) ? tile_w : (num_channels_per_group % tile_w);

    const uint32_t num_batches_per_core = 1;
    const uint32_t num_groups_per_core = num_groups;
    const uint32_t per_core_Mt = Ht;
    const uint32_t per_core_Nt = Wt;
    const uint32_t per_core_N = W;
    const uint32_t per_core_M = Ht * tile_h;         // = H
    const uint32_t num_rows_per_group = per_core_M;  // rows per group per device (N==1)
    const uint32_t num_out_blocks = 1;

    auto [block_wt, num_groups_per_reset] = ttnn::prim::find_max_tile_span(per_core_N, num_channels_per_group);
    const uint32_t block_ht = per_core_Mt;  // num_batches_per_core == 1
    const uint32_t block_wt_last = (per_core_Nt + num_groups_per_core - 1) / num_groups_per_core;
    const uint32_t num_cores_per_mcast_group = 1;
    const uint32_t per_core_N_bytes_padded =
        tt::round_up(per_core_N * datum_size_bytes, output_tensor.buffer()->alignment());

    TT_FATAL(num_groups_per_core <= 16, "num_groups_per_core ({}) must be <= 16 for welford", num_groups_per_core);
    TT_FATAL(
        input_mask->padded_shape()[3] == block_wt * tile_w,
        "input mask width ({}) must equal block_wt * TILE_WIDTH ({})",
        input_mask->padded_shape()[3],
        block_wt * tile_w);

    // ------------------------------------------------------------------------
    // Data formats
    // ------------------------------------------------------------------------
    const tt::DataFormat in_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    const tt::DataFormat out_data_format = datatype_to_dataformat_converter(output_tensor.dtype());
    const tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;  // intermediates + bf16 stats
    const tt::DataFormat fp32_df = tt::DataFormat::Float32;
    tt::DataFormat gamma_beta_cb_data_format = tt::DataFormat::Float16_b;
    if (gamma.has_value()) {
        gamma_beta_cb_data_format = datatype_to_dataformat_converter(gamma->dtype());
    } else if (beta.has_value()) {
        gamma_beta_cb_data_format = datatype_to_dataformat_converter(beta->dtype());
    }
    const tt::DataFormat in_mask_cb_data_format = datatype_to_dataformat_converter(input_mask->dtype());

    const uint32_t in_single_tile_size = tt::tile_size(in_data_format);
    const uint32_t out_single_tile_size = tt::tile_size(out_data_format);
    const uint32_t single_tile_size = tt::tile_size(cb_data_format);
    const uint32_t gamma_beta_single_tile_size = tt::tile_size(gamma_beta_cb_data_format);
    const uint32_t in_mask_single_tile_size = tt::tile_size(in_mask_cb_data_format);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), args.compute_kernel_config);
    TT_FATAL(fp32_dest_acc_en, "dit_fused_distributed_groupnorm requires fp32_dest_acc_en=true");

    const bool welford_unpack_fp32_active = (in_data_format == tt::DataFormat::Float32) && fp32_dest_acc_en;
    const bool welford_fp32_alias = welford_unpack_fp32_active;  // tilize_in == false
    const uint32_t cb_in0_welford_index =
        welford_fp32_alias ? static_cast<uint32_t>(tt::CBIndex::c_19) : static_cast<uint32_t>(tt::CBIndex::c_0);

    const bool has_gamma = gamma.has_value();
    const bool has_beta = beta.has_value();
    const uint32_t groupnorm_mode = static_cast<uint32_t>(ttnn::prim::GroupNormMode::WELFORD_NATIVE);

    // ------------------------------------------------------------------------
    // Ring topology (unchanged from CCL scaffold)
    // ------------------------------------------------------------------------
    std::optional<ttnn::MeshCoordinate> forward_coord = std::nullopt;
    std::optional<ttnn::MeshCoordinate> backward_coord = std::nullopt;
    std::optional<tt::tt_fabric::FabricNodeId> forward_fabric_node_id = std::nullopt;
    std::optional<tt::tt_fabric::FabricNodeId> backward_fabric_node_id = std::nullopt;
    uint32_t device_index = 0;
    if (use_mux) {
        forward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
            input_tensor, mesh_coordinate, /*offset=*/1, args.topology, args.cluster_axis);
        backward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
            input_tensor, mesh_coordinate, /*offset=*/-1, args.topology, args.cluster_axis);
        device_index =
            ttnn::ccl::get_linearized_index_from_physical_coord(input_tensor, mesh_coordinate, args.cluster_axis);
        if (forward_coord.has_value()) {
            forward_fabric_node_id = mesh_device->get_fabric_node_id(forward_coord.value());
        }
        if (backward_coord.has_value()) {
            backward_fabric_node_id = mesh_device->get_fabric_node_id(backward_coord.value());
        }
    }

    uint32_t num_targets_forward = 0;
    uint32_t num_targets_backward = 0;
    if (use_mux) {
        if (args.topology == ttnn::ccl::Topology::Linear) {
            ttnn::ccl::LineTopology line_topology(args.ring_size, device_index);
            num_targets_forward = line_topology.get_distance_to_end_of_line(ttnn::ccl::LineDirection::FORWARD);
            num_targets_backward = line_topology.get_distance_to_end_of_line(ttnn::ccl::LineDirection::BACKWARD);
        } else if (args.topology == ttnn::ccl::Topology::Ring) {
            num_targets_forward = tt::div_up(args.ring_size - 1, 2);
            num_targets_backward = args.ring_size - 1 - num_targets_forward;
            if (device_index % 2 == 0) {
                std::swap(num_targets_forward, num_targets_backward);
            }
        }
    }

    // ------------------------------------------------------------------------
    // Cores: 1 worker (+ 1 forwarder when AG)
    // ------------------------------------------------------------------------
    const auto grid_size = device->compute_with_storage_grid_size();
    const CoreRange core_grid({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    const uint32_t max_cores = core_grid.size();

    const uint32_t num_workers = 1u;
    const uint32_t num_forwarders = use_mux ? 1u : 0u;
    TT_FATAL(
        num_workers + num_forwarders <= max_cores,
        "dit_fused_distributed_groupnorm needs {} cores but only {} available",
        num_workers + num_forwarders,
        max_cores);

    constexpr uint32_t max_rounds = 1u;
    const uint32_t num_chunks_per_device = use_mux ? (num_forwarders * max_rounds) : 0u;

    const auto all_cores_vec = corerange_to_cores(core_grid, max_cores, /*row_major=*/true);
    const CoreCoord worker_core = all_cores_vec[0];
    std::vector<CoreCoord> worker_cores{worker_core};
    std::vector<CoreCoord> forwarder_cores;
    for (uint32_t f = 0; f < num_forwarders; f++) {
        forwarder_cores.push_back(all_cores_vec[num_workers + f]);
    }

    const CoreRangeSet worker_core_set({CoreRange(worker_core, worker_core)});
    CoreRangeSet forwarder_core_set;
    for (const auto& c : forwarder_cores) {
        forwarder_core_set = forwarder_core_set.merge(CoreRangeSet({CoreRange(c, c)}));
    }
    const CoreRangeSet all_core_set({core_grid});

    TT_FATAL(
        !use_mux || stats_dram_tensor != nullptr,
        "create_at requires stats DRAM scratch at tensor_return_value[1] when ring_size > 1");
    tt::tt_metal::Buffer* stats_dram_buffer = use_mux ? stats_dram_tensor->buffer() : nullptr;

    // bf16 [mean, var] per group. Must match compute_sizing/make_sizing: the stick is the
    // per-device DRAM page in the gathered stats buffer, so it is rounded up to a multiple of
    // NOC_DRAM_READ_ALIGNMENT_BYTES (64 on Blackhole) — otherwise pages at non-64-aligned DRAM
    // offsets (odd device indices at 32 B) read back as zero and halve the merged variance.
    const uint32_t stick_bytes = tt::round_up(num_groups * 2u * datum_size_bytes, 64u);
    if (use_mux) {
        const uint32_t max_payload = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();
        TT_FATAL(
            stick_bytes <= max_payload,
            "GN stats stick ({} B) exceeds fabric max payload ({} B)",
            stick_bytes,
            max_payload);
    }

    // ------------------------------------------------------------------------
    // Circular buffers (GroupNorm welford CB scheme on the worker core)
    // ------------------------------------------------------------------------
    auto create_cb =
        [&](const CoreRangeSet& cores, uint32_t cb_id, uint32_t total_bytes, tt::DataFormat df, uint32_t page) {
            CircularBufferConfig cfg(total_bytes, {{cb_id, df}});
            cfg.set_page_size(cb_id, page);
            CreateCircularBuffer(program, cores, cfg);
        };

    const uint32_t in0_block_tiles = block_ht / num_out_blocks * block_wt;
    const uint32_t interm_block_tiles = in0_block_tiles;

    // c_0 input (+ c_19 alias on fp32 path)
    {
        std::map<uint8_t, tt::DataFormat> fmts = {{static_cast<uint8_t>(tt::CBIndex::c_0), in_data_format}};
        if (welford_fp32_alias) {
            fmts[static_cast<uint8_t>(tt::CBIndex::c_19)] = in_data_format;
        }
        CircularBufferConfig cfg(in0_block_tiles * in_single_tile_size, fmts);
        cfg.set_page_size(tt::CBIndex::c_0, in_single_tile_size);
        if (welford_fp32_alias) {
            cfg.set_page_size(tt::CBIndex::c_19, in_single_tile_size);
        }
        CreateCircularBuffer(program, worker_core_set, cfg);
    }
    create_cb(
        worker_core_set,
        tt::CBIndex::c_16,
        in0_block_tiles * out_single_tile_size,
        out_data_format,
        out_single_tile_size);
    create_cb(
        worker_core_set, tt::CBIndex::c_29, in0_block_tiles * in_single_tile_size, in_data_format, in_single_tile_size);
    create_cb(worker_core_set, tt::CBIndex::c_2, single_tile_size, cb_data_format, single_tile_size);
    create_cb(worker_core_set, tt::CBIndex::c_3, single_tile_size, cb_data_format, single_tile_size);  // eps
    create_cb(worker_core_set, tt::CBIndex::c_4, single_tile_size, cb_data_format, single_tile_size);
    if (has_gamma) {
        create_cb(
            worker_core_set,
            tt::CBIndex::c_5,
            per_core_Nt * gamma_beta_single_tile_size,
            gamma_beta_cb_data_format,
            gamma_beta_single_tile_size);
    }
    if (has_beta) {
        create_cb(
            worker_core_set,
            tt::CBIndex::c_6,
            per_core_Nt * gamma_beta_single_tile_size,
            gamma_beta_cb_data_format,
            gamma_beta_single_tile_size);
    }
    create_cb(
        worker_core_set,
        tt::CBIndex::c_28,
        input_mask->physical_volume() * input_mask->element_size(),
        in_mask_cb_data_format,
        in_mask_single_tile_size);
    create_cb(worker_core_set, tt::CBIndex::c_24, single_tile_size * 1, cb_data_format, single_tile_size);  // x
    create_cb(worker_core_set, tt::CBIndex::c_25, single_tile_size * 3, cb_data_format, single_tile_size);  // xmm
    create_cb(
        worker_core_set,
        tt::CBIndex::c_23,
        interm_block_tiles * single_tile_size,
        cb_data_format,
        single_tile_size);  // xmm2
    create_cb(
        worker_core_set,
        tt::CBIndex::c_22,
        interm_block_tiles * single_tile_size,
        cb_data_format,
        single_tile_size);                                                                                 // xmm3
    create_cb(worker_core_set, tt::CBIndex::c_8, single_tile_size * 2, cb_data_format, single_tile_size);  // ex_partial
    // ex_global: dual buffer index c_15 / c_9.
    {
        CircularBufferConfig cfg(
            2u * single_tile_size * num_groups_per_core,
            {{static_cast<uint8_t>(tt::CBIndex::c_15), cb_data_format},
             {static_cast<uint8_t>(tt::CBIndex::c_9), cb_data_format}});
        cfg.set_page_size(tt::CBIndex::c_15, single_tile_size);
        cfg.set_page_size(tt::CBIndex::c_9, single_tile_size);
        CreateCircularBuffer(program, worker_core_set, cfg);
    }
    create_cb(
        worker_core_set,
        tt::CBIndex::c_27,
        single_tile_size * num_groups_per_core,
        cb_data_format,
        single_tile_size);  // ex2pe

    // Fabric AG staging CBs (worker) + packet CB (grid-uniform) + header CB (forwarder).
    constexpr uint32_t stats_local_cb_id = tt::CBIndex::c_1;
    constexpr uint32_t stats_gathered_cb_id = tt::CBIndex::c_10;
    constexpr uint32_t packet_cb_id = tt::CBIndex::c_7;
    constexpr uint32_t reserved_packet_header_cb_id = tt::CBIndex::c_11;
    if (use_mux) {
        create_cb(worker_core_set, stats_local_cb_id, 2u * stick_bytes, fp32_df, stick_bytes);
        create_cb(worker_core_set, stats_gathered_cb_id, args.ring_size * stick_bytes, fp32_df, stick_bytes);
        create_cb(all_core_set, packet_cb_id, 2u * stick_bytes, fp32_df, stick_bytes);
        const uint32_t header_size = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
        create_cb(
            forwarder_core_set, reserved_packet_header_cb_id, 4u * header_size, tt::DataFormat::RawUInt32, header_size);
    }

    // ------------------------------------------------------------------------
    // Semaphores: reduce_sender (id 0, reused as receiver) + fabric arrival / go
    // ------------------------------------------------------------------------
    const uint32_t reduce_sender_semaphore_id = CreateSemaphore(program, all_core_set, 0u);
    const uint32_t reduce_receiver_semaphore_id = reduce_sender_semaphore_id;
    const uint32_t arrival_sem_id = use_mux ? CreateSemaphore(program, all_core_set, 0u) : 0u;
    const uint32_t go_sem_id = use_mux ? CreateSemaphore(program, all_core_set, 0u) : 0u;

    const std::string kernel_base =
        "ttnn/cpp/ttnn/operations/experimental/ccl/dit_fused_distributed_groupnorm/device/kernels/";

    // ------------------------------------------------------------------------
    // Reader (welford GN + fabric AG)
    // ------------------------------------------------------------------------
    std::vector<uint32_t> reader_ct = {
        reduce_receiver_semaphore_id,
        reduce_sender_semaphore_id,
        num_cores_per_mcast_group,
        num_groups_per_core * num_batches_per_core,  // num_batch_group
        num_batches_per_core,                        // num_batches
        per_core_Nt,                                 // per_core_N
        per_core_N_bytes_padded,                     // per_core_N_bytes
        per_core_Nt * tile_w * datum_size_bytes,     // per_core_N_bytes_with_stride
        per_core_Mt,                                 // per_core_M
        tile_h,
        tile_w,
        block_ht,
        block_wt,
        per_core_Mt * Wt / num_batches_per_core,  // num_tiles_per_batch
        num_out_blocks,
        num_channels_per_group,
        num_rows_per_group,
        cb_in0_welford_index,
        static_cast<uint32_t>(welford_fp32_alias),
        args.ring_size,
        stick_bytes,
        num_chunks_per_device,
        arrival_sem_id,
        go_sem_id,
        packet_cb_id,
        stats_local_cb_id,
        stats_gathered_cb_id,
    };
    TensorAccessorArgs(input_tensor.buffer()).append_to(reader_ct);
    TensorAccessorArgs(use_mux ? stats_dram_buffer : input_tensor.buffer()).append_to(reader_ct);

    std::map<std::string, std::string> reader_defines;
    if (has_gamma) {
        reader_defines["FUSE_GAMMA"] = "1";
    }
    if (has_beta) {
        reader_defines["FUSE_BETA"] = "1";
    }
    KernelHandle reader_kernel_id = CreateKernel(
        program,
        kernel_base + "dataflow/dit_gn_welford_reader.cpp",
        worker_core_set,
        ReaderDataMovementConfig(reader_ct, reader_defines));

    // ------------------------------------------------------------------------
    // Writer (welford GN gamma/beta + output)
    // ------------------------------------------------------------------------
    std::vector<uint32_t> writer_ct = {
        1u,  // is_mcast_sender
        static_cast<uint32_t>(has_gamma),
        static_cast<uint32_t>(has_beta),
        per_core_Nt,  // num_cols_tile_gamma_beta
        per_core_Mt,
        per_core_Nt,
        per_core_N * datum_size_bytes,            // per_core_N_bytes
        per_core_Nt * tile_w * datum_size_bytes,  // per_core_N_bytes_with_stride
        num_groups_per_core,
        num_batches_per_core,
        num_channels_per_group_mod_tile_w,
        per_core_Mt * Wt / num_batches_per_core,  // num_tiles_per_batch
        block_wt_last,
        static_cast<uint32_t>((num_channels_per_group_mod_tile_w & (num_channels_per_group_mod_tile_w - 1)) == 0),
        static_cast<uint32_t>(num_channels_per_group < tile_w),
        num_channels_per_group - ((block_wt - 1) * tile_w),  // group_row_offset
        num_out_blocks,
        block_ht,
        block_wt,
        block_ht * block_wt,
        groupnorm_mode,
        tile_w,
    };
    TensorAccessorArgs(output_tensor.buffer()).append_to(writer_ct);
    TensorAccessorArgs(has_gamma ? gamma->buffer() : nullptr).append_to(writer_ct);
    TensorAccessorArgs(has_beta ? beta->buffer() : nullptr).append_to(writer_ct);
    TensorAccessorArgs(input_mask->buffer()).append_to(writer_ct);

    KernelHandle writer_kernel_id = CreateKernel(
        program,
        kernel_base + "dataflow/dit_gn_welford_writer.cpp",
        worker_core_set,
        WriterDataMovementConfig(writer_ct));

    // ------------------------------------------------------------------------
    // Forwarder (fabric ring all-gather)
    // ------------------------------------------------------------------------
    std::vector<KernelHandle> forwarder_kernel_ids(num_forwarders, 0);
    for (uint32_t f = 0; f < num_forwarders; f++) {
        std::vector<uint32_t> fwd_ct = {
            packet_cb_id,
            reserved_packet_header_cb_id,
            args.ring_size,
            device_index,
            num_targets_forward,
            num_targets_backward,
            f,
            num_forwarders,
            1u,  // group_size (workers per forwarder)
            max_rounds,
            stick_bytes,
            num_chunks_per_device,
            arrival_sem_id,
            go_sem_id,
        };
        TensorAccessorArgs(stats_dram_buffer).append_to(fwd_ct);
        forwarder_kernel_ids[f] = CreateKernel(
            program,
            kernel_base + "dataflow/dit_gn_fused_forwarder.cpp",
            CoreRangeSet({CoreRange(forwarder_cores[f], forwarder_cores[f])}),
            WriterDataMovementConfig(fwd_ct));
    }

    // ------------------------------------------------------------------------
    // Compute (stock welford GroupNorm PRE/POST)
    // ------------------------------------------------------------------------
    std::vector<uint32_t> compute_ct = {
        static_cast<uint32_t>(has_gamma),  // do_gamma
        static_cast<uint32_t>(has_beta),   // do_beta
        num_cores_per_mcast_group,
        num_batches_per_core,  // batch
        num_groups_per_core,   // group
        block_ht,
        block_wt,
        per_core_Mt,
        per_core_Nt,
        per_core_Mt * per_core_Nt,       // per_core_MN
        single_tile_size,                // single_tile_size_bytes
        num_groups_per_core * block_wt,  // num_tiles_input_mask
        num_out_blocks,
        num_channels_per_group,
        0u,  // reciprocal_size (welford native)
        tile_w,
        cb_in0_welford_index,
        static_cast<uint32_t>(welford_fp32_alias),
        static_cast<uint32_t>(welford_unpack_fp32_active),
    };
    ComputeConfig compute_config{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .math_approx_mode = math_approx_mode,
        .compile_args = compute_ct,
    };
    if (welford_unpack_fp32_active) {
        std::vector<UnpackToDestMode> unpack_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
        if (welford_fp32_alias) {
            unpack_mode[static_cast<uint32_t>(tt::CBIndex::c_19)] = UnpackToDestMode::UnpackToDestFp32;
        }
        compute_config.unpack_to_dest_mode = unpack_mode;
    }
    KernelHandle compute_kernel_id =
        CreateKernel(program, kernel_base + "compute/dit_gn_welford_compute.cpp", worker_core_set, compute_config);

    // ------------------------------------------------------------------------
    // Runtime args
    // ------------------------------------------------------------------------
    const uint32_t input_addr = input_tensor.buffer()->address();
    const uint32_t output_addr = output_tensor.buffer()->address();
    const uint32_t gamma_addr = has_gamma ? gamma->buffer()->address() : 0u;
    const uint32_t beta_addr = has_beta ? beta->buffer()->address() : 0u;
    const uint32_t input_mask_addr = input_mask->buffer()->address();
    const uint32_t stats_dram_addr = use_mux ? stats_dram_buffer->address() : 0u;

    uint32_t out_ready_sem_bank_addr = 0;
    if (use_mux) {
        TT_FATAL(!args.multi_device_global_semaphore.empty(), "ring_size>1 requires a GlobalSemaphore");
        out_ready_sem_bank_addr = args.multi_device_global_semaphore.at(0).address();
    }

    const CoreCoord worker_virtual = mesh_device->worker_core_from_logical_core(worker_core);
    std::vector<CoreCoord> forwarder_virtual(num_forwarders);
    for (uint32_t f = 0; f < num_forwarders; f++) {
        forwarder_virtual[f] = mesh_device->worker_core_from_logical_core(forwarder_cores[f]);
    }

    // Reader RT args: GN single-core layout (0..13) + fabric AG (14..18 when distributed).
    std::vector<uint32_t> reader_rt = {
        input_addr,
        output_addr,
        0u,                                       // in0_start_id
        0u,                                       // out_tile_start_id
        Wt,                                       // num_channels_tiles
        0u,                                       // has_mcast_first_group
        0u,                                       // has_mcast_last_group
        static_cast<uint32_t>(worker_virtual.x),  // mcast mid start (self)
        static_cast<uint32_t>(worker_virtual.y),
        static_cast<uint32_t>(worker_virtual.x),  // mcast mid end (self)
        static_cast<uint32_t>(worker_virtual.y),
        0u,                                       // num mcast cores mid group - 1
        static_cast<uint32_t>(worker_virtual.x),  // noc_coord_x[0]
        static_cast<uint32_t>(worker_virtual.y),  // noc_coord_y[0]
    };
    std::optional<size_t> stats_dram_addr_reader_arg_idx;
    if (use_mux) {
        stats_dram_addr_reader_arg_idx = reader_rt.size();
        reader_rt.push_back(stats_dram_addr);
        reader_rt.push_back(static_cast<uint32_t>(forwarder_virtual[0].x));
        reader_rt.push_back(static_cast<uint32_t>(forwarder_virtual[0].y));
        reader_rt.push_back(0u);  // my_slot
        reader_rt.push_back(0u);  // my_forwarder_index
    }
    SetRuntimeArgs(program, reader_kernel_id, worker_core, reader_rt);

    // Writer RT args (welford GN).
    std::vector<uint32_t> writer_rt = {
        float_to_u32(args.eps),
        output_addr,
        gamma_addr,
        beta_addr,
        input_mask_addr,
        0u,  // out_start_id
        0u,  // gamma_tile_start_id
        0u,  // beta_tile_start_id
        0u,  // input_mask_tile_start_id
        Wt,  // num_channels_tiles
    };
    SetRuntimeArgs(program, writer_kernel_id, worker_core, writer_rt);

    // Forwarder RT args.
    if (num_forwarders > 0) {
        const auto local_node_id = mesh_device->get_fabric_node_id(mesh_coordinate);
        for (uint32_t f = 0; f < num_forwarders; f++) {
            const auto& core = forwarder_cores[f];
            std::vector<uint32_t> fwd_rt = {stats_dram_addr, out_ready_sem_bank_addr};
            fwd_rt.push_back(static_cast<uint32_t>(worker_virtual.x));
            fwd_rt.push_back(static_cast<uint32_t>(worker_virtual.y));
            fwd_rt.push_back(1u);  // present_count[0]
            fwd_rt.push_back(forward_fabric_node_id.has_value() ? 1u : 0u);
            if (forward_fabric_node_id.has_value()) {
                tt::tt_fabric::append_fabric_connection_rt_args(
                    local_node_id, forward_fabric_node_id.value(), /*link_idx=*/f, program, {core}, fwd_rt);
            }
            fwd_rt.push_back(backward_fabric_node_id.has_value() ? 1u : 0u);
            if (backward_fabric_node_id.has_value()) {
                tt::tt_fabric::append_fabric_connection_rt_args(
                    local_node_id, backward_fabric_node_id.value(), /*link_idx=*/f, program, {core}, fwd_rt);
            }
            SetRuntimeArgs(program, forwarder_kernel_ids[f], core, fwd_rt);
        }
    }

    return {
        std::move(program),
        DitFusedDistributedGroupnormSharedVariables{
            .reader_kernel_ids = {reader_kernel_id},
            .writer_kernel_ids = {writer_kernel_id},
            .compute_kernel_ids = {compute_kernel_id},
            .forwarder_kernel_ids = forwarder_kernel_ids,
            .forwarder_cores = forwarder_cores,
            .cores = worker_cores,
            .stats_dram_addr_writer_arg_idx = stats_dram_addr_reader_arg_idx,
        }};
}

DitFusedDistributedGroupnormMeshWorkloadFactory::cached_mesh_workload_t
DitFusedDistributedGroupnormMeshWorkloadFactory::create_mesh_workload(
    const DitFusedDistributedGroupnormParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const DitFusedDistributedGroupnormInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    for (const auto& range : tensor_coords.ranges()) {
        for (const auto& coord : range) {
            auto cached = create_at(operation_attributes, coord, tensor_args, tensor_return_value);
            workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached.program));
            shared_variables.emplace(ttnn::MeshCoordinateRange(coord), std::move(cached.shared_variables));
        }
    }
    return cached_mesh_workload_t{std::move(workload), std::move(shared_variables)};
}

void DitFusedDistributedGroupnormMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const DitFusedDistributedGroupnormParams& operation_attributes,
    const DitFusedDistributedGroupnormInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    const uint32_t input_addr = tensor_args.input.buffer()->address();
    const uint32_t output_addr = tensor_return_value.at(0).buffer()->address();
    const uint32_t gamma_addr = tensor_args.gamma.has_value() ? tensor_args.gamma->buffer()->address() : 0u;
    const uint32_t beta_addr = tensor_args.beta.has_value() ? tensor_args.beta->buffer()->address() : 0u;
    const uint32_t input_mask_addr =
        tensor_args.input_mask.has_value() ? tensor_args.input_mask->buffer()->address() : 0u;
    const uint32_t stats_dram_addr = tensor_return_value.size() > 1 ? tensor_return_value[1].buffer()->address() : 0u;
    const uint32_t out_ready_sem_addr = operation_attributes.multi_device_global_semaphore.empty()
                                            ? 0u
                                            : operation_attributes.multi_device_global_semaphore.at(0).address();

    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        const auto& shared = cached_workload.shared_variables.at(range);
        auto& reader_runtime_args_by_core = GetRuntimeArgs(program, shared.reader_kernel_ids[0]);
        auto& writer_runtime_args_by_core = GetRuntimeArgs(program, shared.writer_kernel_ids[0]);

        for (const auto& core : shared.cores) {
            auto& reader_args = reader_runtime_args_by_core.at(core.x).at(core.y);
            reader_args[0] = input_addr;
            reader_args[1] = output_addr;
            if (shared.stats_dram_addr_writer_arg_idx.has_value()) {
                reader_args[shared.stats_dram_addr_writer_arg_idx.value()] = stats_dram_addr;
            }

            auto& writer_args = writer_runtime_args_by_core.at(core.x).at(core.y);
            writer_args[1] = output_addr;
            writer_args[2] = gamma_addr;
            writer_args[3] = beta_addr;
            writer_args[4] = input_mask_addr;
        }
        for (size_t f = 0; f < shared.forwarder_kernel_ids.size(); f++) {
            auto& fwd_args_by_core = GetRuntimeArgs(program, shared.forwarder_kernel_ids[f]);
            const auto& fc = shared.forwarder_cores[f];
            fwd_args_by_core.at(fc.x).at(fc.y)[0] = stats_dram_addr;
            fwd_args_by_core.at(fc.x).at(fc.y)[1] = out_ready_sem_addr;
        }
    }
}

}  // namespace ttnn::experimental::prim
