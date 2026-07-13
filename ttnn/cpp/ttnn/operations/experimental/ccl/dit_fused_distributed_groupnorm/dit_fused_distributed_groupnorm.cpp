// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dit_fused_distributed_groupnorm.hpp"

#include <tt-metalium/constants.hpp>

#include "device/dit_fused_distributed_groupnorm_device_operation.hpp"
#include "device/dit_fused_distributed_groupnorm_device_operation_types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

using namespace tt::constants;

namespace ttnn::experimental {

namespace {

uint32_t cluster_width(const MeshDevice& mesh_device, uint32_t cluster_axis) {
    const auto& mesh_view = mesh_device.get_view();
    return static_cast<uint32_t>((cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols());
}

ttnn::experimental::prim::DitFusedDistributedGroupnormSizing make_sizing(uint32_t num_groups, uint32_t ring_size) {
    ttnn::experimental::prim::DitFusedDistributedGroupnormSizing s;
    s.num_groups = num_groups;
    s.is_local = (ring_size <= 1);
    // Per-group bf16 [mean, var] (2 * 2 B). The stick doubles as the per-device DRAM page in
    // the gathered stats buffer, so its size MUST be a multiple of NOC_DRAM_READ_ALIGNMENT_BYTES
    // (64 on Blackhole): the reader NoC-reads page d at DRAM offset d*stick_bytes, and any page
    // whose offset is not 64-aligned reads back as zero on BH. At 32 B, odd device pages landed at
    // 32-aligned (not 64-aligned) offsets → read as 0 → Chan-merge saw [v,0,v,0,...] → variance
    // exactly halved (√2 output scale), independent of ring size. Round up to 64 to keep every
    // page 64-aligned.
    s.stick_bytes = ((num_groups * 4u + 63u) / 64u) * 64u;
    s.total_pages = s.is_local ? 0u : ring_size;
    s.page_size_bytes = s.stick_bytes;
    return s;
}

}  // namespace

ttnn::Tensor dit_fused_distributed_groupnorm(
    const ttnn::Tensor& input_tensor,
    int num_groups,
    float epsilon,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    ttnn::ccl::Topology topology,
    const std::optional<Tensor>& input_mask,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    bool use_welford,
    const std::optional<ttnn::Tensor>& persistent_output_buffer,
    std::optional<size_t> num_preferred_links,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id) {
    (void)cluster_width(mesh_device, cluster_axis);

    // Always launch the fused device op. Width-1 runs local PRE+POST (no fabric);
    // width>1 runs PRE → fabric AG → POST. Plain […,C] γ/β (not GN-packed).
    return ttnn::prim::dit_fused_distributed_groupnorm(
        input_tensor,
        num_groups,
        epsilon,
        cluster_axis,
        mesh_device,
        multi_device_global_semaphore,
        topology,
        input_mask,
        weight,
        bias,
        memory_config,
        std::move(compute_kernel_config),
        use_welford,
        persistent_output_buffer,
        num_preferred_links,
        subdevice_id);
}

std::optional<ttnn::Tensor> dit_fused_distributed_groupnorm_create_stats_buffer(
    const ttnn::Tensor& input_tensor,
    const uint32_t num_groups,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const uint32_t num_links) {
    (void)input_tensor;
    (void)num_links;
    const uint32_t ring_size = cluster_width(mesh_device, cluster_axis);
    auto sizing = make_sizing(num_groups, ring_size);
    if (sizing.is_local) {
        return std::nullopt;
    }

    // One ROW_MAJOR fp32 page per device; each page is one stick of (sum,sumsq,count)*G.
    const uint32_t floats_per_page = sizing.stick_bytes / sizeof(float);
    ttnn::Shape stats_shape({1u, 1u, sizing.total_pages, floats_per_page});
    MemoryConfig stats_mem{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
    tt::tt_metal::TensorSpec spec(
        stats_shape,
        tt::tt_metal::TensorLayout(DataType::FLOAT32, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), stats_mem));
    return tt::tt_metal::create_device_tensor(spec, &const_cast<MeshDevice&>(mesh_device));
}

}  // namespace ttnn::experimental

namespace ttnn::experimental::prim {

DitFusedDistributedGroupnormSizing compute_sizing(const DitFusedDistributedGroupnormParams& args, const Tensor& input) {
    (void)input;
    DitFusedDistributedGroupnormSizing s;
    s.num_groups = args.num_groups;
    s.is_local = (args.ring_size <= 1);
    // Per-group bf16 [mean, var] (2 * 2 B). Must match make_sizing: the stick is the per-device
    // DRAM page in the gathered stats buffer, so its size must be a multiple of
    // NOC_DRAM_READ_ALIGNMENT_BYTES (64 on Blackhole) or non-64-aligned pages read back as zero.
    s.stick_bytes = ((args.num_groups * 4u + 63u) / 64u) * 64u;
    s.total_pages = s.is_local ? 0u : args.ring_size;
    s.page_size_bytes = s.stick_bytes;
    return s;
}

}  // namespace ttnn::experimental::prim
