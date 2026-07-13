// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// sparse_sdpa writer: per token, drain the untilized [H, V_DIM] row-major block from compute (Sqt*vDHt
// tile-sized pages, row-major bytes inside) and write each head-row to the ROW-MAJOR [1,H,S,V_DIM] output.
// Also builds the three persistent compute-input tiles once at startup (the reader is busier): the reduce
// identity scaler, the col-identity (row-sum finalize), and the all-(-inf) mask tile.
// Uses the object-based NoC + circular-buffer APIs.

#include <stdint.h>
#include <tt-metalium/constants.hpp>  // tt::constants::TILE_WIDTH
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp"  // generate_bcast_col_scalar
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "sparse_sdpa_gather.hpp"  // dual-NoC trid-ring gather helper (shared with the reader)

constexpr uint32_t one_bf16_packed = 0x3F803F80u;  // bf16(1.0) double-packed; generate_bcast_col_scalar uses >>16
constexpr uint16_t neg_inf_bf16 = 0xFF80u;

void kernel_main() {
    constexpr uint32_t H = get_compile_time_arg_val(0);
    constexpr uint32_t S = get_compile_time_arg_val(1);
    constexpr uint32_t vDHt = get_compile_time_arg_val(2);
    // CB ids — passed by the factory (the SparseCB enum is the single source); order must match the
    // factory's writer CB-arg block (compile args 3..6).
    constexpr uint32_t cb_out_rm = get_compile_time_arg_val(3);
    constexpr uint32_t cb_scale = get_compile_time_arg_val(4);
    constexpr uint32_t cb_col_identity = get_compile_time_arg_val(5);
    constexpr uint32_t cb_neginf = get_compile_time_arg_val(6);
    constexpr uint32_t out_elem_bytes = get_compile_time_arg_val(7);  // bf16 (from the output tensor's dtype)
    constexpr bool block_cyclic = get_compile_time_arg_val(8) != 0;
    constexpr uint32_t bc_chunk_local = get_compile_time_arg_val(9);
    constexpr uint32_t bc_sp = get_compile_time_arg_val(10);
    constexpr uint32_t bc_shard_stride_gap = get_compile_time_arg_val(11);
    constexpr uint32_t bc_slab_stride_gap = get_compile_time_arg_val(12);
    constexpr bool scaled_kv = get_compile_time_arg_val(13) != 0;
    constexpr uint32_t k_dim = get_compile_time_arg_val(14);
    constexpr uint32_t kv_elem_bytes = get_compile_time_arg_val(15);
    constexpr uint32_t cb_k_rm = get_compile_time_arg_val(16);
    constexpr uint32_t cb_idx = get_compile_time_arg_val(17);
    constexpr uint32_t cb_kreq = get_compile_time_arg_val(18);
    constexpr uint32_t cb_kack = get_compile_time_arg_val(19);
    constexpr uint32_t packed_row_bytes = get_compile_time_arg_val(20);
    constexpr auto out_args = TensorAccessorArgs<21, 0>();
    // kv carries a RUNTIME tensor shape (T dim in common runtime args), so its accessor spans both arg streams.
    constexpr auto kv_args =
        TensorAccessorArgs<out_args.next_compile_time_args_offset(), out_args.next_common_runtime_args_offset()>();

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t tok_start = get_arg_val<uint32_t>(1);
    const uint32_t tok_count = get_arg_val<uint32_t>(2);
    const uint32_t kv_addr = get_arg_val<uint32_t>(3);  // dual-NoC K-half gather
    // Indexed KV cache: page offset (cache_batch_idx * T) selecting the cache's batch slot; 0 if not indexed.
    const uint32_t kv_batch_page_offset = get_arg_val<uint32_t>(4);

    constexpr uint32_t row_bytes = vDHt * tt::constants::TILE_WIDTH * out_elem_bytes;  // V_DIM bytes per head-row
    constexpr uint32_t block_tiles = (H / tt::constants::TILE_HEIGHT) * vDHt;          // Sqt*vDHt: the [H,V_DIM] block

    Noc noc;
    experimental::CB out_cb(cb_out_rm);
    const auto out = TensorAccessor(out_args, out_addr);
    // Dual-NoC K gather: the writer co-gathers half of each K chunk on ITS NoC into the shared cb_k_rm L1
    // (the reader owns the reserve/push; we fill rows [0,half) at the same write ptr). This spreads the
    // K response-data load over both NoC links. Indices live in the reader's shared cb_idx
    // scratch; the kv accessor + row size mirror the reader.
    constexpr uint32_t k_row_bytes = k_dim * kv_elem_bytes;
    constexpr uint32_t latent_row_bytes = vDHt * tt::constants::TILE_WIDTH;
    constexpr uint32_t scale_blocks = latent_row_bytes / 128;
    constexpr uint32_t packed_words = packed_row_bytes / sizeof(uint32_t);
    constexpr uint32_t latent_words = latent_row_bytes / sizeof(uint32_t);
    const auto kv = TensorAccessor(kv_args, kv_addr);
    experimental::CB k_cb(cb_k_rm), idx_cb(cb_idx), kreq_cb(cb_kreq), kack_cb(cb_kack);
    volatile tt_l1_ptr uint32_t* idx_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_cb.get_write_ptr());

    // --- persistent compute-input tiles (built once; the reader is busier with the K gather) ---
    // Reduce identity scaler (value 1.0; the softmax scale is applied in compute's exp).
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_scale,
        ckernel::PoolType::MAX,
        ckernel::ReduceDim::REDUCE_ROW,
        /*reduce_factor=*/1>();

    // Col-identity (column 0 = 1.0): compute matmul-reduces the partial row-sum against it to finalize the
    // within-tile reduction in normalize_row_streaming.
    generate_bcast_col_scalar(CircularBuffer(cb_col_identity), one_bf16_packed);

    // All-(-inf) tile (row 0; compute bcast-adds it to fully-masked key tiles). Only row 0 matters for the
    // row-broadcast add: cols 0-15 -> face0 row0 (offset c), cols 16-31 -> face1 row0 (offset 256+c).
    {
        experimental::CB neginf_cb(cb_neginf);
        neginf_cb.reserve_back(1);
        volatile tt_l1_ptr uint16_t* p = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(neginf_cb.get_write_ptr());
        for (uint32_t c = 0; c < 16; ++c) {
            p[c] = neg_inf_bf16;        // face0 row 0 (cols 0-15)
            p[256 + c] = neg_inf_bf16;  // face1 row 0 (cols 16-31)
        }
        neginf_cb.push_back(1);
    }

    for (uint32_t tok = tok_start; tok < tok_start + tok_count; ++tok) {
        {
            // K-gather phase: process this token's active chunks (reader signals per chunk; is_last ends the
            // token). Gather rows [0,half) on the writer's NoC into the reserved cb_k_rm L1, then ack.
            bool last = false;
            while (!last) {
                kreq_cb.wait_front(1);
                uint32_t base, split, dst_l1, scale_dst_l1 = 0;
                {
                    volatile tt_l1_ptr uint32_t* rq =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kreq_cb.get_read_ptr());
                    base = rq[0];
                    split = rq[1];
                    if constexpr (scaled_kv) {
                        last = rq[3] != 0;
                        dst_l1 = rq[4];
                        scale_dst_l1 = rq[5];
                    } else {
                        last = rq[2] != 0;
                        dst_l1 = rq[3];
                    }
                }
                kreq_cb.pop_front(1);
                if constexpr (scaled_kv) {
                    sparse_sdpa::trid_ring_gather<
                        block_cyclic,
                        bc_chunk_local,
                        bc_sp,
                        bc_shard_stride_gap,
                        bc_slab_stride_gap,
                        sparse_sdpa::SCALED_K_TRID_RING>(
                        noc, kv, dst_l1, idx_ptr, base, 0, split, packed_row_bytes, kv_batch_page_offset);
                    volatile tt_l1_ptr uint32_t* scale_tiles =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scale_dst_l1);
                    volatile tt_l1_ptr uint32_t* packed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_l1);
                    for (uint32_t p = 0; p < split; ++p) {
                        const uint32_t tile_row_offset = p < 16 ? p * 16 : 512 + (p - 16) * 16;
                        volatile tt_l1_ptr uint32_t* row_scales = packed + p * packed_words + latent_words;
                        for (uint32_t block = 0; block < scale_blocks; ++block) {
                            scale_tiles[block * 1024 + tile_row_offset] = row_scales[block];
                        }
                    }
                } else {
                    sparse_sdpa::trid_ring_gather<
                        block_cyclic,
                        bc_chunk_local,
                        bc_sp,
                        bc_shard_stride_gap,
                        bc_slab_stride_gap,
                        sparse_sdpa::K_TRID_RING>(
                        noc, kv, dst_l1, idx_ptr, base, 0, split, k_row_bytes, kv_batch_page_offset);
                }
                kack_cb.reserve_back(1);
                kack_cb.push_back(1);
            }
        }
        out_cb.wait_front(block_tiles);  // one untilized [H, V_DIM] block
        for (uint32_t h = 0; h < H; ++h) {
            // row h (head h) at CB read-ptr byte offset h*row_bytes; DRAM page = h*S + tok.
            noc.async_write(out_cb, out, row_bytes, {.offset_bytes = h * row_bytes}, {.page_id = h * S + tok});
        }
        noc.async_write_barrier();
        out_cb.pop_front(block_tiles);
    }
}
