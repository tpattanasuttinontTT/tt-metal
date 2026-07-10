// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "block_cyclic_layout.hpp"  // ttnn::prim::BlockCyclicLayout (shared)
#include <optional>

namespace ttnn::prim {

// Sparse MLA prefill (DeepSeek DSA).
struct SparseSDPAParams {
    float scale = 1.0f;  // compile-time (folded into the program hash)
    uint32_t v_dim;      // width of V (= leading v_dim cols of the K_DIM-wide KV cache); the output width
    uint32_t k_chunk_size = 128;
    DeviceComputeKernelConfig compute_kernel_config;
    // Indexed KV cache: when set, kv is a [B,1,T,K_DIM] shared cache and this selects the batch slot to
    // attend to (the gather page ids are offset by cache_batch_idx * T). It is a DYNAMIC runtime arg
    // (excluded from the program hash, re-applied every dispatch), so changing it does NOT recompile.
    std::optional<uint32_t> cache_batch_idx = std::nullopt;
    bool has_indexed_kv_cache() const { return cache_batch_idx.has_value(); }
    // Block-cyclic index remap (see BlockCyclicLayout). Fully compile-time: the enable bit, sp, chunk_local,
    // and the T-derived BC_SHARD_STRIDE_GAP (= T/sp - chunk_local) are all program defines, and the cache length T is
    // folded into the hash for this path — so a different cache size is a distinct program (no runtime remap
    // arg). Assumes a consistent-size preallocated cache (per-size recompile, once, is acceptable).
    std::optional<BlockCyclicLayout> block_cyclic = std::nullopt;
    bool has_block_cyclic() const { return block_cyclic.has_value(); }
};

struct SparseSDPAInputs {
    Tensor q;   // [1, H, S, K_DIM] bf16/fp8_e4m3 ROW_MAJOR  (K_DIM = head dim, e.g. 576)
    Tensor kv;  // [1, 1, T, K_DIM] bf16/fp8_e4m3 ROW_MAJOR  (or [B,1,T,K_DIM] when indexed; may be ND-sharded DRAM)
    Tensor indices;  // [1, 1, S, TOPK] uint32 ROW_MAJOR  (0xFFFFFFFF = masked)
};

}  // namespace ttnn::prim
