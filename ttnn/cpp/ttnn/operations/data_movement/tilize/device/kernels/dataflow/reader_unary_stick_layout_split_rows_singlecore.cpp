// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "cpp/ttnn/operations/data_movement/common/kernels/common.hpp"

using tt::data_movement::common::tt_memmove;

void kernel_main() {
    // Constexpr
    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_staging = tt::CBIndex::c_1;
    constexpr uint32_t tile_height = 32;

    // Compile-time args:
    //   0: stick_size          — byte width of one row; used by TensorAccessorArgs.
    //   1: dram_alignment      — 0 for non-BH / wide sticks; 64 for BH+UINT8 narrow sticks.
    //                            When non-zero, c_1 staging is used to avoid NOC misalignment.
    //   2+: TensorAccessorArgs fields.
    constexpr uint32_t dram_alignment = get_compile_time_arg_val(1);

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_sticks = get_arg_val<uint32_t>(1);
    const uint32_t num_tiles_per_block = get_arg_val<uint32_t>(3);
    const uint32_t block_width_size = get_arg_val<uint32_t>(4);
    const uint32_t num_full_blocks_in_row = get_arg_val<uint32_t>(5);
    const uint32_t start_stick_id = get_arg_val<uint32_t>(8);

    // TensorAccessorArgs now start at compile-time index 2 (indices 0 and 1 are
    // stick_size and dram_alignment respectively).
    constexpr auto src_tensor_args = TensorAccessorArgs<2>();

    const auto s = TensorAccessor(src_tensor_args, src_addr);

    Noc noc;
    CircularBuffer cb_in0(cb_id_in0);

    // When alignment staging is active (BH+UINT8 narrow sticks), reserve the staging CB once
    // and align the write pointer to dram_alignment.  The CB is never released — it acts as a
    // permanent scratch buffer for the lifetime of this kernel.
    uint32_t temp_addr = 0;
    if constexpr (dram_alignment > 0) {
        CircularBuffer cb_staging(cb_id_staging);
        cb_staging.reserve_back(1);
        uint32_t temp_addr_raw = cb_staging.get_write_ptr();
        // Round up to the next dram_alignment boundary so that the over-read from DRAM
        // never wraps across a misaligned L1 word.
        temp_addr = (temp_addr_raw + dram_alignment - 1) & ~(dram_alignment - 1);
        cb_staging.push_back(1);
    }

    uint32_t stick_ids[tile_height];
    uint32_t stick_offset = 0;

    auto read_tiles = [&](const uint32_t& num_tiles, const uint32_t& width_size) {
        cb_in0.reserve_back(num_tiles);
        uint32_t l1_write_addr = cb_in0.get_write_ptr();

        for (uint32_t k = 0; k < tile_height; k++) {
            if constexpr (dram_alignment > 0) {
                // Alignment-aware path: check whether the DRAM source offset and the L1
                // destination offset within a dram_alignment-sized block agree.  If they
                // differ, the NOC would read an over-sized block that overwrites adjacent L1,
                // so we redirect into the temp staging buffer and copy only the valid bytes.
                constexpr uint64_t dram_align_mask = ~static_cast<uint64_t>(dram_alignment - 1);
                constexpr uint64_t dram_align_offset_mask = static_cast<uint64_t>(dram_alignment - 1);

                uint64_t src_noc_addr = s.get_noc_addr(stick_ids[k]) + stick_offset;

                if ((src_noc_addr & dram_align_offset_mask) ==
                    (static_cast<uint64_t>(l1_write_addr) & dram_align_offset_mask)) {
                    // Same alignment offset — direct read into c_0 is safe.
                    CoreLocalMem<uint32_t> dst(l1_write_addr);
                    noc.async_read(
                        s,
                        dst,
                        width_size,
                        {.page_id = stick_ids[k], .offset_bytes = stick_offset},
                        {.offset_bytes = 0});
                    noc.async_read_barrier();
                } else {
                    // Misaligned: issue an aligned over-read into the staging buffer, then
                    // copy only the valid width_size bytes into c_0.
                    const uint64_t aligned_src = src_noc_addr & dram_align_mask;
                    const uint32_t src_offset_in_block = static_cast<uint32_t>(src_noc_addr & dram_align_offset_mask);

                    CoreLocalMem<uint32_t> temp_dst(temp_addr);
                    noc.async_read(
                        UnicastEndpoint{},
                        temp_dst,
                        width_size + dram_alignment,
                        {.noc_x = static_cast<uint32_t>(NOC_UNICAST_ADDR_X(aligned_src)),
                         .noc_y = static_cast<uint32_t>(NOC_UNICAST_ADDR_Y(aligned_src)),
                         .addr = static_cast<uint32_t>(NOC_LOCAL_ADDR_OFFSET(aligned_src))},
                        {.offset_bytes = 0});
                    noc.async_read_barrier();

                    // Copy from temp_addr + src_offset_in_block → l1_write_addr.
                    tt_memmove<false, false, true, 0>(noc, l1_write_addr, temp_addr + src_offset_in_block, width_size);
                }
            } else {
                // Fast path (non-BH or wide sticks): read all rows, then barrier once.
                CoreLocalMem<uint32_t> dst(l1_write_addr);
                noc.async_read(
                    s, dst, width_size, {.page_id = stick_ids[k], .offset_bytes = stick_offset}, {.offset_bytes = 0});
            }
            l1_write_addr += width_size;
        }

        stick_offset += width_size;

        if constexpr (dram_alignment == 0) {
            // All reads for this block were issued without per-row barriers; flush them now.
            noc.async_read_barrier();
        }

        cb_in0.push_back(num_tiles);
    };

    uint32_t stick_id = start_stick_id;
    for (uint32_t i = 0; i < num_sticks / tile_height; i++) {
        // Get Base IDs
        for (uint32_t j = 0; j < tile_height; j++) {
            stick_ids[j] = stick_id;
            stick_id++;
        }
        stick_offset = 0;

        for (uint32_t j = 0; j < num_full_blocks_in_row; j++) {
            read_tiles(num_tiles_per_block, block_width_size);
        }
    }
}
