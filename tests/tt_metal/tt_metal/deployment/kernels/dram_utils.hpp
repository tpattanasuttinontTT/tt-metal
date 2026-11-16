// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef _DRAM_TEST_UTILS_H
#define _DRAM_TEST_UTILS_H

#include <stdint.h>
#include "noc_parameters.h"
#include "dev_mem_map.h"
#include "common_dram.hpp"

static_assert(NOC_WORD_BYTES == DRAM_TEST_NOC_WORD_BYTES, "NOC word size mismatch");
static_assert(MEM_DRAM_SIZE == DRAM_TEST_MAX_BANK_BYTES, "DRAM size mismatch");

static inline uint32_t dram_xorshift32_step(uint32_t& state) {
    state ^= (state << 13);
    state ^= (state >> 17);
    state ^= (state << 5);
    return state;
}

static inline uint32_t dram_choose_transfer_len(
    const DramTestParameters& p, uint32_t remaining_bytes, uint32_t& rng_state) {
    uint32_t len = p.chunk_bytes;

    if (p.max_burst_len != 0u && p.max_burst_len < len) {
        len = p.max_burst_len;
    }

    if (remaining_bytes < len) {
        len = remaining_bytes;
    }

    if (p.transfer_len_mode == 1u) {
        uint32_t r = dram_xorshift32_step(rng_state);
        r = ~(r | (r << 5) | (r << 8) | (r << 11));

        if (len != 0u) {
            len = (r % len) + 1u;
        }
    } else if (p.transfer_len_mode == 2u) {
        uint32_t r = dram_xorshift32_step(rng_state);

        uint32_t tile_size_bits = r & 0x1Fu;
        uint32_t header_size_bits = (r >> 5) & 0x3u;

        uint32_t tile_size = tile_size_bits * 512u;

        uint32_t header_size = 0u;
        if (header_size_bits == 1u) {
            header_size = NOC_WORD_BYTES;
        } else if (header_size_bits == 2u) {
            header_size = NOC_WORD_BYTES;
        } else if (header_size_bits == 3u) {
            header_size = 2u * NOC_WORD_BYTES;
        }

        uint32_t candidate = tile_size + header_size;
        if (candidate < NOC_WORD_BYTES) {
            candidate = NOC_WORD_BYTES;
        }

        if (candidate < len) {
            len = candidate;
        }
    }

    len = (len + NOC_WORD_BYTES - 1u) & ~(NOC_WORD_BYTES - 1u);

    if (len > remaining_bytes) {
        len = remaining_bytes & ~(NOC_WORD_BYTES - 1u);
        if (len == 0u) {
            len = remaining_bytes;
        }
    }

    return len;
}

// ============================================================
// Chunk traversal order (sequential vs random access)
// ============================================================
// A job walks its region [0, total_bytes) one chunk at a time. In SEQUENTIAL mode the offsets march
// upward (and may use variable transfer lengths via dram_choose_transfer_len, exactly as before). In
// RANDOM mode the region is a fixed grid of chunk_eff-sized chunks and the grid indices are visited in
// a shuffled, full-coverage (every index exactly once) order, so DRAM addresses jump around.
//
// The shuffle is a seeded balanced Feistel bijection over [0, 2^nbits) with rejection sampling for
// indices >= total_chunks (period < 4*total_chunks, so enumeration costs at most ~4*total_chunks
// steps). No permutation array is stored in L1.

static inline uint32_t dram_ceil_log2_even(uint32_t n) {
    // Smallest EVEN bit count whose power-of-two value is >= n (even so the Feistel halves are equal).
    uint32_t bits = 0u;
    uint32_t value = 1u;
    while (value < n) {
        value <<= 1u;
        bits++;
    }
    if (bits & 1u) {
        bits++;
    }
    return bits;
}

// Balanced Feistel network: a bijection on [0, 2^(2*half_bits)). half_bits==0 maps everything to 0
// (used only when there is a single chunk).
static inline uint32_t dram_feistel_permute(uint32_t x, uint32_t half_bits, uint32_t key) {
    if (half_bits == 0u) {
        return 0u;
    }

    const uint32_t mask = (1u << half_bits) - 1u;
    uint32_t l = (x >> half_bits) & mask;
    uint32_t r = x & mask;

    for (uint32_t round = 0u; round < 4u; ++round) {
        uint32_t mix = r ^ key ^ (round * 0x9E3779B9u);
        const uint32_t f = dram_xorshift32_step(mix);
        const uint32_t new_l = r;
        const uint32_t new_r = l ^ (f & mask);
        l = new_l;
        r = new_r;
    }

    return (l << half_bits) | r;
}

struct DramChunkStep {
    uint32_t offset;
    uint32_t transfer_bytes;
};

struct DramChunkWalker {
    uint32_t mode;
    uint32_t total_bytes;

    // Sequential traversal state.
    uint32_t next_offset;
    uint32_t rng_state;

    // Random traversal state.
    uint32_t chunk_eff;
    uint32_t total_chunks;
    uint32_t emitted;
    uint32_t perm_cursor;
    uint32_t half_bits;
    uint32_t perm_key;
};

static inline void dram_chunk_walker_init(DramChunkWalker& w, const DramTestParameters& p) {
    w.mode = p.access_mode;
    w.total_bytes = p.total_bytes;

    w.rng_state = p.seed ^ p.pass_index;
    if (w.rng_state == 0u) {
        w.rng_state = 1u;
    }
    w.next_offset = 0u;

    // Fixed grid used by random mode. chunk_eff is clamped by max_burst_len and NOC-word aligned so
    // every full chunk is a valid NOC burst; the last chunk absorbs the (NOC-word-aligned) remainder.
    w.chunk_eff = dram_effective_chunk_bytes(p.chunk_bytes, p.max_burst_len);
    w.total_chunks = dram_total_chunks(w.total_bytes, w.chunk_eff);
    w.emitted = 0u;
    w.perm_cursor = 0u;
    w.half_bits = dram_ceil_log2_even(w.total_chunks) / 2u;
    w.perm_key = (p.seed ^ (p.pass_index * 0x85EBCA6Bu)) | 1u;
}

static inline bool dram_chunk_walker_has_next(const DramChunkWalker& w) {
    if (w.mode == DRAM_ACCESS_MODE_RANDOM) {
        return w.emitted < w.total_chunks;
    }
    return w.next_offset < w.total_bytes;
}

static inline DramChunkStep dram_chunk_walker_next(DramChunkWalker& w, const DramTestParameters& p) {
    DramChunkStep step{};

    if (w.mode == DRAM_ACCESS_MODE_RANDOM) {
        uint32_t logical = 0u;
        do {
            logical = dram_feistel_permute(w.perm_cursor, w.half_bits, w.perm_key);
            w.perm_cursor++;
        } while (logical >= w.total_chunks);

        const uint32_t offset = logical * w.chunk_eff;
        uint32_t transfer_bytes = w.chunk_eff;
        if (offset + transfer_bytes > w.total_bytes) {
            transfer_bytes = w.total_bytes - offset;
        }

        step.offset = offset;
        step.transfer_bytes = transfer_bytes;
        w.emitted++;
        return step;
    }

    const uint32_t remaining_bytes = p.total_bytes - w.next_offset;
    const uint32_t transfer_bytes = dram_choose_transfer_len(p, remaining_bytes, w.rng_state);
    step.offset = w.next_offset;
    step.transfer_bytes = transfer_bytes;
    w.next_offset += transfer_bytes;
    return step;
}

#endif /* _DRAM_TEST_UTILS_H */
