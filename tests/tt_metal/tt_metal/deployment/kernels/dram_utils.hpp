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
// RANDOM mode the region is a fixed grid of grid_chunk_bytes-sized chunks and the grid indices are visited in
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

// Balanced Feistel network: a bijection on [0, 2^(2*half_width_bits)). Each input gives a different
// output, so each index comes out one time only. The key selects the shuffle order.
// half_width_bits==0 maps everything to 0 (used only when there is a single chunk).
static inline uint32_t dram_feistel_permute(uint32_t x, uint32_t half_width_bits, uint32_t key) {
    // One chunk: the only index is 0.
    if (half_width_bits == 0u) {
        return 0u;
    }

    // mask has the low half_width_bits bits set (half_width_bits=3 -> 0b111).
    const uint32_t mask = (1u << half_width_bits) - 1u;
    // Divide x into two halves of the same size: l = high half, r = low half.
    uint32_t l = (x >> half_width_bits) & mask;
    uint32_t r = x & mask;

    // 4 rounds. More rounds mix the bits more.
    for (uint32_t round = 0u; round < 4u; ++round) {
        // Mix the right half with the key and a different value for each round.
        // 0x9E3779B9 comes from the golden ratio. It spreads bits well.
        uint32_t mix = r ^ key ^ (round * 0x9E3779B9u);
        // Round function F: one xorshift step scrambles the bits.
        const uint32_t f = dram_xorshift32_step(mix);
        // Feistel round: the old right half becomes the new left half.
        // The new right half is the old left half XOR F, cut to half_width_bits bits.
        // A round can be undone (r = new_l, l = new_r ^ F(new_l)), so no two inputs give the same output.
        const uint32_t new_l = r;
        const uint32_t new_r = l ^ (f & mask);
        l = new_l;
        r = new_r;
    }

    // Join the two halves back into one index.
    return (l << half_width_bits) | r;
}

// One NOC transfer of a job.
struct DramChunkStep {
    uint32_t offset;          // Start of the transfer, in bytes from the start of the job region.
    uint32_t transfer_bytes;  // Size of the transfer.
};

// Struct to store how far a job has got through its region. Each worker core has one walker per job.
struct DramChunkWalker {
    uint32_t mode;         // DRAM_ACCESS_MODE_SEQUENTIAL or DRAM_ACCESS_MODE_RANDOM.
    uint32_t total_bytes;  // Size of the job region.

    // Sequential traversal state.
    uint32_t next_offset;  // Where the next transfer starts.
    uint32_t rng_state;    // Random number state for transfer length modes 1 and 2.

    // Random traversal state.
    uint32_t grid_chunk_bytes;      // Size of one grid chunk, in bytes.
    uint32_t total_chunks;          // Number of chunks in the region.
    uint32_t emitted;               // Number of chunks that the walker has already returned.
    uint32_t perm_cursor;           // Next input to dram_feistel_permute.
    uint32_t perm_half_width_bits;  // Width of each Feistel half, in bits. Shuffle range = 2^(2 * this).
    uint32_t perm_key;              // Feistel key for this job and this pass.
};

static inline void dram_chunk_walker_init(DramChunkWalker& w, const DramTestParameters& p) {
    w.mode = p.access_mode;
    w.total_bytes = p.total_bytes;

    // Mix the pass number into the seed, so each pass gets different transfer sizes.
    w.rng_state = p.seed ^ p.pass_index;
    // If xorshift starts at 0, it returns 0 for ever. Prevent that.
    if (w.rng_state == 0u) {
        w.rng_state = 1u;
    }
    // Sequential mode starts at the start of the region.
    w.next_offset = 0u;

    // Fixed grid used by random mode. grid_chunk_bytes is chunk_bytes, capped at max_burst_len, then rounded
    // down to a NOC word (64 bytes, minimum one NOC word). Thus, every full chunk is a valid NOC burst.
    // The last chunk is smaller than the others when the region is not a multiple of grid_chunk_bytes.
    w.grid_chunk_bytes = dram_effective_chunk_bytes(p.chunk_bytes, p.max_burst_len);
    // Number of chunks, rounded up. The extra bytes get one more (smaller) chunk.
    w.total_chunks = dram_total_chunks(w.total_bytes, w.grid_chunk_bytes);
    // Random mode starts with no chunks returned and the Feistel input at 0.
    w.emitted = 0u;
    w.perm_cursor = 0u;
    // Smallest even bit count that holds every chunk index, divided into two halves of the same size.
    // The Feistel range is at least total_chunks and less than 4 * total_chunks.
    w.perm_half_width_bits = dram_ceil_log2_even(w.total_chunks) / 2u;
    // Key from the seed and the pass number, so each pass visits the chunks in a different order.
    // 0x85EBCA6B is a MurmurHash3 mixing constant. "| 1u" makes sure that the key is never 0.
    w.perm_key = (p.seed ^ (p.pass_index * 0x85EBCA6Bu)) | 1u;
}

static inline bool dram_chunk_walker_has_next(const DramChunkWalker& w) {
    // Random mode: continue until every chunk is returned one time.
    if (w.mode == DRAM_ACCESS_MODE_RANDOM) {
        return w.emitted < w.total_chunks;
    }
    // Sequential mode: continue until the next offset gets to the end of the region.
    return w.next_offset < w.total_bytes;
}

static inline DramChunkStep dram_chunk_walker_next(DramChunkWalker& w, const DramTestParameters& p) {
    DramChunkStep step{};

    if (w.mode == DRAM_ACCESS_MODE_RANDOM) {
        // Shuffle perm_cursor to get a chunk index.
        uint32_t logical = dram_feistel_permute(w.perm_cursor, w.perm_half_width_bits, w.perm_key);
        w.perm_cursor++;
        // The Feistel range can be larger than total_chunks,
        // so try the next cursor value when the index is out of range
        // Each index in range comes out exactly one time.
        while (logical >= w.total_chunks) {
            logical = dram_feistel_permute(w.perm_cursor, w.perm_half_width_bits, w.perm_key);
            w.perm_cursor++;
        }

        // Convert the chunk index to a byte offset in the region.
        const uint32_t offset = logical * w.grid_chunk_bytes;
        // Each transfer is one full chunk.
        uint32_t transfer_bytes = w.grid_chunk_bytes;
        // Only the last chunk can be cut short at the end of the region.
        if (offset + transfer_bytes > w.total_bytes) {
            transfer_bytes = w.total_bytes - offset;
        }

        step.offset = offset;
        step.transfer_bytes = transfer_bytes;
        w.emitted++;
        // Random mode does not call dram_choose_transfer_len, so the transfer length mode has no effect.
        return step;
    }

    // Sequential mode: bytes from the current position to the end of the region.
    const uint32_t remaining_bytes = p.total_bytes - w.next_offset;
    // Size depends on the transfer length mode (0 = fixed, 1 = random, 2 = tile-like).
    // The size is never more than remaining_bytes. Modes 1 and 2 change rng_state.
    const uint32_t transfer_bytes = dram_choose_transfer_len(p, remaining_bytes, w.rng_state);
    step.offset = w.next_offset;
    step.transfer_bytes = transfer_bytes;
    // Move forward by the transfer size, so the transfers cover the region with no gaps and no overlaps.
    w.next_offset += transfer_bytes;
    return step;
}

#endif /* _DRAM_TEST_UTILS_H */
