# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Universal I/O tests for ttnn.roll — comprehensive sharded matrix.

Covers:
  - ROW_MAJOR interleaved (DRAM/L1) — backward compatibility
  - TILE interleaved (DRAM/L1)
  - HEIGHT / WIDTH / BLOCK sharded, ROW_MAJOR (native gather kernel)
  - HEIGHT / WIDTH / BLOCK sharded, TILE tile-aligned (native whole-tile gather)
  - HEIGHT / WIDTH / BLOCK sharded, TILE non-tile-aligned (sharded untilize→roll→tilize)
  - Multi-dim rolls, higher-dim (batch/channel) rolls, last-dim within-row rolls
  - DRAM-sharded ROW_MAJOR: full-shard L1 staging (read DRAM→L1, assemble, write L1→DRAM)
  - DRAM-sharded TILE: per-tile NOC read/write (tile-size naturally DRAM-aligned)
  - Program-cache hash correctness: distinct shifts produce distinct programs
  - Optional output memory_config parameter (sharded → interleaved and vice versa)
  - COL_MAJOR shard orientation (HEIGHT and BLOCK)
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

_PCC = 0.9999


# ─── shard-config helpers ─────────────────────────────────────────────────────


def _explicit_height_shard(device, ncores, sh, sw):
    compute_grid = device.compute_with_storage_grid_size()
    if ncores > compute_grid.x * compute_grid.y:
        pytest.skip(f"Device has {compute_grid.x * compute_grid.y} cores, test needs {ncores}")
    spec = ttnn.ShardSpec(
        ttnn.num_cores_to_corerangeset(ncores, compute_grid, True),
        (sh, sw),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)


def _explicit_width_shard(device, ncores, sh, sw):
    compute_grid = device.compute_with_storage_grid_size()
    if ncores > compute_grid.x * compute_grid.y:
        pytest.skip(f"Device has {compute_grid.x * compute_grid.y} cores, test needs {ncores}")
    spec = ttnn.ShardSpec(
        ttnn.num_cores_to_corerangeset(ncores, compute_grid, True),
        (sh, sw),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, spec)


def _explicit_block_shard(device, grid_y, grid_x, sh, sw):
    compute_grid = device.compute_with_storage_grid_size()
    if grid_y > compute_grid.y or grid_x > compute_grid.x:
        pytest.skip(f"Device grid ({compute_grid.y}x{compute_grid.x}) too small for {grid_y}x{grid_x}")
    spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))}),
        (sh, sw),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, spec)


# ─── runner ───────────────────────────────────────────────────────────────────


def run_roll(device, torch_input, layout, mem_config, shifts, dims):
    ttnn_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mem_config
    )
    ttnn_output = ttnn.roll(ttnn_input, list(shifts), list(dims))
    got = ttnn.to_torch(ttnn_output.cpu())
    ref = torch.roll(torch_input, list(shifts), list(dims))
    assert_with_pcc(ref.float(), got.float(), _PCC)


# ─── DRAM / L1 interleaved — backward compatibility ──────────────────────────


@pytest.mark.parametrize(
    "shape,shifts,dims,layout,mem_config",
    [
        ([1, 1, 4, 8], [2], [3], ttnn.ROW_MAJOR_LAYOUT, ttnn.DRAM_MEMORY_CONFIG),
        ([6, 4, 5, 1], [1, -2], [0, 2], ttnn.ROW_MAJOR_LAYOUT, ttnn.DRAM_MEMORY_CONFIG),
        ([4, 4, 8, 8], [3], [1], ttnn.ROW_MAJOR_LAYOUT, ttnn.DRAM_MEMORY_CONFIG),
        ([1, 1, 32, 64], [16], [3], ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG),
        ([2, 1, 64, 64], [32, -32], [2, 3], ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG),
        ([1, 1, 4, 8], [2], [3], ttnn.ROW_MAJOR_LAYOUT, ttnn.L1_MEMORY_CONFIG),
        ([1, 2, 64, 64], [32, -16], [2, 3], ttnn.TILE_LAYOUT, ttnn.L1_MEMORY_CONFIG),
    ],
)
def test_roll_interleaved(device, shape, shifts, dims, layout, mem_config):
    torch.manual_seed(0)
    run_roll(device, torch.randn(shape, dtype=torch.bfloat16), layout, mem_config, shifts, dims)


# ─── HEIGHT_SHARDED + ROW_MAJOR (native gather kernel) ───────────────────────


@pytest.mark.parametrize(
    "shape,ncores,sh,sw,shifts,dims",
    [
        # last-dim within-row rotation
        ([1, 1, 4, 8], 4, 1, 8, [2], [3]),
        ([1, 1, 8, 16], 4, 2, 16, [4], [3]),
        ([1, 1, 8, 16], 4, 2, 16, [7], [3]),  # non-power-of-2 shift
        # height-dim page permutation
        ([1, 1, 8, 16], 4, 2, 16, [2], [2]),
        ([1, 1, 16, 8], 4, 4, 8, [3], [2]),
        # batch-dim roll
        ([2, 2, 8, 8], 4, 8, 8, [1], [0]),  # total_rows=32, sh=32//4=8
        ([4, 1, 4, 8], 4, 4, 8, [2], [0]),
        # multi-dim roll
        ([1, 1, 8, 16], 4, 2, 16, [2, 4], [2, 3]),
        # negative shift
        ([1, 1, 8, 16], 4, 2, 16, [-3], [3]),
    ],
)
def test_roll_height_sharded_row_major(device, shape, ncores, sh, sw, shifts, dims):
    torch.manual_seed(1)
    mem_config = _explicit_height_shard(device, ncores, sh, sw)
    run_roll(device, torch.randn(shape, dtype=torch.bfloat16), ttnn.ROW_MAJOR_LAYOUT, mem_config, shifts, dims)


# ─── HEIGHT_SHARDED + TILE, tile-aligned shifts (native whole-tile gather) ───


@pytest.mark.parametrize(
    "shape,ncores,sh,sw,shifts,dims",
    [
        ([1, 1, 64, 32], 2, 32, 32, [32], [3]),
        ([1, 1, 64, 64], 2, 32, 64, [32], [2]),
        ([1, 1, 128, 64], 4, 32, 64, [64], [2]),
        ([1, 1, 64, 128], 2, 32, 128, [64], [3]),
        ([1, 1, 64, 64], 2, 32, 64, [32, 32], [2, 3]),  # multi-dim tile-aligned
    ],
)
def test_roll_height_sharded_tile_aligned(device, shape, ncores, sh, sw, shifts, dims):
    torch.manual_seed(2)
    mem_config = _explicit_height_shard(device, ncores, sh, sw)
    run_roll(device, torch.randn(shape, dtype=torch.bfloat16), ttnn.TILE_LAYOUT, mem_config, shifts, dims)


# ─── HEIGHT_SHARDED + TILE, non-tile-aligned (sharded untilize→roll→tilize) ──


@pytest.mark.parametrize(
    "shape,ncores,sh,sw,shifts,dims",
    [
        ([1, 1, 64, 64], 2, 32, 64, [5], [3]),
        ([1, 1, 64, 64], 2, 32, 64, [7], [2]),
        ([1, 1, 128, 64], 4, 32, 64, [13], [3]),
    ],
)
def test_roll_height_sharded_tile_non_aligned(device, shape, ncores, sh, sw, shifts, dims):
    torch.manual_seed(3)
    mem_config = _explicit_height_shard(device, ncores, sh, sw)
    run_roll(device, torch.randn(shape, dtype=torch.bfloat16), ttnn.TILE_LAYOUT, mem_config, shifts, dims)


# ─── WIDTH_SHARDED + ROW_MAJOR (native gather kernel, cross-shard boundary) ──


@pytest.mark.parametrize(
    "shape,ncores,sh,sw,shifts,dims",
    [
        # cross-shard last-dim roll
        ([1, 1, 1, 32], 2, 1, 16, [8], [3]),
        ([1, 1, 1, 64], 4, 1, 16, [16], [3]),
        ([1, 1, 1, 32], 2, 1, 16, [7], [3]),  # non-power-of-2
        # height-dim page permutation
        ([1, 1, 4, 32], 2, 4, 16, [2], [2]),
        ([1, 1, 8, 32], 2, 8, 16, [3], [2]),
        # negative shift
        ([1, 1, 1, 64], 4, 1, 16, [-5], [3]),
    ],
)
def test_roll_width_sharded_row_major(device, shape, ncores, sh, sw, shifts, dims):
    torch.manual_seed(4)
    mem_config = _explicit_width_shard(device, ncores, sh, sw)
    run_roll(device, torch.randn(shape, dtype=torch.bfloat16), ttnn.ROW_MAJOR_LAYOUT, mem_config, shifts, dims)


# ─── WIDTH_SHARDED + TILE, tile-aligned (native whole-tile gather) ────────────


@pytest.mark.parametrize(
    "shape,ncores,sh,sw,shifts,dims",
    [
        ([1, 1, 32, 64], 2, 32, 32, [32], [3]),
        ([1, 1, 32, 128], 4, 32, 32, [32], [3]),
        ([1, 1, 64, 128], 4, 64, 32, [64], [3]),
    ],
)
def test_roll_width_sharded_tile_aligned(device, shape, ncores, sh, sw, shifts, dims):
    torch.manual_seed(5)
    mem_config = _explicit_width_shard(device, ncores, sh, sw)
    run_roll(device, torch.randn(shape, dtype=torch.bfloat16), ttnn.TILE_LAYOUT, mem_config, shifts, dims)


# ─── WIDTH_SHARDED + TILE, non-tile-aligned (sharded untilize→roll→tilize) ───


@pytest.mark.parametrize(
    "shape,ncores,sh,sw,shifts,dims",
    [
        ([1, 1, 32, 128], 4, 32, 32, [13], [3]),
        ([1, 1, 32, 128], 4, 32, 32, [7], [3]),
    ],
)
def test_roll_width_sharded_tile_non_aligned(device, shape, ncores, sh, sw, shifts, dims):
    torch.manual_seed(6)
    mem_config = _explicit_width_shard(device, ncores, sh, sw)
    run_roll(device, torch.randn(shape, dtype=torch.bfloat16), ttnn.TILE_LAYOUT, mem_config, shifts, dims)


# ─── BLOCK_SHARDED + ROW_MAJOR (native gather kernel) ────────────────────────


@pytest.mark.parametrize(
    "shape,grid_y,grid_x,sh,sw,shifts,dims",
    [
        # last-dim within-row rotation
        ([1, 1, 4, 8], 2, 2, 2, 4, [2], [3]),
        ([1, 1, 8, 16], 2, 2, 4, 8, [4], [3]),
        ([1, 1, 8, 16], 2, 2, 4, 8, [3], [3]),  # non-power-of-2
        # height-dim roll
        ([1, 1, 8, 16], 2, 2, 4, 8, [2], [2]),
        ([1, 1, 16, 16], 2, 2, 8, 8, [4], [2]),
        # negative shift
        ([1, 1, 8, 16], 2, 2, 4, 8, [-3], [3]),
    ],
)
def test_roll_block_sharded_row_major(device, shape, grid_y, grid_x, sh, sw, shifts, dims):
    torch.manual_seed(7)
    mem_config = _explicit_block_shard(device, grid_y, grid_x, sh, sw)
    run_roll(device, torch.randn(shape, dtype=torch.bfloat16), ttnn.ROW_MAJOR_LAYOUT, mem_config, shifts, dims)


# ─── BLOCK_SHARDED + TILE, tile-aligned (native whole-tile gather) ────────────


@pytest.mark.parametrize(
    "shape,grid_y,grid_x,sh,sw,shifts,dims",
    [
        ([1, 1, 64, 64], 2, 2, 32, 32, [32], [3]),
        ([1, 1, 64, 128], 2, 2, 32, 64, [32], [2]),
        ([1, 1, 128, 64], 2, 2, 64, 32, [64], [2]),
        ([1, 1, 64, 128], 2, 2, 32, 64, [64], [3]),
        ([1, 1, 64, 64], 2, 2, 32, 32, [32, 32], [2, 3]),  # multi-dim tile-aligned
    ],
)
def test_roll_block_sharded_tile_aligned(device, shape, grid_y, grid_x, sh, sw, shifts, dims):
    torch.manual_seed(8)
    mem_config = _explicit_block_shard(device, grid_y, grid_x, sh, sw)
    run_roll(device, torch.randn(shape, dtype=torch.bfloat16), ttnn.TILE_LAYOUT, mem_config, shifts, dims)


# ─── BLOCK_SHARDED + TILE, non-tile-aligned (sharded untilize→roll→tilize) ───


@pytest.mark.parametrize(
    "shape,grid_y,grid_x,sh,sw,shifts,dims",
    [
        ([1, 1, 64, 64], 2, 2, 32, 32, [5], [3]),
        ([1, 1, 64, 64], 2, 2, 32, 32, [7], [2]),
        ([1, 1, 64, 128], 2, 2, 32, 64, [13], [3]),
    ],
)
def test_roll_block_sharded_tile_non_aligned(device, shape, grid_y, grid_x, sh, sw, shifts, dims):
    torch.manual_seed(9)
    mem_config = _explicit_block_shard(device, grid_y, grid_x, sh, sw)
    run_roll(device, torch.randn(shape, dtype=torch.bfloat16), ttnn.TILE_LAYOUT, mem_config, shifts, dims)


# ─── DRAM-sharded inputs: native support via DRAM bank NOC transfers ─────────


@pytest.mark.parametrize(
    "shape,sh,sw,shifts,dims",
    [
        ([1, 1, 8, 16], 2, 16, [4], [3]),  # last-dim within-row rotation
        ([1, 1, 8, 16], 2, 16, [2], [2]),  # height-dim page permutation
        ([1, 1, 16, 32], 4, 32, [8], [3]),  # larger shard
    ],
)
def test_roll_dram_sharded_row_major_native(device, shape, sh, sw, shifts, dims):
    """DRAM-sharded ROW_MAJOR roll — mode 2: full-shard L1 staging.
    Reads entire source shard from DRAM into L1, assembles the rolled result in L1 via
    element-level local copies, then writes the complete shard from L1 back to DRAM.
    All DRAM NOC transfers are shard-sized (32-byte aligned) — no sub-alignment issue.
    """
    torch.manual_seed(10)
    compute_grid = device.compute_with_storage_grid_size()
    total_h = shape[0] * shape[1] * shape[2]
    ncores = total_h // sh
    if ncores > compute_grid.x * compute_grid.y:
        pytest.skip(f"Device has insufficient cores ({ncores} needed)")
    shard_spec = ttnn.ShardSpec(
        ttnn.num_cores_to_corerangeset(ncores, compute_grid, True),
        [sh, sw],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, shard_spec)
    run_roll(device, torch.randn(shape, dtype=torch.bfloat16), ttnn.ROW_MAJOR_LAYOUT, mem_config, shifts, dims)


@pytest.mark.parametrize(
    "shape,sh,sw,shifts,dims",
    [
        ([1, 1, 64, 64], 32, 64, [32], [2]),  # height-dim tile permutation
        ([1, 1, 64, 128], 32, 128, [64], [3]),  # last-dim tile rotation
    ],
)
def test_roll_dram_sharded_tile_native(device, shape, sh, sw, shifts, dims):
    """DRAM-sharded TILE roll — mode 1: per-tile NOC read+write.
    Tiles are 2048 bytes (bf16), naturally satisfying the 32-byte DRAM NOC write
    alignment requirement — no full-shard staging needed.
    """
    torch.manual_seed(10)
    compute_grid = device.compute_with_storage_grid_size()
    total_h = shape[0] * shape[1] * shape[2]
    ncores = total_h // sh
    if ncores > compute_grid.x * compute_grid.y:
        pytest.skip(f"Device has insufficient cores ({ncores} needed)")
    shard_spec = ttnn.ShardSpec(
        ttnn.num_cores_to_corerangeset(ncores, compute_grid, True),
        [sh, sw],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, shard_spec)
    run_roll(device, torch.randn(shape, dtype=torch.bfloat16), ttnn.TILE_LAYOUT, mem_config, shifts, dims)


# ─── Program-cache hash: distinct shifts must produce distinct programs ──────


@pytest.mark.parametrize("shift", [2, 5, 7])
def test_roll_program_cache_distinct_shifts(device, shift):
    """Rolls with different shifts on the same shape must produce correct results."""
    torch.manual_seed(11)
    shape = [1, 1, 8, 16]
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        [2, 16],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    run_roll(device, torch.randn(shape, dtype=torch.bfloat16), ttnn.ROW_MAJOR_LAYOUT, mem_config, [shift], [3])


# ─── Program-cache hash: shape must be part of the key (regression for the fast path) ──


def test_roll_program_cache_distinct_shapes_same_shard_spec(device):
    """Two rolls that share (shift, dim, mem_config, dtype, layout) but differ only in
    padded shape must NOT collide in the program cache.

    [1,1,8,16] and [1,2,4,16] both map to the same 4-core [2,16] HEIGHT_SHARDED shard
    spec and the same (shift=4, dim=3), yet compute_roll_plan derives different row
    routing / offsets from their shapes. Before the shape was folded into
    RollDeviceOperation::compute_program_hash they produced an identical key, so the
    second call took the descriptor fast path and re-used the first call's stale plan.
    This asserts they now create distinct cache entries and both results are correct.
    """
    torch.manual_seed(15)
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        [2, 16],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

    entries_before = device.num_program_cache_entries()
    run_roll(device, torch.randn([1, 1, 8, 16], dtype=torch.bfloat16), ttnn.ROW_MAJOR_LAYOUT, mem_config, [4], [3])
    entries_after_first = device.num_program_cache_entries()
    run_roll(device, torch.randn([1, 2, 4, 16], dtype=torch.bfloat16), ttnn.ROW_MAJOR_LAYOUT, mem_config, [4], [3])
    entries_after_second = device.num_program_cache_entries()

    assert entries_after_first > entries_before, "first shape must be a cache miss"
    assert entries_after_second > entries_after_first, "differently shaped roll must not collide with the cached plan"


# ─── Program-cache: cache-hit retargeting on freshly allocated buffers ────────


def _roll_fresh(device, shape, mem_config, shifts, dims, bias):
    """Allocate a brand-new input/output for a roll and validate the result.

    Returns (torch_input, ttnn_input, ttnn_output); callers hold the references alive so
    that a subsequent allocation lands at a different address, forcing the cache-hit path
    to retarget its per-bank base+offset runtime args / CB bindings.
    """
    torch_input = torch.randn(shape, dtype=torch.bfloat16) + bias
    ttnn_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem_config
    )
    ttnn_output = ttnn.roll(ttnn_input, list(shifts), list(dims))
    got = ttnn.to_torch(ttnn_output.cpu())
    ref = torch.roll(torch_input, list(shifts), list(dims))
    assert_with_pcc(ref.float(), got.float(), _PCC)
    return torch_input, ttnn_input, ttnn_output


def test_roll_l1_sharded_cache_hit_retargets_buffers(device):
    """L1 HEIGHT_SHARDED roll: a second identical-spec roll on freshly allocated buffers
    must reuse the cached program yet still be correct.

    Guards the descriptor fast path: L1 mode re-emits a hash-stable placeholder to trip
    the fast path so the CB .buffer addresses get re-patched. A stale CB binding from the
    first dispatch would corrupt the second result even though the program is reused.
    """
    torch.manual_seed(16)
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        [2, 16],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

    _, in1, out1 = _roll_fresh(device, [1, 1, 8, 16], mem_config, [4], [3], 0.0)
    entries_after_first = device.num_program_cache_entries()
    _, in2, out2 = _roll_fresh(device, [1, 1, 8, 16], mem_config, [4], [3], 100.0)
    entries_after_second = device.num_program_cache_entries()

    assert entries_after_second == entries_after_first, "identical-spec sharded roll must be a cache hit"
    assert in2.buffer_address() != in1.buffer_address(), "second input must occupy a fresh address"


def test_roll_dram_sharded_cache_hit_retargets_buffers(device):
    """DRAM HEIGHT_SHARDED roll: a second identical-spec roll on freshly allocated buffers
    must reuse the cached program yet still be correct.

    Guards the descriptor fast path for the DRAM mode that bakes per-bank base+shard_offset
    NOC addresses: those slots are re-emitted on the cache hit. A stale per-bank address
    from the first dispatch would read/write the wrong DRAM bank on reuse.
    """
    torch.manual_seed(17)
    shape = [1, 1, 8, 16]
    sh, sw = 2, 16
    compute_grid = device.compute_with_storage_grid_size()
    ncores = (shape[0] * shape[1] * shape[2]) // sh
    if ncores > compute_grid.x * compute_grid.y:
        pytest.skip(f"Device has insufficient cores ({ncores} needed)")
    shard_spec = ttnn.ShardSpec(
        ttnn.num_cores_to_corerangeset(ncores, compute_grid, True),
        [sh, sw],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, shard_spec)

    _, in1, out1 = _roll_fresh(device, shape, mem_config, [4], [3], 0.0)
    entries_after_first = device.num_program_cache_entries()
    _, in2, out2 = _roll_fresh(device, shape, mem_config, [4], [3], 100.0)
    entries_after_second = device.num_program_cache_entries()

    assert entries_after_second == entries_after_first, "identical-spec DRAM-sharded roll must be a cache hit"
    assert in2.buffer_address() != in1.buffer_address(), "second input must occupy a fresh address"


# ─── Optional output memory_config ──────────────────────────────────────────


def test_roll_output_memory_config_dram(device):
    """Roll a sharded input and request DRAM interleaved output."""
    torch.manual_seed(12)
    shape = [1, 1, 8, 16]
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        [2, 16],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    x = torch.randn(shape, dtype=torch.bfloat16)
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=input_mc)
    out = ttnn.roll(t, [3], [3], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    assert out.memory_config().buffer_type == ttnn.BufferType.DRAM
    assert_with_pcc(torch.roll(x, [3], [3]).float(), ttnn.to_torch(out.cpu()).float(), _PCC)


def test_roll_output_memory_config_l1_interleaved(device):
    """Roll an interleaved input and request L1 interleaved output."""
    torch.manual_seed(13)
    shape = [1, 1, 32, 64]
    x = torch.randn(shape, dtype=torch.bfloat16)
    t = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out = ttnn.roll(t, [16], [3], memory_config=ttnn.L1_MEMORY_CONFIG)
    assert out.memory_config().buffer_type == ttnn.BufferType.L1
    assert_with_pcc(torch.roll(x, [16], [3]).float(), ttnn.to_torch(out.cpu()).float(), _PCC)


# ─── COL_MAJOR shard orientation ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "shape,sh,sw,shifts,dims,layout",
    [
        ([1, 1, 8, 16], 2, 16, [2], [3], ttnn.ROW_MAJOR_LAYOUT),  # last-dim rotation
        ([1, 1, 8, 16], 2, 16, [2], [2], ttnn.ROW_MAJOR_LAYOUT),  # height-dim permutation
        ([1, 1, 64, 64], 32, 64, [32], [3], ttnn.TILE_LAYOUT),  # tile-aligned, last-dim; sh=32 for tile alignment
        ([1, 1, 64, 64], 32, 64, [32], [2], ttnn.TILE_LAYOUT),  # tile-aligned, height-dim
    ],
)
def test_roll_col_major_height_sharded(device, shape, sh, sw, shifts, dims, layout):
    """HEIGHT_SHARDED with COL_MAJOR orientation — native kernel."""
    torch.manual_seed(14)
    compute_grid = device.compute_with_storage_grid_size()
    total_rows = shape[0] * shape[1] * shape[2]
    ncores = total_rows // sh
    if ncores > compute_grid.x * compute_grid.y:
        pytest.skip(f"Device has insufficient cores for this test ({ncores} needed)")
    shard_spec = ttnn.ShardSpec(
        ttnn.num_cores_to_corerangeset(ncores, compute_grid, True),
        [sh, sw],
        ttnn.ShardOrientation.COL_MAJOR,
    )
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    run_roll(device, torch.randn(shape, dtype=torch.bfloat16), layout, mem_config, shifts, dims)
