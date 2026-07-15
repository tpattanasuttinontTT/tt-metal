# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Blackhole-box tests for the two ttnn.all_gather features ported from ring_mla's fused gather:

  1. batch_slice_idx     - gather only a single index along dim 0 (output dim-0 collapses to 1).
  2. valid_gather_extent - gather only the valid prefix of an overallocated (padded) gather dim,
                           leaving the padded tail of the output untouched.

Covers correctness (bit-exact), memory-shape guarantees (G4/G5), the even-ring split-forwarding
path (G2), a persistent-buffer/no-recompile trace case (G6), the combined feature (G3), and the
fail-fast guards (G8). Run with scripts/run_safe_pytest.sh --dev so a split-forwarding hang is
caught and auto-triaged.
"""

import torch
import pytest
from loguru import logger
import ttnn

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal
from models.common.utility_functions import skip_for_wormhole_b0, skip_for_n_or_less_dev
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.box.nightly.test_all_gather_nightly import validate_test


def _read_replicated(mesh_device, tt_out, batch):
    """The all-gather output is replicated across the mesh; return device-0's copy on host."""
    host = ttnn.from_device(tt_out)
    full = ttnn.to_torch(host, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    return full[0:batch]


def _make_submesh(bh_1d_mesh_device, num_devices):
    return bh_1d_mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))


# ---------------------------------------------------------------------------
# Feature 1: single-batch slice
# ---------------------------------------------------------------------------
@skip_for_wormhole_b0()
@skip_for_n_or_less_dev(1)
@pytest.mark.parametrize("num_devices", [4, 8])
@pytest.mark.parametrize("batch", [2, 8])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("ag_input_dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["bf16", "bf8"])
@pytest.mark.parametrize("num_links", [1, 2], ids=["1_link", "2_links"])
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112}, ttnn.Topology.Ring),
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
    ids=["ring", "line"],
)
def test_all_gather_batch_slice(
    bh_1d_mesh_device, num_devices, batch, dim, ag_input_dtype, num_links, all_gather_topology
):
    validate_test(num_devices, all_gather_topology, bh_1d_mesh_device.shape, 0)
    mesh = _make_submesh(bh_1d_mesh_device, num_devices)

    heads, s_local, head_dim = 4, 32, 128
    # full_input assembled so sharding along `dim` gives each device its slice.
    full_shape = (
        [batch, heads, s_local * num_devices, head_dim] if dim == 2 else [batch, heads, s_local, head_dim * num_devices]
    )
    full_input = torch.randn(full_shape).bfloat16()

    input_mesh = ttnn.from_torch(
        full_input,
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ag_input_dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=dim),
    )

    for slice_idx in (0, 1, batch - 1):
        tt_out = ttnn.all_gather(
            input_mesh,
            dim=dim,
            cluster_axis=0,
            num_links=num_links,
            topology=all_gather_topology,
            batch_slice_idx=slice_idx,
        )
        ttnn.synchronize_device(mesh)

        # G4: output dim-0 collapses to 1.
        assert tuple(tt_out.shape)[0] == 1, f"expected dim-0==1, got {tt_out.shape}"

        out = _read_replicated(mesh, tt_out, batch=1)
        golden = full_input[slice_idx : slice_idx + 1]
        eq, msg = comp_equal(out, golden)
        assert eq, f"batch_slice_idx={slice_idx} dim={dim} nd={num_devices}: {msg}"
        logger.info(f"OK batch_slice_idx={slice_idx} dim={dim} nd={num_devices} topo={all_gather_topology}")


# ---------------------------------------------------------------------------
# Feature 2: partial / overallocated gather (persistent output, sentinel tail)
# ---------------------------------------------------------------------------
@skip_for_wormhole_b0()
@skip_for_n_or_less_dev(1)
@pytest.mark.parametrize("num_devices", [4, 8], ids=["ring4", "ring8"])  # even -> split-forwarding
@pytest.mark.parametrize(
    "s_pad_local, valid_local",
    [
        (128, 64),  # (a) worker-aligned-ish valid prefix
        (128, 96),  # (b) different valid length
        (128, 32),  # (c) small valid (some workers idle)
    ],
    ids=["valid64", "valid96", "valid32"],
)
@pytest.mark.parametrize("num_links", [1, 2], ids=["1_link", "2_links"])
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112}, ttnn.Topology.Ring),
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
    ids=["ring", "line"],
)
def test_all_gather_partial(bh_1d_mesh_device, num_devices, s_pad_local, valid_local, num_links, all_gather_topology):
    validate_test(num_devices, all_gather_topology, bh_1d_mesh_device.shape, 0)
    mesh = _make_submesh(bh_1d_mesh_device, num_devices)

    batch, heads, head_dim = 1, 2, 128
    dim = 2  # gather along height (rank-2), the only supported dim for valid_gather_extent
    SENTINEL = -99.0

    # Per-device local plane is padded to s_pad_local along the gather dim; only valid_local rows are real.
    full_shape = [batch, heads, s_pad_local * num_devices, head_dim]
    full_input = torch.randn(full_shape).bfloat16()

    input_mesh = ttnn.from_torch(
        full_input,
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=dim),
    )

    # Persistent output pre-filled with a sentinel; only valid prefixes should be overwritten (M3).
    out_shape = [batch, heads, s_pad_local * num_devices, head_dim]
    persistent_out = ttnn.from_torch(
        torch.full(out_shape, SENTINEL).bfloat16(),
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )

    tt_out = ttnn.all_gather(
        input_mesh,
        dim=dim,
        cluster_axis=0,
        num_links=num_links,
        topology=all_gather_topology,
        output_tensor=persistent_out,
        valid_gather_extent=valid_local,
    )
    ttnn.synchronize_device(mesh)

    out = _read_replicated(mesh, tt_out, batch=batch)

    # Build golden: sentinel everywhere, then write each device's valid prefix into its slab.
    golden = torch.full(out_shape, SENTINEL).bfloat16()
    for dev in range(num_devices):
        base = dev * s_pad_local
        src = full_input[:, :, base : base + valid_local, :]
        golden[:, :, base : base + valid_local, :] = src

    eq, msg = comp_equal(out, golden)
    assert eq, f"partial nd={num_devices} s_pad={s_pad_local} valid={valid_local} topo={all_gather_topology}: {msg}"
    logger.info(f"OK partial nd={num_devices} valid={valid_local}/{s_pad_local} topo={all_gather_topology}")


# ---------------------------------------------------------------------------
# G6: no recompile when only valid_gather_extent changes (persistent buffer, trace-like reuse)
# ---------------------------------------------------------------------------
@skip_for_wormhole_b0()
@skip_for_n_or_less_dev(1)
@pytest.mark.parametrize("num_devices", [8])
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112}, ttnn.Topology.Ring)],
    indirect=["device_params"],
    ids=["ring"],
)
def test_all_gather_partial_value_change_no_recompile(bh_1d_mesh_device, num_devices, all_gather_topology):
    validate_test(num_devices, all_gather_topology, bh_1d_mesh_device.shape, 0)
    mesh = _make_submesh(bh_1d_mesh_device, num_devices)

    batch, heads, head_dim, s_pad_local = 1, 2, 128, 128
    dim = 2
    full_input = torch.randn([batch, heads, s_pad_local * num_devices, head_dim]).bfloat16()
    input_mesh = ttnn.from_torch(
        full_input,
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=dim),
    )

    def run(valid_local):
        persistent_out = ttnn.from_torch(
            torch.full([batch, heads, s_pad_local * num_devices, head_dim], -99.0).bfloat16(),
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        out = ttnn.all_gather(
            input_mesh,
            dim=dim,
            cluster_axis=0,
            num_links=1,
            topology=all_gather_topology,
            output_tensor=persistent_out,
            valid_gather_extent=valid_local,
        )
        ttnn.synchronize_device(mesh)
        return _read_replicated(mesh, out, batch=batch)

    # Warm the cache, then vary only the valid extent; program cache entry count must not grow.
    run(32)
    entries_after_first = mesh.num_program_cache_entries()
    for valid_local in (64, 96, 32):
        out = run(valid_local)
        golden = torch.full([batch, heads, s_pad_local * num_devices, head_dim], -99.0).bfloat16()
        for dev in range(num_devices):
            base = dev * s_pad_local
            golden[:, :, base : base + valid_local, :] = full_input[:, :, base : base + valid_local, :]
        eq, msg = comp_equal(out, golden)
        assert eq, f"value-change valid={valid_local}: {msg}"
    entries_after = mesh.num_program_cache_entries()
    assert (
        entries_after == entries_after_first
    ), f"program recompiled on value-only change: {entries_after_first} -> {entries_after}"
    logger.info(f"OK no-recompile: cache entries stable at {entries_after}")


# ---------------------------------------------------------------------------
# G3: combined - single batch slice of an overallocated cache
# ---------------------------------------------------------------------------
@skip_for_wormhole_b0()
@skip_for_n_or_less_dev(1)
@pytest.mark.parametrize("num_devices", [4, 8])
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112}, ttnn.Topology.Ring)],
    indirect=["device_params"],
    ids=["ring"],
)
def test_all_gather_combined_slice_and_partial(bh_1d_mesh_device, num_devices, all_gather_topology):
    validate_test(num_devices, all_gather_topology, bh_1d_mesh_device.shape, 0)
    mesh = _make_submesh(bh_1d_mesh_device, num_devices)

    batch, heads, head_dim, s_pad_local, valid_local = 4, 2, 128, 128, 64
    dim = 2
    slice_idx = 2
    SENTINEL = -99.0
    full_input = torch.randn([batch, heads, s_pad_local * num_devices, head_dim]).bfloat16()
    input_mesh = ttnn.from_torch(
        full_input,
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=dim),
    )
    persistent_out = ttnn.from_torch(
        torch.full([1, heads, s_pad_local * num_devices, head_dim], SENTINEL).bfloat16(),
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )

    tt_out = ttnn.all_gather(
        input_mesh,
        dim=dim,
        cluster_axis=0,
        num_links=1,
        topology=all_gather_topology,
        output_tensor=persistent_out,
        batch_slice_idx=slice_idx,
        valid_gather_extent=valid_local,
    )
    ttnn.synchronize_device(mesh)
    assert tuple(tt_out.shape)[0] == 1
    out = _read_replicated(mesh, tt_out, batch=1)

    golden = torch.full([1, heads, s_pad_local * num_devices, head_dim], SENTINEL).bfloat16()
    for dev in range(num_devices):
        base = dev * s_pad_local
        golden[:, :, base : base + valid_local, :] = full_input[
            slice_idx : slice_idx + 1, :, base : base + valid_local, :
        ]
    eq, msg = comp_equal(out, golden)
    assert eq, f"combined nd={num_devices}: {msg}"
    logger.info(f"OK combined nd={num_devices}")


# ---------------------------------------------------------------------------
# G8: fail-fast guards
# ---------------------------------------------------------------------------
@skip_for_wormhole_b0()
@skip_for_n_or_less_dev(1)
@pytest.mark.parametrize("num_devices", [8])
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112}, ttnn.Topology.Ring)],
    indirect=["device_params"],
    ids=["ring"],
)
def test_all_gather_feature_guards(bh_1d_mesh_device, num_devices, all_gather_topology, expect_error):
    validate_test(num_devices, all_gather_topology, bh_1d_mesh_device.shape, 0)
    mesh = _make_submesh(bh_1d_mesh_device, num_devices)

    batch, heads, head_dim, s_local = 2, 2, 128, 32
    full_input = torch.randn([batch, heads, s_local * num_devices, head_dim]).bfloat16()
    input_mesh = ttnn.from_torch(
        full_input,
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=2),
    )

    # batch_slice_idx with gather dim == 0 must fail.
    input_dim0 = ttnn.from_torch(
        torch.randn([s_local * num_devices, heads, batch, head_dim]).bfloat16(),
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )
    with expect_error(RuntimeError, "batch_slice_idx is not supported when gathering along dim 0"):
        ttnn.all_gather(input_dim0, dim=0, cluster_axis=0, topology=all_gather_topology, batch_slice_idx=0)
        ttnn.synchronize_device(mesh)

    # out-of-range batch_slice_idx must fail.
    with expect_error(RuntimeError, "out of range for dim-0 extent"):
        ttnn.all_gather(input_mesh, dim=2, cluster_axis=0, topology=all_gather_topology, batch_slice_idx=batch)
        ttnn.synchronize_device(mesh)

    # valid_gather_extent on the width dim (rank-1) must fail (height-only feature).
    with expect_error(RuntimeError, "valid_gather_extent is only supported when gathering along the height dim"):
        ttnn.all_gather(input_mesh, dim=3, cluster_axis=0, topology=all_gather_topology, valid_gather_extent=64)
        ttnn.synchronize_device(mesh)

    logger.info("OK feature guards raise as expected")


# ---------------------------------------------------------------------------
# ttnn.experimental.all_gather_async batch_slice_idx: composite (row-major, the
# _gather_kvpe_prefix path) and prim (tile) routes.
# ---------------------------------------------------------------------------
@skip_for_wormhole_b0()
@skip_for_n_or_less_dev(1)
@pytest.mark.parametrize("num_devices", [4, 8])
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=["row_major", "tile"])
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Linear)],
    indirect=["device_params"],
    ids=["line"],
)
def test_all_gather_async_batch_slice(bh_1d_mesh_device, num_devices, layout, all_gather_topology):
    validate_test(num_devices, all_gather_topology, bh_1d_mesh_device.shape, 0)
    mesh = _make_submesh(bh_1d_mesh_device, num_devices)

    compute_grid = mesh.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid.x - 1, compute_grid.y - 1))}
    )
    mesh.set_sub_device_stall_group([ttnn.SubDeviceId(0)])
    sems = [ttnn.create_global_semaphore(mesh, crs, 0) for _ in range(2)]
    barrier = ttnn.create_global_semaphore(mesh, crs, 0)

    batch, heads, s_local, head_dim = 4, 1, 64, 576
    full_input = torch.randn([batch, heads, s_local * num_devices, head_dim]).bfloat16()
    input_mesh = ttnn.from_torch(
        full_input,
        device=mesh,
        layout=layout,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=2),
    )

    for slice_idx in (0, batch - 1):
        tt_out = ttnn.experimental.all_gather_async(
            input_mesh,
            dim=2,
            multi_device_global_semaphore=sems,
            barrier_semaphore=barrier,
            num_links=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=all_gather_topology,
            cluster_axis=0,
            batch_slice_idx=slice_idx,
        )
        ttnn.synchronize_device(mesh)
        assert tuple(tt_out.shape)[0] == 1, f"expected dim-0==1, got {tt_out.shape}"
        out = _read_replicated(mesh, tt_out, batch=1)
        golden = full_input[slice_idx : slice_idx + 1]
        eq, msg = comp_equal(out, golden)
        assert eq, f"async batch_slice_idx={slice_idx} layout={layout} nd={num_devices}: {msg}"
        logger.info(f"OK async batch_slice_idx={slice_idx} layout={layout} nd={num_devices}")

    mesh.reset_sub_device_stall_group()


# ---------------------------------------------------------------------------
# ttnn.experimental.all_gather_async valid_gather_extent on the COMPOSITE (row-major) path:
# gather only the leading `valid` elements per device shard -> tight (sp*valid) output. This is the
# capability _gather_kvpe_prefix uses to drop unwritten trailing block-cyclic slabs.
# ---------------------------------------------------------------------------
@skip_for_wormhole_b0()
@skip_for_n_or_less_dev(1)
@pytest.mark.parametrize("num_devices", [4, 8])
@pytest.mark.parametrize("valid_local", [32, 64], ids=["valid32", "valid64"])
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Linear)],
    indirect=["device_params"],
    ids=["line"],
)
def test_all_gather_async_composite_valid(bh_1d_mesh_device, num_devices, valid_local, all_gather_topology):
    validate_test(num_devices, all_gather_topology, bh_1d_mesh_device.shape, 0)
    mesh = _make_submesh(bh_1d_mesh_device, num_devices)

    compute_grid = mesh.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid.x - 1, compute_grid.y - 1))}
    )
    mesh.set_sub_device_stall_group([ttnn.SubDeviceId(0)])
    sems = [ttnn.create_global_semaphore(mesh, crs, 0) for _ in range(2)]
    barrier = ttnn.create_global_semaphore(mesh, crs, 0)

    heads, s_local, head_dim = 1, 128, 576  # per-device shard = s_local along dim 2 (row-major)
    full_input = torch.randn([1, heads, s_local * num_devices, head_dim]).bfloat16()
    input_mesh = ttnn.from_torch(
        full_input,
        device=mesh,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=2),
    )

    tt_out = ttnn.experimental.all_gather_async(
        input_mesh,
        dim=2,
        multi_device_global_semaphore=sems,
        barrier_semaphore=barrier,
        num_links=1,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=all_gather_topology,
        cluster_axis=0,
        valid_gather_extent=valid_local,
    )
    ttnn.synchronize_device(mesh)

    # tight output: each device contributes only its leading valid_local rows.
    assert tuple(tt_out.shape)[2] == valid_local * num_devices, f"expected tight dim-2, got {tt_out.shape}"
    out = _read_replicated(mesh, tt_out, batch=1)
    golden = torch.cat(
        [full_input[:, :, d * s_local : d * s_local + valid_local, :] for d in range(num_devices)], dim=2
    )
    eq, msg = comp_equal(out, golden)
    assert eq, f"async valid_gather_extent={valid_local} nd={num_devices}: {msg}"
    logger.info(f"OK async composite valid={valid_local} nd={num_devices} -> T={valid_local * num_devices}")

    mesh.reset_sub_device_stall_group()
