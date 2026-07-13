# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
End-to-end routed-expert + combine perf worker on a REAL workload.

Motivation
----------
`test_ttnn_moe.py` measures the whole MoE forward on golden-trace input and shows
the routed-expert / combine overlap is faster; the full chunked transformer test does
not reproduce that win. The difference is the per-expert token distribution: golden-trace
/ synthetic input spreads tokens differently than a real prefill. This worker replays the
REAL captured routing (like `test_prefill_dispatch_combine.py`) so the routed-expert FFN
and the combine see production per-expert load, and measures ONLY the routed-expert+combine
region with overlap on vs off.

Pipeline (mirrors tt_moe.py):
    dispatch  →  TILE+bfp8 layout transform  →  { routed_expert ⇄ combine }

The routed-expert/combine block is copied verbatim from `TtMoe.forward` (tt_moe.py), and
the sub-device + routed_expert_global_semaphore setup is copied from `TtMoe.__init__`, so
the on-device overlap behaviour is identical to production.

Measurement ("both"):
    * Host wall-clock: the region is run twice (iter0 populates the persistent program
      cache, iter1 is the steady-state measurement) and profiler.times[...][1] is logged.
    * Tracy signposts (`routed_expert_and_combine_start/end_iter{i}`) so the same worker
      can be captured by a device-perf harness.

Dummy random device weights are used: matmul cycle count is value-independent — only the
(real) per-expert token counts drive the work.

Required/optional env vars:
    TT_DS_CAPTURED_COL            optional, Galaxy column override (defaults to the layer's
                                  hottest column, see _HOT_LAYER_COL)
    TT_DS_USE_CAPTURED_INDICES    optional path override (defaults to LONGBOOK_QA_ENG_25600)
"""

import os

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.common.utility_functions import is_blackhole, profiler
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    ExpertMapping,
    compute_constants,
    create_fabric_router_config,
    extract_mesh_config,
    get_dispatch_input_mesh_mapper,
    get_ep_mesh_mapper,
    get_expert_token_counts_mesh_mapper,
    get_gate_outputs,
    load_captured_routing,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_combine import TtCombineModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatchModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe import MOE_L1_SMALL_REGION_SIZE
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl

# Hottest dispatch-group-imbalance (layer, col) pairs from LONGBOOK_QA_ENG_25600
# (see test_dispatch_combine_perf.py). L27/col2 is the hottest -> the default.
_HOT_LAYER_COL = {27: 2, 38: 0, 50: 0, 28: 1}


# (8,1) linear on Blackhole with an L1-small region for the routed-expert global semaphore
# (get_routed_expert_global_semaphore allocates it in BufferType.L1_SMALL). Matches the
# overlap-tested config in test_ttnn_moe.py (linear-8). The captured-routing loader simulates
# one Galaxy column on the 8-chip SP axis, so an 8-chip SP dimension is required.
_MESH_CONFIGS = [
    pytest.param(
        (8, 1),
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "fabric_router_config": create_fabric_router_config(max_payload_size=DeepSeekV3Config.FABRIC_PAYLOAD_SIZE),
            "l1_small_size": MOE_L1_SMALL_REGION_SIZE,
        },
        2 if is_blackhole() else 1,
        ttnn.Topology.Linear,
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 1), topology="linear"),
        id="linear-8",
    ),
]


@pytest.mark.parametrize(
    "seq_len_per_chip, emb_dim, hidden_dim, num_routed_experts, num_experts_per_tok, "
    "dispatch_buffer_capacity_factor, experts_per_chip_override",
    [
        pytest.param(
            3200,
            DeepSeekV3Config.EMB_SIZE,
            DeepSeekV3Config.MOE_INTERMEDIATE_SIZE,
            256,
            8,
            8,
            8,
            id="perf_real_indices",
        )
    ],
)
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    _MESH_CONFIGS,
    indirect=["mesh_device", "device_params"],
)
# Layer selection (default: L27, the hottest). Pick one with e.g. `-k l38`.
@pytest.mark.parametrize("layer", [27, 38, 50, 28], ids=lambda l: f"l{l:02d}")
@pytest.mark.parametrize("overlap", [True, False], ids=["overlap", "no_overlap"])
@pytest.mark.timeout(0)
def test_ttnn_routed_expert_combine(
    mesh_device,
    seq_len_per_chip,
    emb_dim,
    hidden_dim,
    num_routed_experts,
    num_experts_per_tok,
    dispatch_buffer_capacity_factor,
    experts_per_chip_override,
    num_links,
    topology,
    layer,
    overlap,
):
    # The routed-expert / combine overlap relies on the unified_routed_expert_moe op, which
    # is only supported on Blackhole (TtRoutedExpert / TtMoe assert this).
    if not is_blackhole():
        pytest.skip("routed_expert + combine (unified_routed_expert_moe) is only supported on Blackhole")

    col = int(os.getenv("TT_DS_CAPTURED_COL", _HOT_LAYER_COL[layer]))

    num_devices = mesh_device.get_num_devices()
    torch.manual_seed(42)
    profiler.clear()

    mesh_config = extract_mesh_config(mesh_device)
    sp_axis = mesh_config.sp_axis
    dispatch_group_size = mesh_config.dispatch_group_size
    num_dispatch_groups = mesh_config.num_dispatch_groups

    signpost(
        f"RoutedExpertCombine layer={layer} col={col} overlap={overlap} mesh={tuple(mesh_device.shape)} "
        f"num_links={num_links} topology={topology}"
    )

    (
        experts_per_chip,
        metadata_len,
        max_dispatch_buffer_token_size,
        max_dispatched_tokens_per_expert,
    ) = compute_constants(
        seq_len_per_chip,
        num_routed_experts,
        num_experts_per_tok,
        num_devices,
        dispatch_group_size,
        dispatch_buffer_capacity_factor,
        experts_per_chip_override=experts_per_chip_override,
    )
    logger.info(
        f"[routed_expert_combine] config: layer={layer} col={col} overlap={overlap} "
        f"experts_per_chip={experts_per_chip} metadata_len={metadata_len} "
        f"max_dispatch_buffer_token_size={max_dispatch_buffer_token_size} "
        f"max_dispatched_tokens_per_expert={max_dispatched_tokens_per_expert} "
        f"num_dispatch_groups(mesh)={num_dispatch_groups}"
    )

    # ------------------------------------------------------------------
    # Real workload: captured routing (indices remapped to [0, 64) ∪ {255}, col-0 table).
    # ------------------------------------------------------------------
    indices, expert_dispatch_table = load_captured_routing(
        dispatch_group_size=dispatch_group_size,
        seq_len_per_chip=seq_len_per_chip,
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        layer=layer,
        col=col,
        captured_indices_path=os.getenv("TT_DS_USE_CAPTURED_INDICES"),
    )

    # get_gate_outputs produces 4-row outputs (Galaxy-global); slice to [0:1] for LB's single dispatch group.
    expert_offsets, expert_token_counts, expert_region_offsets, _ = get_gate_outputs(
        indices,
        dispatch_group_size,
        num_routed_experts,
        experts_per_chip,
        seq_len_per_chip,
        num_experts_per_tok,
        expert_dispatch_table=expert_dispatch_table,
    )
    expert_offsets = expert_offsets[0:1].contiguous()
    expert_token_counts = expert_token_counts[0:1].contiguous()
    expert_region_offsets = expert_region_offsets[0:1].contiguous()

    # ------------------------------------------------------------------
    # Dispatch inputs. Values don't drive kernel cycle count — only indices/offsets do.
    # ------------------------------------------------------------------
    x = torch.randn(dispatch_group_size, seq_len_per_chip, emb_dim, dtype=torch.bfloat16)
    weights = torch.zeros(dispatch_group_size, seq_len_per_chip, num_experts_per_tok, dtype=torch.bfloat16)

    input_mapper = get_dispatch_input_mesh_mapper(mesh_device, sp_axis)
    tt_x = ttnn.from_torch(
        x, mesh_mapper=input_mapper, layout=ttnn.TILE_LAYOUT, device=mesh_device, dtype=ttnn.bfloat16
    )
    tt_weights = ttnn.from_torch(
        weights, mesh_mapper=input_mapper, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, dtype=ttnn.bfloat16
    )
    tt_indices = ttnn.from_torch(
        indices.to(torch.int16),
        mesh_mapper=input_mapper,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.uint16,
    )

    tt_table = TtDispatchModule.shard_expert_dispatch_table(mesh_device, expert_dispatch_table, sp_axis)
    tt_offsets = TtDispatchModule.shard_expert_offsets(mesh_device, expert_offsets)

    # Counts/offsets for combine — per-device (1, 1, num_routed_experts), as combine expects.
    counts_mapper = get_expert_token_counts_mesh_mapper(mesh_device)
    tt_counts_combine = ttnn.from_torch(
        expert_token_counts,
        mesh_mapper=counts_mapper,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.int32,
    )
    tt_region_offsets_combine = ttnn.from_torch(
        expert_region_offsets,
        mesh_mapper=counts_mapper,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.int32,
    )

    # Counts/offsets for routed_expert — per-device (1, num_routed_experts), as the unified op expects.
    tt_counts_re = TtRoutedExpert.shard_expert_token_counts(mesh_device, expert_token_counts)
    tt_region_offsets_re = TtRoutedExpert.shard_expert_token_counts(mesh_device, expert_region_offsets)

    # ------------------------------------------------------------------
    # Sub-devices + routed-expert global semaphore (copied from TtMoe.__init__).
    # dm strip = combine, compute strip = routed expert. Only built when overlap=True.
    # ------------------------------------------------------------------
    tt_ccl = get_tt_ccl(mesh_device)
    sd_manager_id = None
    dm_sd_id = None
    compute_sd_id = None
    routed_expert_global_semaphore = None
    if overlap:
        dm_sd_rows = 1
        grid = mesh_device.compute_with_storage_grid_size()
        grid_x, grid_y = grid.x, grid.y
        assert 0 < dm_sd_rows < grid_y, f"dm_sd_rows={dm_sd_rows} must be in (0, grid_y={grid_y})"
        dm_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, dm_sd_rows - 1))})
        compute_cores = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, dm_sd_rows), ttnn.CoreCoord(grid_x - 1, grid_y - 1))}
        )
        dm_sd = ttnn.SubDevice([dm_cores])
        compute_sd = ttnn.SubDevice([compute_cores])
        sd_manager_id = mesh_device.create_sub_device_manager([dm_sd, compute_sd], 0)
        dm_sd_id = ttnn.SubDeviceId(0)
        compute_sd_id = ttnn.SubDeviceId(1)
        routed_expert_global_semaphore = tt_ccl.get_routed_expert_global_semaphore(dm_cores)

    # ------------------------------------------------------------------
    # Modules.
    # ------------------------------------------------------------------
    dispatch_module = TtDispatchModule(
        mesh_device=mesh_device,
        dispatch_group_size=dispatch_group_size,
        experts_per_chip=experts_per_chip,
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        metadata_len=metadata_len,
        max_dispatch_buffer_token_size=max_dispatch_buffer_token_size,
        seq_len_per_chip=seq_len_per_chip,
        emb_dim=emb_dim,
        cluster_axis=sp_axis,
        num_links=num_links,
        topology=topology,
    )
    combine_module = TtCombineModule(
        mesh_device=mesh_device,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=1,  # LB has 1 physical dispatch group; we simulate one Galaxy col on it.
        experts_per_chip=experts_per_chip,
        num_experts_per_tok=num_experts_per_tok,
        seq_len_per_chip=seq_len_per_chip,
        cluster_axis=sp_axis,
        num_links=num_links,
        topology=topology,
        init_zeros=False,  # matches tt_moe.py's combine in the routed-expert overlap
        subdevice_id=dm_sd_id if overlap else None,
        global_semaphore=routed_expert_global_semaphore if overlap else None,
    )

    # Build the (group, chip, local_expert) -> global expert id table, sharded across the EP
    # mesh so each device holds (experts_per_chip,) after squeezing the two singleton dims.
    global_expert_idx_tt = ttnn.from_torch(
        ExpertMapping.create_global_expert_idx_table(
            experts_per_chip=experts_per_chip,
            dispatch_group_size=dispatch_group_size,
            num_dispatch_groups=num_dispatch_groups,
        ),
        mesh_mapper=get_ep_mesh_mapper(mesh_device),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.uint32,
    )
    global_expert_idx_tt = ttnn.squeeze(global_expert_idx_tt, 0)
    global_expert_idx_tt = ttnn.squeeze(global_expert_idx_tt, 0)

    routed_expert = TtRoutedExpert(
        mesh_device=mesh_device,
        experts_per_chip=experts_per_chip,
        global_expert_idx_table=global_expert_idx_tt,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        max_tokens=max_dispatched_tokens_per_expert,
        torch_weights=None,  # random device weights (values don't drive cycle count)
        activations_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat4_b,
        activation=ttnn.RoutedExpertActivation.Silu,
        subdevice_id=compute_sd_id if overlap else None,
        global_semaphore=routed_expert_global_semaphore if overlap else None,
    )

    # ------------------------------------------------------------------
    # Dispatch once (outside the measured region), producing a real dispatched buffer.
    # ------------------------------------------------------------------
    dispatched_buffer, metadata = dispatch_module(tt_x, tt_weights, tt_indices, tt_offsets, tt_table)
    ttnn.synchronize_device(mesh_device)

    # ------------------------------------------------------------------
    # Measured region: routed_expert + combine (verbatim from tt_moe.py). Run twice —
    # iter0 builds the persistent program cache, iter1 is the steady-state measurement.
    # ------------------------------------------------------------------
    for i in range(2):
        # Convert dispatched_buffer to TILE_LAYOUT for routed experts (re-created each iter;
        # the ops consume the tiled buffer). Mirrors tt_moe.py.
        dispatched_buffer_tiled = ttnn.to_layout(
            dispatched_buffer, ttnn.TILE_LAYOUT, dtype=routed_expert.activations_dtype
        )

        signpost(header=f"routed_expert_and_combine_start_iter{i}")
        profiler.start("routed_expert_and_combine")
        if overlap:
            # Overlap: routed expert (compute sub-device) and combine (dm sub-device) run
            # concurrently, synchronized by the routed-expert global semaphore.
            mesh_device.load_sub_device_manager(sd_manager_id)
            mesh_device.set_sub_device_stall_group([compute_sd_id])
            combined_output = combine_module(
                dispatched_buffer_tiled, metadata, tt_counts_combine, tt_region_offsets_combine
            )
            expert_outputs = routed_expert(dispatched_buffer_tiled, tt_counts_re, tt_region_offsets_re)
            mesh_device.reset_sub_device_stall_group()
            mesh_device.clear_loaded_sub_device_manager()
        else:
            # No overlap: combine must read the buffer only AFTER routed_expert has written it.
            expert_outputs = routed_expert(dispatched_buffer_tiled, tt_counts_re, tt_region_offsets_re)
            combined_output = combine_module(expert_outputs, metadata, tt_counts_combine, tt_region_offsets_combine)
        ttnn.synchronize_device(mesh_device)
        profiler.end("routed_expert_and_combine")
        signpost(header=f"routed_expert_and_combine_end_iter{i}")

        ttnn.deallocate(combined_output)
        ttnn.deallocate(expert_outputs)

    second_run_ms = profiler.times["routed_expert_and_combine"][1] * 1000
    logger.info(
        f"[routed_expert_combine] layer={layer} col={col} overlap={overlap} "
        f"routed_expert+combine (second run): {second_run_ms:.2f} ms"
    )
