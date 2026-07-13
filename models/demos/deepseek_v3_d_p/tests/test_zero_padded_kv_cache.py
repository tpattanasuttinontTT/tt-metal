# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""On-device validation for the zero_padded_kv_cache experimental op.

Seeds a block-cyclic cache with all-ones, runs the op, reconstructs natural order, asserts:
  * [0, valid_global)              real (==1)
  * [valid_global, ceil_128(v))    zero  (pad window)
  * [ceil_128(v), :)               real (==1)  -- nothing past the window
"""

import math

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.tt.mla.utils import blockcyclic_positions
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache

# Production chunk_size_global only (chunk_local = 5120/8 = 640). At this csg a 128-pad window never
# crosses a chip border, so every case below lands its window on a single chip; the op's multi-chip
# path (window crossing) stays exercised by the on-device per-chip logic but is not csg-varied here.
# (chunk_size_global, seq_len_cache, valid_global, expected_boundary_chip)  expected_chip=None => skip
_CASES = [
    (5120, 5120, 740, 1),  # chip1 slab0, single-tile partial
    (5120, 5120, 2600, 4),  # chip4 slab0, 3-tile window
    (5120, 5120, 4512, 7),  # chip7 slab0, row_start=0 -> 3 full tiles
    (5120, 5120, 20, 0),  # chip0 slab0, 4-tile window
    (5120, 5120, 4480, None),  # 128-aligned -> skip
    (5120, 10240, 6668, 2),  # chip2 slab1, 4-tile window
]
_IDS = [f"v{v}_chip{ch}" if ch is not None else f"v{v}_aligned" for (_c, _s, v, ch) in _CASES]


@pytest.mark.parametrize("mesh_device", [(2, 4)], ids=["2x4"], indirect=True)
@pytest.mark.parametrize(
    "dtype,layout",
    [
        (ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
        (ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
        (ttnn.fp8_e4m3, ttnn.ROW_MAJOR_LAYOUT),
    ],
    ids=["bfp8_tile", "bf16_rm", "fp8_rm"],
)
@pytest.mark.timeout(0)
def test_zero_padded_kv_cache_formats(mesh_device, dtype, layout):
    """Sparse ROW_MAJOR coverage plus a local TILE regression on the same boundary geometry.

    Fill a two-user/two-layer cache entirely with ones through update_padded_kv_cache, then zero two
    different pad windows on the same structural layer. The first window crosses an SP chip boundary;
    the second call changes slot_idx and valid_global through the program-cache hit path. Untargeted
    slots and every row outside the exact windows must remain one.
    """
    if dtype == ttnn.fp8_e4m3 and not is_blackhole():
        pytest.skip("FP8_E4M3 is Blackhole-only")

    sp_axis = 0
    sp = mesh_device.shape[sp_axis]
    mesh_shape = list(mesh_device.shape)
    chunk_size_global = 384  # local=192: [180,256) crosses chip0 -> chip1
    seq_len_cache = 768
    head_dim = 64
    num_users = 3
    num_layers = 2

    cache = init_kvpe_cache(
        head_dim,
        mesh_device,
        seq_len_cache,
        mesh_shape,
        sp_axis,
        num_kvpe_cache_layers=num_layers,
        num_users=num_users,
        dtype=dtype,
        layout=layout,
    )
    assert cache.dtype == dtype
    assert cache.layout == layout
    if dtype == ttnn.fp8_e4m3:
        initialized = ttnn.to_torch(
            cache,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape),
        ).float()
        assert torch.count_nonzero(initialized) == 0

    # Build one globally sharded chunk of ones. FP8 cannot enter through this mesh-mapper path
    # directly, so create BF16 row-major and typecast on device (the production MLA path does the same).
    ones = torch.ones(1, 1, chunk_size_global, head_dim, dtype=torch.bfloat16)
    tt_ones = ttnn.from_torch(
        ones,
        device=mesh_device,
        dtype=ttnn.bfloat16 if dtype == ttnn.fp8_e4m3 else dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_shape), dims=(2, None)),
    )
    if dtype == ttnn.fp8_e4m3:
        tt_ones = ttnn.typecast(tt_ones, ttnn.fp8_e4m3)

    # Two chunk writes fill every physical cache row with one for every (user, layer) slot.
    for slot_idx in range(num_users):
        for layer_idx in range(num_layers):
            for kv_actual_global in (0, chunk_size_global):
                ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                    cache,
                    tt_ones,
                    slot_idx=slot_idx,
                    layer_idx=layer_idx,
                    num_layers=num_layers,
                    kv_actual_global=kv_actual_global,
                    cluster_axis=sp_axis,
                )

    mesh_device.enable_program_cache()
    cache_entries_before = mesh_device.num_program_cache_entries()
    windows = {
        (0, 1): (180, 256),  # crosses chip0 -> chip1
        (1, 1): (350, 384),  # ends exactly at a block-cyclic slab boundary
        (2, 1): (430, 512),  # program-cache hit in the next slab
    }
    for (slot_idx, layer_idx), (valid_global, _pad_end) in windows.items():
        ttnn.experimental.deepseek_prefill.zero_padded_kv_cache(
            cache,
            slot_idx,
            layer_idx,
            num_layers,
            valid_global,
            chunk_size_global,
            sp_axis,
            128,
        )
    ttnn.synchronize_device(mesh_device)

    cache_sr = ttnn.to_torch(
        cache,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape),
    ).to(torch.bfloat16)[:, :1]
    positions = blockcyclic_positions(sp, chunk_size_global, seq_len_cache)
    for slot_idx in range(num_users):
        for layer_idx in range(num_layers):
            batch_idx = slot_idx * num_layers + layer_idx
            natural = torch.empty(seq_len_cache, head_dim, dtype=torch.bfloat16)
            natural[positions] = cache_sr[batch_idx, 0]
            expected = torch.ones_like(natural)
            if (slot_idx, layer_idx) in windows:
                start, end = windows[(slot_idx, layer_idx)]
                expected[start:end] = 0
            assert torch.equal(natural, expected), (
                f"{layout} {dtype} cache mismatch for slot={slot_idx}, layer={layer_idx}; "
                f"min={natural.min().item()} max={natural.max().item()}"
            )

    # The two calls share structural args and layer_idx, so slot/valid changes must reuse one program.
    assert mesh_device.num_program_cache_entries() == cache_entries_before + 1


@pytest.mark.parametrize("mesh_device", [(8, 4)], ids=["8x4"], indirect=True)
@pytest.mark.parametrize("chunk_size_global,seq_len_cache,valid_global,expected_chip", _CASES, ids=_IDS)
@pytest.mark.timeout(0)
def test_zero_padded_kv_cache(mesh_device, chunk_size_global, seq_len_cache, valid_global, expected_chip):
    mesh_shape = list(mesh_device.shape)
    sp_axis = 0
    sp = mesh_shape[sp_axis]
    tp = mesh_shape[1]
    kvpe = 64
    seq_len_local = seq_len_cache // sp

    cache = init_kvpe_cache(kvpe, mesh_device, seq_len_cache, mesh_shape, sp_axis, num_kvpe_cache_layers=1)

    # seed all-ones across the whole cache
    ones = torch.ones(1, 1, seq_len_local, kvpe, dtype=torch.bfloat16)
    tt_ones = ttnn.from_torch(
        ones,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    ttnn.fill_cache(cache, tt_ones, 0, update_idx=0)
    ttnn.synchronize_device(mesh_device)

    if expected_chip is not None:
        tile_start = (valid_global // 32) * 32
        chip = (tile_start % chunk_size_global) // (chunk_size_global // sp)  # block-cyclic boundary chip
        assert chip == expected_chip, f"v={valid_global} -> chip {chip}, expected {expected_chip}"
        logger.info(f"valid_global={valid_global} boundary chip={chip}")

    # ---- the device op ----
    ttnn.experimental.deepseek_prefill.zero_padded_kv_cache(
        cache, 0, 0, 1, valid_global, chunk_size_global, sp_axis, 128
    )
    ttnn.synchronize_device(mesh_device)

    # ---- reconstruct natural order ----
    positions = blockcyclic_positions(sp, chunk_size_global, seq_len_cache)
    natural = torch.zeros(seq_len_cache)
    for di, dt in enumerate(ttnn.get_device_tensors(cache)):
        if di % tp != 0:
            continue
        sp_coord = di // tp
        rows = ttnn.to_torch(dt).float()[0, 0, :, :].mean(dim=-1)
        for lr in range(seq_len_local):
            natural[int(positions[sp_coord * seq_len_local + lr].item())] = rows[lr].item()

    ceil_v = math.ceil(valid_global / 128) * 128
    real, pad, rest = natural[:valid_global], natural[valid_global:ceil_v], natural[ceil_v:]
    logger.info(
        f"v={valid_global} ceil128={ceil_v}: real.min={real.min():.2f} "
        f"pad.max={pad.max() if len(pad) else 0:.2f} rest.min={rest.min() if len(rest) else 1:.2f}"
    )
    assert real.min() > 0.9, f"real [0,{valid_global}) clobbered (min={real.min()})"
    if len(pad):
        assert pad.max() < 0.1, f"pad [{valid_global},{ceil_v}) not zeroed (max={pad.max()})"
    if len(rest):
        assert rest.min() > 0.9, f"rows past window [{ceil_v},:) touched (min={rest.min()})"
    logger.success(f"zero_padded_kv_cache v={valid_global} PASSED")


# (num_layers, num_users, slot_idx, layer_idx, valid_global, expected_chip) -- exercises the cache
# batch-slot linearization batch_idx = slot_idx*num_layers + layer_idx (layer_idx>0, slot_idx>0) and
# that the op touches ONLY the target slot. Full 61-layer model + multi-user kv cache.
_MULTI_CASES = [
    (61, 1, 0, 0, 2600, 4),  # full 61-layer cache, user0, layer0
    (61, 1, 0, 60, 2600, 4),  # full 61-layer cache, user0, LAST layer (batch=60)
    (61, 2, 1, 60, 2600, 4),  # full 61-layer cache, user1, LAST layer (batch=121)
    (8, 2, 1, 5, 4512, 7),  # multi-user, user1 layer5 (batch=13), full-tile window
]
_MULTI_IDS = [f"L{nl}_U{nu}_slot{s}_layer{ly}_v{v}" for (nl, nu, s, ly, v, _ch) in _MULTI_CASES]


@pytest.mark.parametrize("mesh_device", [(8, 4)], ids=["8x4"], indirect=True)
@pytest.mark.parametrize(
    "num_layers,num_users,slot_idx,layer_idx,valid_global,expected_chip", _MULTI_CASES, ids=_MULTI_IDS
)
@pytest.mark.timeout(0)
def test_zero_padded_kv_cache_layers_users(
    mesh_device, num_layers, num_users, slot_idx, layer_idx, valid_global, expected_chip
):
    """Seed every (user, layer) cache slot to all-ones, zero ONE slot's pad window, and assert the
    target slot's window is zeroed AND every other slot is untouched -- validating the
    batch_idx = slot_idx*num_layers + layer_idx addressing across a full 61-layer / multi-user cache."""
    mesh_shape = list(mesh_device.shape)
    sp_axis = 0
    sp = mesh_shape[sp_axis]
    tp = mesh_shape[1]
    kvpe = 64
    chunk_size_global = 5120
    seq_len_cache = 5120
    seq_len_local = seq_len_cache // sp
    num_batches = num_users * num_layers
    target_batch = slot_idx * num_layers + layer_idx

    cache = init_kvpe_cache(
        kvpe,
        mesh_device,
        seq_len_cache,
        mesh_shape,
        sp_axis,
        num_kvpe_cache_layers=num_layers,
        num_users=num_users,
    )

    # seed EVERY batch slot to all-ones so a wrongly-touched slot shows up as a non-1 reading
    ones = torch.ones(1, 1, seq_len_local, kvpe, dtype=torch.bfloat16)
    tt_ones = ttnn.from_torch(
        ones,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    for b in range(num_batches):
        ttnn.fill_cache(cache, tt_ones, b, update_idx=0)
    ttnn.synchronize_device(mesh_device)

    logger.info(
        f"layers={num_layers} users={num_users} -> {num_batches} slots; "
        f"zeroing slot={slot_idx} layer={layer_idx} -> target_batch={target_batch}"
    )

    # ---- the device op (only the target (slot, layer)) ----
    ttnn.experimental.deepseek_prefill.zero_padded_kv_cache(
        cache, slot_idx, layer_idx, num_layers, valid_global, chunk_size_global, sp_axis, 128
    )
    ttnn.synchronize_device(mesh_device)

    # ---- verify: target slot window zeroed, all other slots untouched ----
    positions = blockcyclic_positions(sp, chunk_size_global, seq_len_cache)
    natural = torch.zeros(seq_len_cache)
    other_min = 1.0
    for di, dt in enumerate(ttnn.get_device_tensors(cache)):
        if di % tp != 0:
            continue
        sp_coord = di // tp
        t = ttnn.to_torch(dt).float()  # [num_batches, 1, seq_len_local, kvpe]
        for b in range(num_batches):
            rows = t[b, 0, :, :].mean(dim=-1)
            if b == target_batch:
                for lr in range(seq_len_local):
                    natural[int(positions[sp_coord * seq_len_local + lr].item())] = rows[lr].item()
            else:
                other_min = min(other_min, rows.min().item())

    ceil_v = math.ceil(valid_global / 128) * 128
    real, pad, rest = natural[:valid_global], natural[valid_global:ceil_v], natural[ceil_v:]
    logger.info(
        f"target_batch={target_batch}: real.min={real.min():.2f} pad.max={pad.max():.2f} "
        f"rest.min={rest.min():.2f} | other_slots.min={other_min:.2f}"
    )
    assert other_min > 0.9, f"a non-target slot was corrupted (min={other_min})"
    assert real.min() > 0.9, f"target real [0,{valid_global}) clobbered (min={real.min()})"
    assert pad.max() < 0.1, f"target pad [{valid_global},{ceil_v}) not zeroed (max={pad.max()})"
    assert rest.min() > 0.9, f"target rows past window touched (min={rest.min()})"
    logger.success(f"layers={num_layers} users={num_users} slot={slot_idx} layer={layer_idx} PASSED")
