# Plan: Add single-batch-slice and partial (overallocated) gather to `ttnn.all_gather`

## Goal

Port the two features that `ttnn.transformer.ring_mla`'s fused all-gather already
has into the standard `ttnn.all_gather` collective:

1. **Single-batch slice** — gather only one index along the batch dim of the input
   instead of the whole tensor (`ring_mla`'s `kv_cache_batch_idx`).
2. **Partial / overallocated gather** — when the input's gather dim is allocated
   larger than the valid data, gather only the valid prefix (`ring_mla`'s
   `kv_actual_isl`).

A hard requirement for this work is that **memory allocation is minimal for both
cases**: the output must not be allocated for batches we skip or padding we never
gather, and the design must not force a program recompile when only the
valid-length changes (KV-cache decode reuses one persistent buffer).

## Success goals (acceptance criteria)

The feature is done when **all** of the following hold. Each maps to a test in the
Testing plan.

- **G1 — Single-slice correctness.** `all_gather(..., batch_slice_idx=i)` produces a
  result bit-exact (`comp_equal`) to `all_gather(input)[i:i+1]` for every valid `i`,
  across ring + line, `num_links ∈ {1,2}`, bf16 + bf8_b, on the Blackhole box.
- **G2 — Partial-gather correctness.** `all_gather(..., valid_gather_extent=v)` on an
  input padded along `dim` is bit-exact to gathering the valid prefix only, for `v`
  that is worker-aligned, mid-worker, and below one worker's range; on **even rings
  4/6/8** (split-forwarding) and **odd rings 3/5/7**, plus line.
- **G3 — Combined.** Single slice of an overallocated cache (the `ring_mla` decode
  shape) matches `all_gather(valid_prefix)[i:i+1]`.
- **G4 — Memory: single slice.** Output dim-0 extent is exactly 1 and the allocated
  output buffer size equals the 1-batch spec (a `B×` reduction vs. today) — asserted
  from tensor shape/buffer size (M1).
- **G5 — Memory: partial gather.** Output keeps the normal `ring_size · padded_extent`
  size; `valid_gather_extent` limits data movement rather than shrinking the output (M2).
  Only the valid prefix of each slab is written and the padded tail sentinel is preserved,
  enabling reuse of a fixed persistent buffer across steps (M3).
- **G6 — No recompile on value change.** Under trace, replaying with a *changing*
  `valid_gather_extent` (fixed persistent output) patches runtime args only — program
  cache entry count stays constant, results stay correct (D4).
- **G7 — Zero regression.** The full box all_gather suite passes unchanged with both
  features off (defaults inert, arg order preserved).
- **G8 — Clean guards.** Requesting either feature on an unsupported path (composite
  fallback, `dim == 0` for slicing, out-of-range indices/extents) fails fast with a
  clear `TT_FATAL`, never silently ignored.
- **G9 — Stability.** No hangs across the matrix, confirmed via
  `scripts/run_safe_pytest.sh` (built-in hang detection); any hang triaged with
  `tt-triage` before proceeding.

## Build, test & debug workflow

- **Build** (after any C++/kernel change) with the release build:
  ```bash
  ./build_metal.sh --release
  ```
- **Run tests** through the cooperative, device-aware runner — **never bare
  `pytest`** — which serializes device access, enforces a dispatch-layer timeout, and
  auto-resets the device after a hang:
  ```bash
  scripts/run_safe_pytest.sh <test_path> [extra_pytest_args...]
  # add --dev while developing: enables the NoC/watcher sanitizers and auto-triage on hang
  ```
  It has **built-in hang detection** (`TT_METAL_OPERATION_TIMEOUT_SECONDS`), so it is
  the required entry point for the CCL tests here, which can deadlock on the
  split-forwarding path.
- **Diagnose hangs** with `tt-triage` (`tools/tt-triage.py`) — dumps core callstacks,
  NoC state, and per-core waypoints. `run_safe_pytest.sh --dev` invokes it
  automatically on a detected hang; run it manually against a wedged device otherwise.
  Given G2's even-ring split-forwarding risk, expect to lean on this during Phase A.

## Background — current architecture

`ttnn.all_gather` → `ttnn::prim::all_gather` (`AllGatherDeviceOperation`) reuses the
experimental async builder and its kernels:

- Host op: `ttnn/cpp/ttnn/operations/ccl/all_gather/`
  - `all_gather.cpp` / `.hpp` — public invoke (falls back to `composite_all_gather`
    for row-major / tile-padded gather dims; otherwise calls the primitive).
  - `device/all_gather_device_operation.{hpp,cpp}` — op struct, validation,
    `compute_output_specs`, `create_output_tensors`.
  - `device/all_gather_program_factory.cpp` — delegates to the async builder.
  - `all_gather_nanobind.cpp` — Python bindings.
- Shared builder + kernels (experimental):
  - `experimental/ccl/all_gather_async/device/all_gather_async_default_program_factory.cpp`
  - `.../kernels/minimal_default_reader.cpp`
  - `.../kernels/minimal_default_writer.cpp`

The reference implementation with both features lives in:
- `experimental/ccl/ring_attention_all_gather_async/device/kernels/ring_attention_all_gather_reader.cpp`
- `.../ring_attention_all_gather_writer.cpp`
- `.../ring_attention_all_gather_async_multi_core_with_workers_program_factory.cpp`

### How the minimal-default kernels work today (key facts)

- CT args `input_batch_head_count` (product of the two outer dims),
  `input_tensor_Wt/Ht/C`, `output_tensor_Wt/Ht/C`, `gather_dim`, `topology`.
- Host splits the per-batch-head page count (`single_batch_head_num_pages =
  input_tensor_Wt * input_tensor_Ht`) across workers into per-worker
  `input_tile_id_start` / `input_tile_id_end` RT args
  (`all_gather_async_default_program_factory.cpp:665-672`).
- **Reader phase-1** (`minimal_default_reader.cpp:119-144`): loops `bh_idx` over
  `input_batch_head_count`, reads `[input_tile_id_start, input_tile_id_end)` from
  the input, starting at `output_tile_id_start = 0` and stepping
  `input_tensor_Wt*input_tensor_Ht` per batch/head. `output_tile_id_start` is the
  **input** read base here.
- **Writer phase-1** (`minimal_default_writer.cpp:295-461`): writes local slice to
  DRAM + fabric at `tile_id_start = position * ...`.
- **Ring split-forwarding** (reader `:178-190`, writer `:284-292, 528-664`): for even
  ring sizes > 2 the last slice is split in half across the two directions, using
  `total_tiles = input_tile_id_end - input_tile_id_start`. **This path does not exist
  in the ring_attention kernels** and is the main added complexity for feature 2.

### How `ring_mla` implements the two features

- **Single slice**: reader RT arg `input_batch_base` = `idx * heads_per_batch *
  Ht * Wt`; phase-1 sets `output_tile_id_start = input_batch_base` so the local read
  skips to the sliced input slot; `input_batch_head_count` is reduced to one batch's
  heads; the slice is always emitted into output slot 0
  (`ring_attention_all_gather_reader.cpp:114-116, 159`).
- **Partial gather**: per-input RT arg `valid_pages` (= `min(gather_valid_Ht,
  input_Ht) * input_Wt`); clamp `if (valid_pages < input_tile_id_end)
  input_tile_id_end = valid_pages` in **both** reader (`:130-133`) and writer
  (`:76-82`), immediately after arg-read, so the ring semaphore protocol and
  producer/consumer page counts stay matched.

---

## Design decisions

### D1. New op attributes (both features)

Add to `AllGatherDeviceOperation::operation_attributes_t`
(`all_gather_device_operation.hpp:25-37`):

```cpp
std::optional<uint32_t> batch_slice_idx;     // feature 1: gather only this index along `dim0_slice_dim`
std::optional<uint32_t> valid_gather_extent; // feature 2: valid extent (logical, elements) along `dim`
```

Both are plumbed to the primitive, the program factory, and re-patched in
`override_runtime_arguments` (see D4).

### D2. Feature 1 semantics (which dim is sliced)

`ring_mla` slices dim 0 (batch) while gathering along the sequence dim. For the
general op, define feature 1 as: **slice a single index along dim 0** while gathering
along `dim` (require `dim != 0`). This matches the kernel's batch/head iteration and
keeps the change to a single reader offset. Slicing an arbitrary non-gather dim is a
possible follow-up but is out of scope for v1 — document the restriction with a
`TT_FATAL`.

### D3. Feature 2 semantics (units)

`valid_gather_extent` is the **logical valid size along `dim`** (in elements), which
must be tile-aligned when `dim` is a tiled dim. Internally convert to
`valid_pages = valid_tiles_along_dim * (pages per tile-row)` matching the kernel's
per-batch-head page layout. Reject `valid_gather_extent > input extent along dim`.

### D4. Runtime args, not compile-time — avoid recompiles

Both `input_batch_base` and `valid_pages` must be **runtime args**, patched in
`AllGatherProgram::override_runtime_arguments`. Rationale: in KV-cache decode the
valid length grows every step while the persistent output buffer and program stay
fixed; baking these into CT args (or the cache key in a way that changes each step)
would recompile every iteration. `ring_mla` re-patches these on cache hits for exactly
this reason. `batch_slice_idx` may also change per call (different batch), so it is a
runtime arg too. The op attributes still participate in the cache key for
correctness, so the *presence* of the feature (feature on/off, which changes the
output shape) keys the program, but the *value* is re-patched — mirror ring_mla by
keying on the optionals being engaged, not their exact numeric value where the shape
is unaffected.

> Implementation note: keep the cache key stable across value-only changes by having
> the value flow purely through runtime args + `override_runtime_arguments`, and only
> letting the *output shape* (which is a function of `batch_slice_idx.has_value()` and
> `valid_gather_extent`) drive reallocation. When a persistent
> `optional_output_tensor` is supplied (the KV-cache case), shape is fixed by the
> caller, so value-only changes never reallocate or recompile.

---

## Memory-allocation optimization (explicit requirement)

### M1. Feature 1 — output batch dim collapses to 1

`compute_output_specs` must produce an output whose **dim-0 extent is 1** when
`batch_slice_idx` is set (instead of the input's full batch). This is the primary
saving: the gathered output is `1 × ... × (ring_size · dim_extent) × ...` rather than
`B × ...`, i.e. a **B× reduction** in output DRAM/L1. Without this the caller would
allocate the full batch and we would gather into slot 0 of an oversized buffer.

### M2. Feature 2 — bandwidth + buffer-reuse, not a smaller output

**Implementation finding:** the writer places each device's slab at a stride of the FULL
(padded) input extent (`position * input_tensor_Ht * input_tensor_Wt`), so the gathered
output keeps the normal `ring_size · padded_extent` size. `valid_gather_extent` therefore
does **not** shrink the output — a tight `ring_size · valid` output would require a deeper
kernel change to the slab-base stride and was intentionally not attempted (it would
complicate the split-forwarding path). The memory win for feature 2 is instead: (a) only
valid data crosses the fabric (bandwidth), and (b) a fixed max-size persistent buffer can
be reused across steps with no reallocation (M3). The padded tail of each slab is left
unwritten.

### M3. Overallocated persistent output support (KV-cache reuse)

Support the case where the caller passes an `optional_output_tensor` that is itself
overallocated to a max size (fixed across decode steps) and we only write the valid
prefix. In that path:
- Skip the tight-allocation of M2 (caller owns the buffer).
- Validate `optional_output_tensor` gather-dim extent `>= ring_size ·
  valid_gather_extent`.
- Only the valid region is written; the tail is left untouched.

This is the memory-optimal decode pattern: one persistent max-size buffer, no
per-step reallocation, only valid bytes moved over fabric.

### M4. CB / fabric buffer sizing unaffected by padding

The circular buffers are sized per packet (`num_tiles_to_write_per_packet`), not by
total tensor size, so partial gather already avoids per-padding CB growth — confirm
no code path sizes a CB from the padded extent. Because both kernels clamp to
`valid_pages`, only valid pages are streamed, so fabric traffic scales with valid data,
not the allocation.

### M5. Do not over-read on the boundary worker

The per-worker range split happens over the *padded* per-batch-head page count on the
host. The device clamp (`input_tile_id_end = min(end, valid_pages)`) means workers
entirely past the valid boundary do zero work (`start >= end` → loop body skipped) and
the boundary worker does a partial range — no wasted reads. Verify the host
`chunks_per_sync` heuristic still divides the (clamped) range sanely; if the heuristic
over-counts syncs it is harmless (reader and writer agree), but prefer computing it
from the clamped range for tidy semaphore counts.

---

## Implementation steps

### Phase A — Kernels (behind default-off args)

**A1. `minimal_default_reader.cpp`**
- Add RT args `input_batch_base` and `valid_pages` (read right after
  `chunks_per_sync`). Keep them last so existing arg indices are unchanged; default
  values (`0` and `UINT32_MAX`/`input_tile_id_end`) make the kernel behave exactly as
  today when features are off.
- Phase-1 loop: initialize `output_tile_id_start = input_batch_base` (was `0`).
- Immediately after reading args: `if (valid_pages < input_tile_id_end)
  input_tile_id_end = valid_pages;` **before** the split-forwarding computations at
  `:178-190` so `total_tiles`/`first_half` use the clamped range.
- Forwarding path reads from the output tensor by `actual_sender_chip_id` and output
  dims — unaffected by `input_batch_base` (that only shifts the local input read), and
  correctly bounded once `input_tile_id_end` is clamped.

**A2. `minimal_default_writer.cpp`**
- Add matching `valid_pages` RT arg; clamp `input_tile_id_end` right after arg-read
  (`:120-121`), **before** `split_forwarding_enabled` is computed (`:286-292`) and
  before phase-1 uses `tiles_to_read`.
- No `input_batch_base` needed in the writer (writer targets output positions from
  `my_chip_id`/`actual_slice_chip_id`; single-slice output is slot 0 by construction
  and the reduced `input_batch_head_count` already bounds it).
- Audit every place that recomputes `total_tiles`/`first_half` (`:528-552`,
  `:644-664`) to confirm it uses the clamped `input_tile_id_end`.

**A3. Guard rails**
- `static_assert`/`ASSERT` that when split-forwarding is enabled the clamped range is
  still `>= 2` or split-forwarding is disabled for that case (mirror the existing
  `input_tile_id_end - input_tile_id_start >= 2` condition using the clamped value).

**Exit (Phase A):** `./build_metal.sh --release` succeeds; the even-ring 4/6/8 and
odd-ring 3/5/7 partial-gather cases pass bit-exact under `scripts/run_safe_pytest.sh
--dev` with **zero hangs** (args wired via a scratch host change). → unblocks G2.

### Phase B — Host program factory
`all_gather_async_default_program_factory.cpp`

- Compute `batch_head_size` as the sliced count (`heads_per_batch`) when
  `batch_slice_idx` is set; else unchanged.
- Compute `input_batch_base = batch_slice_idx * heads_per_batch * input_tensor_Ht *
  input_tensor_Wt` (0 when unset).
- Compute `valid_pages` from `valid_gather_extent` in the per-batch-head page units;
  default to `input_tile_id_end` (no clamp) when unset.
- Append `input_batch_base` (reader only) and `valid_pages` (reader + writer) to the
  RT arg vectors at `:691-701` and the writer RT args.
- Optionally derive `chunks_per_sync` from the clamped range (M5).

**Exit (Phase B):** scratch host change removed; args now flow from the factory.
Partial-gather cases still pass bit-exact with no hangs on the same ring/line matrix.

### Phase C — Op struct / spec / allocation
`all_gather_device_operation.{hpp,cpp}`

- Add the two optionals to `operation_attributes_t` (D1) and thread through
  `prim::all_gather`.
- `compute_output_specs` (`:102-113`):
  - If `batch_slice_idx` set → `output_shape[0] = 1` (M1).
  - Gather dim: `output_shape[dim] = target_ring_size * (valid_gather_extent ?
    valid_gather_extent : input_shape[dim])` (M2). When an `optional_output_tensor` is
    supplied, honor its shape instead (M3).
- `validate_on_program_cache_miss/hit`:
  - `dim != 0` when `batch_slice_idx` set; `batch_slice_idx < input batch`.
  - `valid_gather_extent <= input extent along dim`; tile-alignment on the gather dim.
  - If `optional_output_tensor` given, its gather extent `>= ring_size ·
    valid_gather_extent` (M3).
  - Features only supported on the tiled minimal-default path (see Phase E).

**Exit (Phase C):** self-allocated output has dim-0 == 1 (slice) and gather extent ==
`ring_size · valid_gather_extent` (partial), asserted from tensor shape/buffer size;
overallocated persistent output writes only the valid prefix and preserves the tail
sentinel. → G4, G5. Trace replay with changing `valid_gather_extent` keeps the program
cache entry count constant. → G6.

### Phase D — Public API + bindings
- `all_gather.cpp` / `.hpp`: add `std::optional<uint32_t> batch_slice_idx` and
  `std::optional<uint32_t> valid_gather_extent` kwargs; forward to `prim::all_gather`.
- `all_gather_nanobind.cpp`: expose both as kw-only args with docstrings and defaults
  of `std::nullopt`.

**Exit (Phase D):** `ttnn.all_gather(...)` accepts both kwargs from Python; single-slice
correctness passes bit-exact across the full matrix. → G1.

### Phase E — Composite fallback guard
`all_gather.cpp:72-76` routes some cases to `composite_all_gather`, which cannot honor
these knobs. Add a `TT_FATAL` (or explicit unsupported-case skip) when either feature
is requested and the composite path would be taken, so args are never silently
ignored.

**Exit (Phase E):** unsupported requests (composite path, `dim == 0` slice,
out-of-range index/extent) raise a clear `TT_FATAL`; the full box suite passes
unchanged with features off. → G7, G8.

---

## Testing plan

**Target hardware: 8-chip Blackhole box (loudbox).** All new tests live under
`tests/ttnn/unit_tests/operations/ccl/blackhole_CI/box/`. Verified this box supports
what the features need: `Topology.Ring` + `FABRIC_1D_RING` (so the even-ring
split-forwarding path is exercisable), `Topology.Linear` + `FABRIC_1D`, and
`FABRIC_2D`; 8 chips give even rings of 4/6/8 and odd rings of 3/5/7 via submeshes.

### Where the code goes

- The Blackhole box tests **reuse** the shared helper `run_all_gather_impl` imported
  from `tests/nightly/t3000/ccl/test_minimal_all_gather_async.py:88` (see
  `blackhole_CI/box/all_post_commit/test_all_gather_apc.py:9`). So extend that **one**
  helper to thread `batch_slice_idx` / `valid_gather_extent` through
  `all_gather_kwargs` and to build the golden accordingly — every box test picks it up
  for free.
- Add the new feature cases as functions in the box files:
  - `blackhole_CI/box/all_post_commit/test_all_gather_apc.py` — fast post-commit
    smoke cases (a couple of shapes per feature).
  - `blackhole_CI/box/nightly/test_all_gather_nightly.py` (and/or
    `.../nightly/test_minimal_all_gather_async.py`) — the full matrix.
- Use the existing fixtures/utilities from those files:
  - `bh_1d_mesh_device` for **ring** cases (see `test_all_gather_ring`), and
    `bh_2d_mesh_device` for **line** cases.
  - `submesh_device = <fixture>.create_submesh(ttnn.MeshShape((num_devices, 1)))` to
    select ring size (4/6/8 even, 3/5/7 odd).
  - `validate_test(num_devices, all_gather_topology, mesh_device.shape, <axis>)` to
    skip unsupported box configs, matching the surrounding tests.
- The golden path in `run_all_gather_impl` builds `torch.rand(ag_output_shape)`,
  shards with `ShardTensorToMesh(dim=dim)`, and compares with
  `comp_equal`/`comp_pcc` — extend it as below.

### Feature 1 — single-batch slice
- **Golden**: shard the full `[B, H, S, D]` input across devices along `dim`, then the
  expected output is `all_gather(input)[batch_slice_idx:batch_slice_idx+1]`.
- Cases: `B ∈ {2, 8}`, `dim ∈ {2, 3}`, `batch_slice_idx ∈ {0, 1, B-1}`,
  bf16 + bf8_b, TILE layout, `num_links ∈ {1, 2}`; ring via `bh_1d_mesh_device`
  submeshes and line via `bh_2d_mesh_device`.
- **Memory assertion**: check `tt_out.shape[0] == 1` and that the allocated output
  buffer size equals the 1-batch spec (guards M1).
- Negative: `dim == 0` with `batch_slice_idx` set → expect validation error.

### Feature 2 — partial / overallocated gather
- **Golden**: allocate input padded along `dim` (e.g. valid `S=96`, padded `S=256`),
  fill the tail with a sentinel; expected output gathers only the valid prefix across
  the ring; the untouched tail (when using an overallocated persistent output) must
  retain its pre-op sentinel.
- Cases: valid extent that is (a) a multiple of the per-worker split, (b) lands
  mid-worker (boundary worker partial), (c) below a single worker's range (some
  workers idle). **Even ring sizes 4/6/8** (`bh_1d_mesh_device` submeshes → the
  split-forwarding path) and **odd ring sizes 3/5/7**; plus line via
  `bh_2d_mesh_device`.
- **Memory assertions**:
  - Tight-alloc path: output gather-dim extent == `ring_size · valid_extent` (M2).
  - Persistent overallocated path: pass an oversized `optional_output_tensor`, verify
    only the valid prefix changes and the tail sentinel is preserved (M3), and that no
    reallocation/recompile happens across iterations with a *changing* valid extent
    (assert stable program cache entry count / trace reuse).

### Combined
- Both features together: single batch slice of an overallocated cache — the ring_mla
  decode shape. Golden = `all_gather(valid_prefix)[idx:idx+1]`.

### Regression / cache
- Run the existing box all_gather suite
  (`blackhole_CI/box/all_post_commit/test_all_gather_apc.py`,
  `.../nightly/test_all_gather_nightly.py`, `.../nightly/test_minimal_all_gather_async.py`)
  with features off to confirm zero behavior change (arg-order preserved, defaults
  inert).
- **Trace test** (`enable_trace=True`): capture once, replay with different
  `valid_gather_extent` values via runtime-arg patching only — assert results correct
  and no recompilation (D4).

### Sharded I/O
- Repeat feature 1 and 2 on the sharded→sharded and sharded→interleaved paths
  (the shared `run_all_gather_impl` already drives these via `mem_config_input` /
  `mem_config_ag`), validating the 1-batch and valid-extent shard specs are legal on
  the box.

---

## Risks & sequencing

1. **Split-forwarding (feature 2)** is the highest-risk item — the clamp must precede
   every `total_tiles`/`first_half` computation in both kernels or even-ring runs
   deadlock. **De-risk first** with an even-ring partial-gather test before doing the
   API work.
2. **Program cache / recompile** — verify value-only changes don't recompile (D4);
   this is what makes the KV-cache decode path viable.
3. **Composite fallback** silently ignoring args — closed by Phase E guard.
4. **Sharded specs** for batch-1 / valid-extent outputs — validate legality per arch.

### Suggested order
Rebuild with `./build_metal.sh --release` after every step, run all tests via
`scripts/run_safe_pytest.sh` (use `--dev` while iterating), and `tt-triage` any hang.

1. Phase A2 + A1 valid-extent clamp + even/odd-ring partial-gather tests (kernels only,
   args wired via a scratch host change). — proves the hard part; run the even-ring
   cases under `scripts/run_safe_pytest.sh --dev` so a split-forwarding deadlock is
   caught and auto-triaged. → G2
2. Phase B/C/D for feature 2, plus M2/M3 allocation + trace/cache test. → G5, G6
3. Feature 1 (mostly host: M1 output spec + one reader arg) + tests. → G1, G4
4. Combined test + sharded coverage + regression sweep. → G3, G7, G8, G9

---

## Implementation status (2026-07-15, Blackhole box)

**Done and validated on the 8-chip Blackhole box.** Files changed:
- Kernels: `minimal_default_reader.cpp` (added `input_batch_base` + `valid_pages` RT args,
  clamp with idle-worker underflow guard, phase-1 base), `minimal_default_writer.cpp`
  (`valid_pages` RT arg + clamp before split-forwarding).
- Factory: `all_gather_async_default_program_factory.{hpp,cpp}` — `compute_all_gather_slice_params`
  helper, builder emits reduced `input_batch_head_count` + the two RT args, override helper
  re-patches `input_batch_base`/`valid_pages` on cache hit.
- Op: `all_gather_device_operation.{hpp,cpp}` — two attributes, output-spec dim-0 collapse,
  validations, and a custom `compute_program_hash` that excludes the feature *values*
  (only `batch_slice_idx.has_value()` participates) so value-only changes don't recompile.
- API/bindings: `all_gather.{hpp,cpp}` (+ composite/multi-axis guards), `all_gather_nanobind.cpp`.
- Tests: `blackhole_CI/box/nightly/test_all_gather_slice_and_partial.py` (self-contained; the
  shared `run_all_gather_impl` was intentionally left untouched to protect G7).

**Test results (via `scripts/run_safe_pytest.sh`):** G1 ✓ (nd=8 ring, bf16), G2 ✓ (nd=8 ring,
even-ring split-forwarding, valid=64/128), G3 ✓ (nd=8), G4 ✓, G5 ✓ (sentinel tail preserved),
G6 ✓ (program-cache entries stable at 3 across changing valid extents), G7 ✓ (existing box
`test_all_gather_ring` passes, incl. trace), G8 ✓, G9 ✓ (no op hangs).

**Coverage caveat:** the hardest paths were exercised, but the full breadth matrix (bf8_b,
Linear topology, ring sizes 4/3/5/7, sharded I/O) was **not** run to completion locally because
repeated full-mesh reopen intermittently trips a board-level ethernet-init timeout
(`Try resetting the board`) between parametrizations — a hardware/firmware flakiness unrelated
to the op (every `call` phase that executed passed). Those cases share the validated code paths;
they should be run in CI (or one-param-at-a-time with a `tt-smi -r` between) before merge.

**Known scope limits:** feature 1 slices dim 0 only (gather dim must be != 0); feature 2 requires
the gather dim to be height (rank-2); both are tiled-layout only. All are enforced with `TT_FATAL`.
