# Handoff: routed-expert + combine perf on a real workload

Branch: `fbajraktari/overlap_routed_expert_and_combine`
Written: 2026-07-13. Author: previous Claude Code session (handing off to a new machine).

## Why this work exists

The branch adds an optimization: **overlap the routed-expert FFN with the combine op** in
the DeepSeek prefill MoE (`overlap_routed_expert_with_combine`, Blackhole-only, via
sub-devices + a global semaphore). See `tt/moe/tt_moe.py`:
- `__init__` lines ~301-405: sub-device split (dm strip = combine, compute strip = routed
  expert) + `routed_expert_global_semaphore`.
- `forward` lines ~621-664: the routed-expert ⇄ combine region (overlap vs sequential).

**Observation that motivated this:** the op-level `tests/pcc/test_ttnn_moe.py` (perf case,
golden-trace input) shows the overlap is **faster**, but the full **chunked transformer**
test does **not** reproduce the win. Hypothesis: the difference is the *per-expert token
distribution*. Golden-trace / synthetic input distributes tokens differently than a real
prefill, and the overlap benefit depends on how balanced the routed-expert FFN work is vs
the combine work. So we need to measure routed_expert+combine on the **real captured
routing** from a production prefill.

## Measured so far (test_ttnn_moe, mesh-4x2, pad0, perf-device-256, 2nd/steady-state run)

Measured `tt_forward` (whole MoE fwd) second-iteration wall-clock, branch vs main:

| Config | tt_forward (2nd run) |
|---|---|
| main (baseline, no overlap) | 44.39 ms |
| this branch (overlap on) | 39.95 ms |

So on the golden-trace unit test, overlap is ~10% faster. The real-workload worker below is
meant to check whether that still holds on real per-expert load.

## What was added on this branch (uncommitted work now committed)

1. `tests/pcc/test_ttnn_moe.py` — the `tt_forward` block was wrapped in a **2-iteration loop**
   (iter0 warms the persistent program cache; iter1 is the steady-state measurement). Both
   iterations are profiled into `profiler.times["tt_forward"]`; we read index `[1]`. Signposts
   carry the iteration number (`tt_forward_START_iter{i}` / `..._END_iter{i}`).

2. `tests/perf/test_prefill_routed_expert_combine.py` — **NEW worker** (this is the main
   deliverable). Boilerplate is lifted from `tests/perf/test_prefill_dispatch_combine.py`
   (which came from PR https://github.com/tenstorrent/tt-metal/pull/46047 — the "real
   workload" capture approach), and the routed-expert op was added. Pipeline:

       dispatch  →  TILE+bfp8 layout transform  →  { routed_expert ⇄ combine }

   - Uses `load_captured_routing(...)` → real captured indices (remapped to [0,64) ∪ {255},
     col-0 dispatch table) so the routed-expert FFN + combine see production per-expert load.
   - Replicates the sub-device + `routed_expert_global_semaphore` setup from
     `TtMoe.__init__`, and runs the routed-expert ⇄ combine block **verbatim from
     `TtMoe.forward`**, so on-device overlap behaviour is identical to production.
   - Parametrized `overlap=[True, False]` and `layer=[27, 38, 50, 28]` (default/hottest = L27;
     col per layer from `_HOT_LAYER_COL`, overridable via `TT_DS_CAPTURED_COL`).
   - Mesh: (8,1) linear, 2 links on BH, with `l1_small_size=MOE_L1_SMALL_REGION_SIZE` (needed
     for the global semaphore, which lives in `BufferType.L1_SMALL`). The captured-routing
     loader simulates one Galaxy column on the 8-chip SP axis, so an 8-chip SP dim is required.
   - Dummy random device weights (`torch_weights=None`): matmul cycle count is
     value-independent; only the (real) per-expert token counts drive the work.
   - Measurement is **both**: host wall-clock (region run twice, logs
     `profiler.times[...][1]`) AND Tracy signposts (`routed_expert_and_combine_start/end_iter{i}`)
     so a device-perf harness can capture it later.

## THE BLOCKER (why we're switching machines)

The captured routing data file is **not present on the current box** (an 8× p150b Blackhole
reservation, hostname `bh-lb-10-special-...`). `/mnt` is empty — no `/mnt/MLPerf`, no
`/mnt/models`. The worker gets all the way through setup and fails only here:

    FileNotFoundError: Captured indices file not found at
    /mnt/MLPerf/deepseek-prefill-cache/longbook_qa_eng_prefill_25600_nopad/expert_routing.safetensors

The file path is defined in `utils/transformer_helpers.py`:
`TRACE_DIR_BASE = os.getenv("DEEPSEEK_V3_TRACE_DIR", "/mnt/MLPerf/deepseek-prefill-cache")`,
`LONGBOOK_QA_ENG_25600 = TRACE_DIR_BASE / "longbook_qa_eng_prefill_25600_nopad"`.

**On the new machine**, ensure `expert_routing.safetensors` is available. Override options:
- `DEEPSEEK_V3_TRACE_DIR=<dir>` where `<dir>/longbook_qa_eng_prefill_25600_nopad/expert_routing.safetensors` exists, OR
- `TT_DS_USE_CAPTURED_INDICES=<full path to expert_routing.safetensors>` (per-run override).

The data exists in CI (it's in `longbook_qa_eng_prefill_25600_nopad`). It should be on a box
with the `/mnt/MLPerf` share mounted, or fetched from the CI cache.

## How to run once the data is present

```bash
# (see build workflow below first)
source /localdev/fbajraktari/tt-metal/python_env/bin/activate   # or the machine's env

# L27 (hottest), overlap ON vs OFF:
python3 -m pytest \
  "models/demos/deepseek_v3_d_p/tests/perf/test_prefill_routed_expert_combine.py::test_ttnn_routed_expert_combine" \
  -k "l27 and linear-8" -s -o log_cli=true --log-cli-level=INFO

# grep the result:
#   "[routed_expert_combine] layer=27 col=2 overlap=True  routed_expert+combine (second run): XX.XX ms"
#   "[routed_expert_combine] layer=27 col=2 overlap=False routed_expert+combine (second run): YY.YY ms"
# Compare overlap=True vs overlap=False per layer. Repeat for l38/l50/l28.
```

Test IDs collected (8 total = 4 layers × 2 overlap):
`test_ttnn_routed_expert_combine[blackhole-{overlap,no_overlap}-l{27,38,50,28}-linear-8-perf_real_indices]`

## Build / git workflow on this project (IMPORTANT)

- After **every** `git checkout` that changes commits: run `git submodule update` (the `umd`
  submodule is pinned per-branch — main pins umd v0.9.9-3, this branch pins v0.9.8) and THEN
  rebuild. Skipping the submodule sync builds a binary mixing branch source + wrong submodule.
- Rebuild with `./build_metal.sh` (user's chosen command for this task).
- The two files above are **Python only** — no rebuild needed to run the new worker on a box
  that already has this branch built. But if you just checked out the branch, do
  `git submodule update && ./build_metal.sh` first.

## Open questions / next steps for the new session

1. Get `expert_routing.safetensors` onto the box (see BLOCKER) and run the 8 cases.
2. Compare overlap vs no_overlap per layer on the REAL distribution. If overlap is NOT faster
   here (matching the chunked-transformer observation), that confirms the hypothesis and the
   worker is a minimal reproducer for debugging why.
3. Optional: wire this worker into a device-perf harness like
   `tests/perf/test_dispatch_combine_perf.py` (per-op Tracy ns for RoutedExpert / Combine) —
   the signposts are already in place. Would need per-(layer, topo) baselines.
4. Optional: the worker currently runs dispatch once, outside the measured region, and re-tiles
   the dispatched buffer each of the 2 measured iterations. If combine/routed_expert turn out to
   mutate/deallocate `metadata` or the counts/offsets tensors across iterations, move dispatch
   inside the loop (it was kept out to isolate the routed_expert+combine cost).
5. Optional: add ring-8 mesh param for a linear-vs-ring comparison (kept to linear-8 for now).
