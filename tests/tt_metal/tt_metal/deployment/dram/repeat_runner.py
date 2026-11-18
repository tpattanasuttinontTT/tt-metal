# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Runs test_runner.py again and again and records each failure as one row in a CSV file.

Each iteration runs the full DRAM deployment suite through test_runner.py. The script reads the
JSON result of the iteration and adds one CSV row for each of these failures:

- data_mismatch       A "Mismatch on dram_controller=" line (one row per worker core and pattern).
- bank_error          A bank with write or read errors in the per-bank result.
- test_failed         A gtest test that did not end with OK or SKIPPED.
- edc_uncorrectable   A GDDR module with uncorr_rd or uncorr_wr above 0 (needs DRAM_TEST_EDC_CHECK=1).
- process_error       The test process exited with an error and no other failure explains it,
                      or the iteration gave no JSON result.

The script writes the CSV after each iteration, so the rows stay if a later iteration stops the
process. Set the DRAM_TEST_* environment variables before you start the script. Each iteration
uses them.

Run from the root of the repository:

    python3 tests/tt_metal/tt_metal/deployment/dram/repeat_runner.py -n 10
"""

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Optional

import subprocess
import json
import time
import csv
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
TEST_RUNNER = SCRIPT_DIR / "test_runner.py"
# test_runner.py starts build/test/tt_metal/unit_tests_deployment with a relative path, so it must run
# from the repository root: tests/tt_metal/tt_metal/deployment/dram -> 5 levels up.
REPO_ROOT = SCRIPT_DIR.parents[4]

PASS_STATUSES = {"OK", "SKIPPED"}

COLUMNS = [
    "timestamp",
    "iteration",
    "failure_type",
    "test",
    "test_status",
    "chip_id",
    "bdf",
    "bank",
    "core",
    "pattern",
    "repeat",
    "pass",
    "failures",
    "write_failures",
    "read_failures",
    "first_fail_classified_as",
    "checked_bytes",
    "gddr_module",
    "uncorr_rd",
    "uncorr_wr",
    "corr_rd",
    "corr_wr",
    "temp_top",
    "temp_bottom",
    "edc_new",
    "exit_code",
    "json",
    "log",
]


def shortname(n: str) -> str:
    return n if "_" not in n else n.split("_", 1)[1]


def load_results(path: Path) -> Optional[dict]:
    """Returns {"runs": [...], "edc": [...]}, or None when the file is missing or not valid JSON."""
    try:
        with path.open() as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    # Older test_runner.py versions wrote a bare list of runs.
    if isinstance(data, list):
        return {"runs": data, "edc": []}
    return data


def failure_rows(results: Optional[dict], exit_code: int, prev_uncorr: set) -> tuple[list[dict], set]:
    """Returns the failure rows of one iteration and the set of (dev_id, module) keys with an
    uncorrectable EDC flag. The EDC flags are sticky until chip reset, so `edc_new` marks the
    modules that did not have the flag in the previous iteration."""
    rows: list[dict] = []
    uncorr: set = set()

    if results is None:
        rows.append({"failure_type": "process_error", "exit_code": exit_code})
        return rows, uncorr

    runs = results.get("runs", [])
    for run in runs:
        test = shortname(run["name"])
        status = run["status"]

        if status not in PASS_STATUSES:
            rows.append({"failure_type": "test_failed", "test": test, "test_status": status})

        for e in run.get("errors", []):
            rows.append(
                {
                    "failure_type": "data_mismatch",
                    "test": test,
                    "test_status": status,
                    "chip_id": e["dev_id"],
                    "bdf": e["bdf"],
                    "bank": e["bank"],
                    "core": e["core"],
                    "pattern": e["pattern"],
                    "repeat": e["repeat"],
                    "pass": e["pass"],
                    "failures": e["failures"],
                    "write_failures": e["write_failures"],
                    "read_failures": e["read_failures"],
                    "first_fail_classified_as": e["first_fail_classified_as"],
                }
            )

        for chip_id, chip in run.get("chips", {}).items():
            for bank_id, bank in chip.get("banks", {}).items():
                if bank["write_errors"] == 0 and bank["read_errors"] == 0:
                    continue
                rows.append(
                    {
                        "failure_type": "bank_error",
                        "test": test,
                        "test_status": status,
                        "chip_id": chip_id,
                        "bdf": chip["bdf"],
                        "bank": bank_id,
                        "write_failures": bank["write_errors"],
                        "read_failures": bank["read_errors"],
                        "checked_bytes": bank["checked_bytes"],
                    }
                )

    for e in results.get("edc", []):
        if e["uncorr_rd"] == 0 and e["uncorr_wr"] == 0:
            continue
        key = (e["dev_id"], e["module"])
        uncorr.add(key)
        rows.append(
            {
                "failure_type": "edc_uncorrectable",
                "chip_id": e["dev_id"],
                "bdf": e["bdf"],
                "gddr_module": e["module"],
                "uncorr_rd": e["uncorr_rd"],
                "uncorr_wr": e["uncorr_wr"],
                "corr_rd": e["corr_rd"],
                "corr_wr": e["corr_wr"],
                "temp_top": e["temp_top"],
                "temp_bottom": e["temp_bottom"],
                "edc_new": key not in prev_uncorr,
            }
        )

    # A crash, a dispatch timeout or an abort can give a non-zero exit code with no failure in the
    # parsed log. Record it so that the iteration is not counted as a pass. "No runs" means the
    # suite did not start or the log had no gtest lines.
    if (exit_code != 0 and not rows) or not runs:
        rows.append({"failure_type": "process_error", "exit_code": exit_code})

    return rows, uncorr


def append_rows(csv_path: Path, rows: list[dict]) -> None:
    new_file = not csv_path.exists() or csv_path.stat().st_size == 0
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, restval="")
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def run_iteration(iteration: int, out_dir: Path, show: bool, verbose: bool) -> tuple[int, Path, Path, bool]:
    """Runs test_runner.py one time. Returns (exit code, json path, log path, interrupted)."""
    json_path = out_dir / f"iter_{iteration:04d}.json"
    log_path = out_dir / f"iter_{iteration:04d}.log"

    cmd = [sys.executable, str(TEST_RUNNER), "-o", str(json_path), "-l", str(log_path)]
    if not show:
        cmd += ["-q", "-n"]
    if verbose:
        cmd.append("-v")

    proc = subprocess.Popen(cmd, cwd=REPO_ROOT)
    interrupted = False
    while True:
        try:
            exit_code = proc.wait()
            break
        except KeyboardInterrupt:
            # Ctrl+C also goes to the child processes. The test stops after its current work and
            # test_runner.py still writes the JSON, so wait for it and record the partial result.
            if interrupted:
                proc.kill()
            interrupted = True
            print("\nInterrupted: waiting for the current iteration to finish (Ctrl+C again to kill)")

    return exit_code, json_path, log_path, interrupted


def main():
    parser = ArgumentParser(description="Run the DRAM deployment suite repeatedly and record failures to a CSV.")
    parser.add_argument("-n", type=int, default=0, help="Number of iterations. 0 = run until Ctrl+C or --hours.")
    parser.add_argument("--hours", type=float, default=0.0, help="Stop starting new iterations after this time.")
    parser.add_argument("-c", "--csv", type=str, default="", help="CSV file. Default: <out-dir>/failures.csv")
    parser.add_argument("-d", "--out-dir", type=str, default="", help="Directory for per-iteration JSON and log files.")
    parser.add_argument("--stop-on-fail", action="store_true", help="Stop after the first iteration with a failure.")
    parser.add_argument("--show", action="store_true", help="Show the test log and summary tables of each iteration.")
    parser.add_argument("-v", action="store_true", help="Pass -v (all log types) to test_runner.py.")
    opts = parser.parse_args()

    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    out_dir = Path(opts.out_dir) if opts.out_dir else Path.cwd() / f"dram_repeat_{stamp}"
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(opts.csv).resolve() if opts.csv else out_dir / "failures.csv"

    print(f"Iteration results: {out_dir}")
    print(f"Failure CSV:       {csv_path}")

    deadline = time.monotonic() + opts.hours * 3600 if opts.hours > 0 else None
    prev_uncorr: set = set()
    failed_iterations = 0
    iteration = 0

    while True:
        if opts.n and iteration >= opts.n:
            break
        if deadline is not None and time.monotonic() >= deadline:
            print("Time limit reached")
            break

        iteration += 1
        started = datetime.now().isoformat(timespec="seconds")
        print(f"\n=== Iteration {iteration}{f'/{opts.n}' if opts.n else ''} started {started} ===")

        exit_code, json_path, log_path, interrupted = run_iteration(iteration, out_dir, opts.show, opts.v)
        rows, prev_uncorr = failure_rows(load_results(json_path), exit_code, prev_uncorr)

        for r in rows:
            r.update(
                {
                    "timestamp": started,
                    "iteration": iteration,
                    "json": str(json_path),
                    "log": str(log_path),
                }
            )
            r.setdefault("exit_code", exit_code)
        if rows:
            append_rows(csv_path, rows)
            failed_iterations += 1

        counts: dict[str, int] = {}
        for r in rows:
            counts[r["failure_type"]] = counts.get(r["failure_type"], 0) + 1
        result = "PASS" if not rows else "FAIL " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"=== Iteration {iteration}: {result} (exit code {exit_code}) ===")

        if interrupted:
            break
        if rows and opts.stop_on_fail:
            print("Stopping on first failure")
            break

    print(f"\n{iteration} iteration(s), {failed_iterations} with failures. CSV: {csv_path}")
    sys.exit(1 if failed_iterations else 0)


if __name__ == "__main__":
    main()
