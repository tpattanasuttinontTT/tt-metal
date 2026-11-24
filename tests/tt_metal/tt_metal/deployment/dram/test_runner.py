# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, AsyncIterator, Iterator, TextIO
from dataclasses import dataclass, asdict
from argparse import ArgumentParser
from asyncio import StreamReader
from enum import Enum, auto
from sys import stdout
import fileinput
import asyncio
import pprint
import json
import re
import os

timeregex = "\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+"
teststart = re.compile("\[\s+RUN\s+\] (.*)")
testend = re.compile("\[\s+([^ ]*)\s+\] (.*) \(.*\)")
testbdf = re.compile(f"({timeregex}).*Test \| \[bdf=(.*)\]\[device_id=(.*)\].*")
testbankerror = re.compile(
    f"({timeregex}).*Test \| bdf=(.*) device_id=(.*) bank_id=(.*) checked_bytes=(.*) write_err=(.*) read_err=(.*) .*"
)
testmismatch = re.compile(
    f"({timeregex}).*Test \| \[bdf=(.*)\]\[device_id=(.*)\]"
    + " Mismatch on dram_controller=(.*) core (.*) pattern=(.*) repeat=(.*)"
    + " pass=(.*): failures=(.*), first_fail_classified_as=(.*),"
    + " write_failures=(\d+), read_failures=(\d+)(?:, words_checked=(\d+))?"
)
# Per-bank NCRISC DRAM bandwidth line emitted by log_bandwidth_summary() in test_dram.cpp.
testdrambw = re.compile(
    f"({timeregex}).*Test \| DRAM_BW bdf=(\S+) device_id=(\S+) bank=(\S+) bytes=(\S+) "
    + "write_gbps=(\S+) read_gbps=(\S+) write_bpc=(\S+) read_bpc=(\S+)"
)
# Per-GDDR-module EDC telemetry line emitted by log_edc_counter_summary() in test_dram.cpp.
testgddredc = re.compile(
    f"({timeregex}).*Test \| GDDR_EDC bdf=(\S+) device_id=(\S+) module=(\S+) "
    + "temp_top=(\S+) temp_bottom=(\S+) corr_rd=(\S+) corr_wr=(\S+) uncorr_rd=(\S+) uncorr_wr=(\S+)"
)
# Cycle line emitted by the t6py-like retention tests (DramDeployment_T6pyLikeRetention*) in test_dram.cpp.
testretentioncycle = re.compile(
    f"({timeregex}).*Test \| RETENTION_CYCLE phase=(\S+) index=(\d+) total=(\d+) loop=(\d+) pattern=(\S+) pass=(\d+)"
)
# Per-bank bandwidth of one phase of the t6py-like retention tests.
testt6pybw = re.compile(
    f"({timeregex}).*Test \| T6PY_BW bdf=(\S+) device_id=(\S+) bank=(\d+) phase=(\S+) bytes=(\d+) ticks=(\d+) gbps=(\S+)"
)
# Markers that this runner writes around the process-split t6py-like retention case.
retentionsplit = re.compile("RETENTION_SPLIT (begin|end) case=(\S+)(?: status=(\S+))?")

# gtest binary, and the tests that the t6py-like retention case runs, one process per phase.
PROGRAM = "build/test/tt_metal/unit_tests_deployment"
PHASE_WRITE = "MeshDispatchFixture.DramDeployment_T6pyLikeRetentionWrite"
PHASE_READ = "MeshDispatchFixture.DramDeployment_T6pyLikeRetentionRead"
# The single-phase tests are not part of the main run. The runner runs them itself (see run_t6py_like_retention).
MAIN_FILTER = "*DramDeployment_*:-*T6pyLikeRetention*"
RETENTION_CASE = "T6pyLikeRetention"
# Time between the write process and the read process. No process has the device open during this time.
DEFAULT_RETENTION_SLEEP_S = 25

print_logs = True


@dataclass
class TestedBank:
    checked_bytes: int
    write_errors: int
    read_errors: int
    write_gbps: float = 0.0
    read_gbps: float = 0.0
    write_bpc: float = 0.0
    read_bpc: float = 0.0

    def to_dict(self):
        return self.__dict__


@dataclass
class TestedChips:
    bdf: str
    banks: dict[str, TestedBank]

    def to_dict(self):
        return {
            "bdf": self.bdf,
            "banks": {i: self.banks[i].to_dict() for i in self.banks},
        }


@dataclass
class TestRun:
    name: str
    errors: list[dict]
    chips: dict[str, TestedChips]
    status: str

    def to_dict(self):
        return {
            "name": self.name,
            "errors": self.errors,
            "status": self.status,
            "chips": {i: self.chips[i].to_dict() for i in self.chips},
        }


def results_to_dict(runs: list[TestRun], edc: list[dict]) -> dict:
    """`edc` is the suite-level GDDR EDC read taken once after all tests (DRAM_TEST_EDC_CHECK=1)."""
    return {"runs": [r.to_dict() for r in runs], "edc": edc}


def runs_to_json(runs: list[TestRun], edc: list[dict], **kwargs) -> str:
    return json.dumps(results_to_dict(runs, edc), **kwargs)


def runs_to_jsonf(fname: str, runs: list[TestRun], edc: list[dict], **kwargs) -> None:
    with open(fname, "w") as f:
        json.dump(results_to_dict(runs, edc), f, **kwargs)


class EventType(Enum):
    TESTSTART = auto()
    TESTEND = auto()
    MISMATCH = auto()
    BDF = auto()
    BANKERROR = auto()
    DRAMBW = auto()
    GDDREDC = auto()
    RETENTIONCYCLE = auto()
    RETENTIONSPLIT = auto()
    T6PYBW = auto()


@dataclass
class Event:
    typ: EventType
    extra: dict

    def to_dict(self):
        return {"typ": self.typ.name, "extra": self.extra}


def parse_teststart(l: str) -> Optional[Event]:
    m = teststart.match(l)
    if m is None:
        return None

    testname = m.group(1)
    # print(f"\tTESTSTART '{testname}'")

    return Event(EventType.TESTSTART, {"name": testname})


def parse_testend(l: str) -> Optional[Event]:
    m = testend.match(l)
    if m is None:
        return None

    status = m.group(1)
    testname = m.group(2)
    # print(f"\tTESTEND '{testname}' '{status}'")

    return Event(EventType.TESTEND, {"name": testname, "status": status})


def parse_mismatch(l: str) -> Optional[Event]:
    m = testmismatch.match(l)
    if m is None:
        return None

    extra = {
        "bdf": m.group(2),
        "dev_id": m.group(3),
        "bank": int(m.group(4)),
        "core": m.group(5),
        "pattern": m.group(6),
        "repeat": int(m.group(7)),
        "pass": int(m.group(8)),
        "failures": int(m.group(9)),
        "first_fail_classified_as": m.group(10),
        "write_failures": int(m.group(11)),
        "read_failures": int(m.group(12)),
        # log_dram_failure() in dram_base.cpp does not print words_checked; keep None when absent.
        "words_checked": int(m.group(13)) if m.group(13) is not None else None,
    }
    # print(f"\tMISMATCH '{extra}'")

    return Event(EventType.MISMATCH, extra)


def parse_bdf(l: str) -> Optional[Event]:
    m = testbdf.match(l)
    if m is None:
        return None

    extra = {
        "bdf": m.group(2),
        "dev_id": m.group(3),
    }
    # print(f"\tBDF '{extra}'")

    return Event(EventType.BDF, extra)


def parse_bankerror(l: str) -> Optional[Event]:
    m = testbankerror.match(l)
    if m is None:
        return None

    extra = {
        "bdf": m.group(2),
        "dev_id": m.group(3),
        "bank": int(m.group(4)),
        "checked_bytes": int(m.group(5)),
        "write_err": int(m.group(6)),
        "read_err": int(m.group(7)),
    }
    # print(f"\tBANKERROR '{extra}'")

    return Event(EventType.BANKERROR, extra)


def parse_drambw(l: str) -> Optional[Event]:
    m = testdrambw.match(l)
    if m is None:
        return None

    extra = {
        "bdf": m.group(2),
        "dev_id": m.group(3),
        "bank": int(m.group(4)),
        "bytes": int(m.group(5)),
        "write_gbps": float(m.group(6)),
        "read_gbps": float(m.group(7)),
        "write_bpc": float(m.group(8)),
        "read_bpc": float(m.group(9)),
    }
    # print(f"\tDRAMBW '{extra}'")

    return Event(EventType.DRAMBW, extra)


def parse_gddredc(l: str) -> Optional[Event]:
    m = testgddredc.match(l)
    if m is None:
        return None

    extra = {
        "bdf": m.group(2),
        "dev_id": m.group(3),
        "module": int(m.group(4)),
        "temp_top": float(m.group(5)),
        "temp_bottom": float(m.group(6)),
        "corr_rd": int(m.group(7)),
        "corr_wr": int(m.group(8)),
        "uncorr_rd": int(m.group(9)),
        "uncorr_wr": int(m.group(10)),
    }
    # print(f"\tGDDREDC '{extra}'")

    return Event(EventType.GDDREDC, extra)


def parse_retentioncycle(l: str) -> Optional[Event]:
    m = testretentioncycle.match(l)
    if m is None:
        return None

    extra = {
        "phase": m.group(2),
        "index": int(m.group(3)),
        "total": int(m.group(4)),
        "loop": int(m.group(5)),
        "pattern": m.group(6),
        "pass": int(m.group(7)),
    }

    return Event(EventType.RETENTIONCYCLE, extra)


def parse_t6pybw(l: str) -> Optional[Event]:
    m = testt6pybw.match(l)
    if m is None:
        return None

    extra = {
        "bdf": m.group(2),
        "dev_id": m.group(3),
        "bank": int(m.group(4)),
        "phase": m.group(5),
        "bytes": int(m.group(6)),
        "ticks": int(m.group(7)),
        "gbps": float(m.group(8)),
    }

    return Event(EventType.T6PYBW, extra)


def parse_retentionsplit(l: str) -> Optional[Event]:
    m = retentionsplit.match(l)
    if m is None:
        return None

    return Event(EventType.RETENTIONSPLIT, {"what": m.group(1), "case": m.group(2), "status": m.group(3)})


def parse_line(l: str, outf: Optional[TextIO]) -> Optional[Event]:
    if print_logs:
        print(f"l: {l}")

    if outf:
        print(f"l: {l}", file=outf)

    parsers = [
        parse_teststart,
        parse_bankerror,
        parse_drambw,
        parse_gddredc,
        parse_mismatch,
        parse_testend,
        parse_bdf,
        parse_retentioncycle,
        parse_t6pybw,
        parse_retentionsplit,
    ]

    for p in parsers:
        if r := p(l):
            return r

    return None


async def parse_logs_stream(inf: asyncio.StreamReader, outf: Optional[TextIO]) -> AsyncIterator[Event]:
    while True:
        l = await inf.readline()
        if l == b"":
            break

        r = parse_line(l.decode("utf-8").strip(), outf)
        if r is not None:
            yield r


async def parse_logs(inf: asyncio.StreamReader, outf: Optional[TextIO]) -> list[Event]:
    evs = []
    async for e in parse_logs_stream(inf, outf):
        evs.append(e)

    return evs


def parse_evs(evs: list[Event], bdfs: dict[str, str], edc: list[dict]) -> Iterator[TestRun]:
    """Yields one TestRun per gtest test. GDDR EDC lines are suite-level (printed once after the
    last test), so they are appended to `edc` instead of being attached to a test."""
    test: str = ""
    runs: list[TestRun] = []
    errors: list[dict] = []
    chips: dict[str, TestedChips] = {}

    stresstest = "MeshDispatchFixture.TensixDeploymentEthernet05StressTest"
    drambidir = "MeshDispatchFixture.TensixDeploymentEthernet04DataIntegrityDramBidir"
    noprocs = [drambidir]

    def ensure_chip(extra: dict):
        chipid = extra["dev_id"]
        if chipid not in chips:
            chips[chipid] = TestedChips(extra["bdf"], {})

    # T6PY_BW values of the current run, by (dev_id, bank, phase). A process-split run has one value per cycle;
    # the bank shows the mean.
    t6pybw: dict[tuple[str, int, str], list[float]] = {}

    def apply_t6pybw():
        for (dev_id, bank_id, phase), values in t6pybw.items():
            banks = chips[dev_id].banks
            bank = banks.get(bank_id)
            if bank is None:
                bank = TestedBank(0, 0, 0)
                banks[bank_id] = bank
            mean = sum(values) / len(values)
            if phase == "write":
                bank.write_gbps = mean
            else:
                bank.read_gbps = mean
        t6pybw.clear()

    # Inside a process-split case, the gtest runs of all its processes are merged into one TestRun.
    split_case: Optional[str] = None

    it = iter(evs)
    for e in it:
        if e.typ == EventType.RETENTIONSPLIT:
            if e.extra["what"] == "begin":
                split_case = e.extra["case"]
            else:
                apply_t6pybw()
                name = f"MeshDispatchFixture.DramDeployment_{e.extra['case']}"
                runs.append(TestRun(name, errors, chips, e.extra["status"] or "FAILED"))
                split_case = None
            errors = []
            chips = {}
        elif e.typ == EventType.TESTSTART:
            if split_case is None:
                test = e.extra["name"]
                errors = []
        elif e.typ == EventType.TESTEND:
            if split_case is None:
                apply_t6pybw()
                runs.append(TestRun(test, errors, chips, e.extra["status"]))
                errors = []
                chips = {}
        elif e.typ == EventType.T6PYBW:
            ensure_chip(e.extra)
            t6pybw.setdefault((e.extra["dev_id"], e.extra["bank"], e.extra["phase"]), []).append(e.extra["gbps"])
        elif e.typ == EventType.MISMATCH:
            errors.append(e.extra)
        elif e.typ == EventType.BDF:
            bdfs[e.extra["dev_id"]] = e.extra["bdf"]
        elif e.typ == EventType.BANKERROR:
            ensure_chip(e.extra)
            b = TestedBank(e.extra["checked_bytes"], e.extra["write_err"], e.extra["read_err"])
            chips[e.extra["dev_id"]].banks[e.extra["bank"]] = b
            # print(chips)
        elif e.typ == EventType.DRAMBW:
            ensure_chip(e.extra)
            banks = chips[e.extra["dev_id"]].banks
            bank = banks.get(e.extra["bank"])
            if bank is None:
                bank = TestedBank(e.extra["bytes"], 0, 0)
                banks[e.extra["bank"]] = bank
            bank.write_gbps = e.extra["write_gbps"]
            bank.read_gbps = e.extra["read_gbps"]
            bank.write_bpc = e.extra["write_bpc"]
            bank.read_bpc = e.extra["read_bpc"]
        elif e.typ == EventType.GDDREDC:
            bdfs[e.extra["dev_id"]] = e.extra["bdf"]
            # Each test process reads EDC once at its end. The counters are cumulative since reset,
            # so keep only the latest reading of each module.
            edc[:] = [x for x in edc if (x["dev_id"], x["module"]) != (e.extra["dev_id"], e.extra["module"])]
            edc.append(e.extra)

    yield from runs


def table(title: str, headers: list[str], rows: list[list]) -> str:
    rows_str = [[str(x) for x in row] for row in rows]
    all_rows = [headers] + rows_str
    widths = [max(len(row[i]) for row in all_rows) for i in range(len(headers))]

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    out = []
    out.append("")
    out.append(title)
    out.append(sep)
    out.append("| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |")
    out.append(sep)
    for row in rows_str:
        out.append("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |")
    out.append(sep)
    return "\n".join(out)


def shortname(n: str) -> str:
    return n if "_" not in n else n.split("_")[1]


def print_summary(runs: list[TestRun], outf: TextIO = stdout):
    headers = ["test name", "status"]
    rows = []
    for r in runs:
        rows.append([shortname(r.name), r.status])

    print(table("Test summary", headers, rows), file=outf)


def print_test_summary(r: TestRun, bdfs: dict[str, str], outf: TextIO = stdout):
    passes = [[True for i in range(8)] for i in range(32)]
    for e in r.errors:
        passes[int(e["dev_id"])][int(e["bank"])] = False

    header = ["chip id", "bdf"]
    for i in range(8):
        header.append(f"D{i}")

    rows = []
    for d in range(len(passes)):
        if not str(d) in bdfs:
            continue
        row = [d, bdfs[str(d)]]
        for i in range(len(passes[d])):
            row.append("PASS" if passes[d][i] else "FAIL")
        rows.append(row)

    rows.sort(key=lambda x: x[1])

    print(table(f"{shortname(r.name)} summary", header, rows), file=outf)


def print_test_summaries(runs: list[TestRun], bdfs: dict[str, str], outf: TextIO = stdout):
    for r in runs:
        print_test_summary(r, bdfs, outf)


def print_test_bandwidth_summary(r: TestRun, bdfs: dict[str, str], outf: TextIO = stdout):
    """Per-controller NCRISC DRAM bandwidth as read/write GB/s (from the DRAM_BW log lines).

    Separate from the PASS/FAIL summary; cells show "<read>/<write>" GB/s, or "-" when no
    bandwidth was reported for that controller.
    """
    header = ["chip id", "bdf"]
    for i in range(8):
        header.append(f"D{i}")

    rows = []
    for d in range(32):
        if not str(d) in bdfs:
            continue
        chip = r.chips.get(str(d))
        row = [d, bdfs[str(d)]]
        for i in range(8):
            bank = chip.banks.get(i) if chip else None
            if bank is not None and (bank.read_gbps or bank.write_gbps):
                row.append(f"{bank.read_gbps:.1f}/{bank.write_gbps:.1f}")
            else:
                row.append("-")
        rows.append(row)

    rows.sort(key=lambda x: x[1])

    print(table(f"{shortname(r.name)} bandwidth GB/s (read/write)", header, rows), file=outf)


def print_test_bandwidth_summaries(runs: list[TestRun], bdfs: dict[str, str], outf: TextIO = stdout):
    for r in runs:
        print_test_bandwidth_summary(r, bdfs, outf)


def print_edc_summary(edc: list[dict], outf: TextIO = stdout):
    """Per-GDDR-module EDC telemetry as a bordered table, one metric per cell.

    Read once at the end of the suite, only when run with DRAM_TEST_EDC_CHECK=1 (Blackhole
    firmware telemetry). Counters are cumulative since reset; corrected saturates at 255.
    """
    if not edc:
        return

    header = ["chip id", "bdf", "GDDR", "TempTop", "TempBot", "CorrRd", "CorrWr", "UncorrRd", "UncorrWr"]
    rows = []
    for e in edc:
        rows.append(
            [
                e["dev_id"],
                e["bdf"],
                e["module"],
                f'{e["temp_top"]:.1f}C',
                f'{e["temp_bottom"]:.1f}C',
                e["corr_rd"],
                e["corr_wr"],
                e["uncorr_rd"],
                e["uncorr_wr"],
            ]
        )

    rows.sort(key=lambda x: (x[1], int(x[2])))

    print(table("GDDR EDC telemetry (end of suite)", header, rows), file=outf)


def print_results(runs: list[TestRun], bdfs: dict[str, str], edc: list[dict], outf: TextIO = stdout):
    print_test_summaries(runs, bdfs, outf)
    print_test_bandwidth_summaries(runs, bdfs, outf)
    print_edc_summary(edc, outf)
    print_summary(runs, outf)


async def file_to_streamreader(path: str, chunk_size: int = 8192) -> StreamReader:
    loop = asyncio.get_running_loop()
    reader = StreamReader()

    def feed_blocking():
        with open(path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                data = chunk
                loop.call_soon_threadsafe(reader.feed_data, data)
        loop.call_soon_threadsafe(reader.feed_eof)

    loop.run_in_executor(None, feed_blocking)
    return reader


async def parse_file(file: str, outf: Optional[TextIO]) -> list[Event]:
    sr = await file_to_streamreader(file)
    return await parse_logs(sr, outf)


def note(msg: str, outf: Optional[TextIO]) -> None:
    """Runner progress message: shown with the test log and written to the log file."""
    if print_logs:
        print(msg)
    if outf:
        print(msg, file=outf)


async def run_gtest(gtest_filter: str, env: dict, outf: Optional[TextIO]) -> tuple[int, list[Event]]:
    """Runs the gtest binary in a new process and parses its log."""
    proc = await asyncio.create_subprocess_exec(
        PROGRAM, f"--gtest_filter={gtest_filter}", stdout=asyncio.subprocess.PIPE, env=env
    )
    exit_status, evs = await asyncio.gather(proc.wait(), parse_logs(proc.stdout, outf))
    return exit_status, evs


def finished_statuses(evs: list[Event]) -> list[str]:
    return [e.extra["status"] for e in evs if e.typ == EventType.TESTEND]


async def run_t6py_like_retention(env: dict, sleep_s: int, outf: Optional[TextIO]) -> tuple[int, list[Event]]:
    """Runs the t6py-like retention case over all cycles. A cycle is one (loop, pattern, pass) of the
    deployment pattern list; the test processes select it with DRAM_TEST_RETENTION_CYCLE.

    For each cycle: a write process writes the whole region of every bank and exits. The runner sleeps for
    sleep_s seconds with no process open on the device. Then a new read process reads the whole region and
    checks it. This is the same as t6py `dram-test ... --skip_reads 1 all`, then `dram-test ... --skip_writes 1 all`.

    Every process logs RETENTION_CYCLE with the total number of cycles. The case stops early if a process
    does not finish its test (crash, watchdog, Ctrl+C).
    """
    evs: list[Event] = []
    exit_status = 0
    status = "OK"

    evs.append(parse_line(f"RETENTION_SPLIT begin case={RETENTION_CASE}", outf))

    steps = [PHASE_WRITE, None, PHASE_READ]

    cycle = 0
    total: Optional[int] = None
    stop = False
    while not stop and (total is None or cycle < total):
        for step in steps:
            if step is None:
                # The sleep step comes after the write step, so total is known here.
                note(
                    f"RETENTION_SPLIT {RETENTION_CASE} cycle {cycle + 1}/{total}: sleep {sleep_s} s, device closed",
                    outf,
                )
                await asyncio.sleep(sleep_s)
                continue

            step_env = dict(env, DRAM_TEST_RETENTION_CYCLE=str(cycle))
            step_status, step_evs = await run_gtest(step, step_env, outf)
            evs.extend(step_evs)
            exit_status = exit_status or step_status

            cycles = [e for e in step_evs if e.typ == EventType.RETENTIONCYCLE]
            if cycles:
                total = cycles[0].extra["total"]

            statuses = finished_statuses(step_evs)
            # gtest marks the test SKIPPED when SetUpTestSuite fails (for example, the device cannot be opened),
            # and the process exits non-zero. A real skip exits 0, and a data mismatch shows as a FAILED test.
            process_failed = step_status != 0 and "FAILED" not in statuses
            if "SKIPPED" in statuses and not process_failed:
                status = "SKIPPED"
                stop = True
            elif process_failed or len(statuses) != 1 or total is None:
                status = "FAILED"
                note(
                    f"RETENTION_SPLIT {RETENTION_CASE} cycle {cycle + 1}: process did not finish its test, stopping",
                    outf,
                )
                stop = True
            elif "FAILED" in statuses:
                # A data mismatch: record it and continue with the next cycles.
                status = "FAILED"

            if stop:
                break

        cycle += 1

    if status != "OK" and exit_status == 0:
        exit_status = 1

    evs.append(parse_line(f"RETENTION_SPLIT end case={RETENTION_CASE} status={status}", outf))
    return exit_status, evs


async def main():
    parser = ArgumentParser()
    parser.add_argument("-q", action="store_true", help="Don't print the logs as the tests run")
    parser.add_argument("-n", action="store_true", help="Don't print the summary")
    parser.add_argument("-v", action="store_true", help="Verbose output")
    parser.add_argument("-i", type=str, help="Input file to parse instead of running the tests")
    parser.add_argument("-o", type=str, help="Output path for the json file")
    parser.add_argument("-l", type=str, help="File path to output logs to")
    parser.add_argument("--no-retention", action="store_true", help="Don't run the t6py-like data-retention case")
    parser.add_argument("--retention-only", action="store_true", help="Run only the t6py-like data-retention case")
    parser.add_argument(
        "--retention-sleep-s",
        type=int,
        default=DEFAULT_RETENTION_SLEEP_S,
        help=f"Seconds between the write and the read process of each retention cycle (default {DEFAULT_RETENTION_SLEEP_S})",
    )
    opts = parser.parse_args()

    logpath = opts.l if opts.l else None
    logf = open(logpath, "w") if logpath else None
    exit_status = 0

    if opts.q:
        global print_logs
        print_logs = False

    logfile = opts.o if opts.o else "out.json"
    print(f"Writing results to '{logfile}'")
    if logf:
        print(f"Writing logs to '{logpath}'")

    if opts.i:
        evs = await parse_file(opts.i, logf)
    else:
        env = dict(os.environ)
        if not opts.v:
            env["TT_LOGGER_TYPES"] = "Test"

        evs = []
        if not opts.retention_only:
            exit_status, evs = await run_gtest(MAIN_FILTER, env, logf)

        if not opts.no_retention:
            case_status, case_evs = await run_t6py_like_retention(env, opts.retention_sleep_s, logf)
            evs.extend(case_evs)
            exit_status = exit_status or case_status

    # pprint.pp(evs)
    bdfs = {}
    edc: list[dict] = []
    runs = list(parse_evs(evs, bdfs, edc))
    # pprint.pp(runs)
    # pprint.pp(bdfs)
    # print(runs_to_json(runs, edc, sort_keys=True, indent=4))

    if not opts.n:
        print_results(runs, bdfs, edc)

    if logf:
        print_results(runs, bdfs, edc, logf)

    runs_to_jsonf(logfile, runs, edc, sort_keys=True, indent=4)

    if logf:
        logf.close()

    exit(exit_status)


if __name__ == "__main__":
    asyncio.run(main())
