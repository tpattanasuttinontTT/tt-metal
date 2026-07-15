#!/usr/bin/env python3
"""Scan recent GitHub Actions jobs for runner-failure signatures."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib import error, request
from urllib.parse import quote, urlencode, urlparse

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - handled in load_config
    yaml = None


SLACK_API_URL = "https://slack.com/api/chat.postMessage"
MAX_SLACK_TEXT_LENGTH = 35000
STATE_SCHEMA_VERSION = 1
SIGNATURE_VERSION = "runner-failure-signatures-2026-07-14-v1"
NON_FAILED_CONCLUSIONS = {"success", "skipped", "cancelled"}
UNKNOWN_RUNNER = "(unknown runner)"

OSC_SEQUENCE_RE = re.compile(r"\x1b\].*?\x1b\\")
CSI_SEQUENCE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
FABRIC_LINK_MISMATCH_RE = re.compile(
    r"target\s+graph\s+edge\s+from\s+node\s+"
    r"\((?P<src_mesh>M\d+),\s*(?P<src_device>D\d+)\)\s+to\s+"
    r"\((?P<dst_mesh>M\d+),\s*(?P<dst_device>D\d+)\)\s+"
    r"requires\s+\d+\s+channels,\s+but\s+physical\s+edge\s+from\s+"
    r"\S+\s+to\s+\S+\s+only\s+has\s+\d+\s+channels",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ErrorSignature:
    key: str
    label: str
    needle: str | None = None
    pattern: str | None = None
    case_sensitive: bool = True


@dataclass(frozen=True)
class WorkflowJobFilter:
    include_exact: tuple[str, ...] = ()
    include_prefixes: tuple[str, ...] = ()
    exclude_exact: tuple[str, ...] = ()
    exclude_prefixes: tuple[str, ...] = ()


@dataclass(frozen=True)
class WorkflowTarget:
    owner_repo: str
    workflow_id: str
    name: str
    source: str
    job_filter: WorkflowJobFilter


@dataclass(frozen=True)
class RecentJob:
    owner_repo: str
    workflow: str
    workflow_id: str
    run_id: str
    run_attempt: str
    run_url: str
    job_id: str
    name: str
    runner_name: str
    status: str
    conclusion: str
    html_url: str
    started_at: str
    completed_at: str


@dataclass(frozen=True)
class LogLookupResult:
    log_text: str | None
    status: str


@dataclass(frozen=True)
class JobScanResult:
    job: RecentJob
    log_status: str
    log_checked: bool
    signature_labels: tuple[str, ...]
    fabric_missing_links: str


@dataclass(frozen=True)
class SlackConfig:
    token: str
    channel: str


ERROR_SIGNATURES = (
    ErrorSignature(
        key="TLB_ERROR_FOUND",
        label="TLB error",
        needle="Failed to allocate TLB window.",
    ),
    ErrorSignature(
        key="MISSING_DEVICES_FOUND",
        label="Missing devices",
        pattern=(
            r"(?:Requested\s+mesh\s+grid\s+shape\s+[^\n]*?"
            r"is\s+larger\s+than\s+number\s+of\s+available\s+devices|"
            r"Requested\s+mesh\s+shape\s+[^\n]*?"
            r"requires\s+\d+\s+devices,\s+but\s+only\s+\d+\s+devices\s+"
            r"(?:are\s+)?available|"
            r"Error\s+in\s+detecting\s+devices|"
            r"Query\s+mappings\s+failed\s+on\s+device\s+\d+)"
        ),
        case_sensitive=False,
    ),
    ErrorSignature(
        key="FABRIC_LINK_DOWN_MGD_TOPOLOGY_FOUND",
        label="Fabric link down (MGD topology)",
        needle="Graph specified in MGD could not fit in the discovered physical topology",
        case_sensitive=False,
    ),
    ErrorSignature(
        key="OUT_OF_DISK_FOUND",
        label="Out of disk",
        pattern=(
            r"(no\s+space\s+left\s+on\s+device|enospc|"
            r"disk\s+usage\s+is\s+(?:9\d|100)\s*%|"
            r"disk\s+usage\s+is\s+high)"
        ),
        case_sensitive=False,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan recent jobs from selected GitHub Actions workflows and report "
            "runner-failure signatures."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("out/runner_failure_scan_workflows.yaml"),
        help="YAML workflow selector config.",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Lookback window in hours (default: 24).",
    )
    parser.add_argument(
        "--state-in",
        type=Path,
        help="Optional checked-job state from a previous workflow artifact.",
    )
    parser.add_argument(
        "--state-out",
        type=Path,
        default=Path("out/runner_failure_state.json"),
        help="Path to write updated checked-job state.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("out/runner_failure_report.json"),
        help="Path to write the JSON report.",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=Path("out/runner_failure_report.md"),
        help="Path to write the Markdown report.",
    )
    parser.add_argument(
        "--post-slack-from-report",
        type=Path,
        help=(
            "Read a completed runner-failure report JSON, post Slack notifications, "
            "and exit without scanning GitHub."
        ),
    )
    parser.add_argument(
        "--force-fresh",
        action="store_true",
        help="Ignore --state-in and scan all matching failed jobs.",
    )
    parser.add_argument(
        "--gh-timeout",
        type=int,
        default=120,
        help="Timeout in seconds for each gh api request (default: 120).",
    )
    parser.add_argument(
        "--log-workers",
        type=int,
        default=int(os.environ.get("RUNNER_FAILURE_SCAN_LOG_WORKERS", "8")),
        help=(
            "Maximum number of GitHub job logs to scan in parallel "
            "(default: RUNNER_FAILURE_SCAN_LOG_WORKERS or 8)."
        ),
    )
    parser.add_argument(
        "--slack-channel",
        default=os.environ.get("RUNNER_FAILURE_SCAN_SLACK_CHANNEL"),
        help="Slack channel ID for runner-failure summaries.",
    )
    return parser.parse_args()


def ensure_gh_available() -> None:
    if shutil.which("gh") is None:
        raise RuntimeError("Missing dependency: GitHub CLI `gh` is not available.")


def format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_github_time(value: str) -> datetime:
    if not value:
        return datetime.fromtimestamp(0, timezone.utc)
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
            timezone.utc
        )
    except ValueError:
        return datetime.fromtimestamp(0, timezone.utc)


def string_tuple_from_config(
    entry: dict[str, Any], field_name: str, workflow_label: str
) -> tuple[str, ...]:
    value = entry.get(field_name)
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, list):
        raise ValueError(f"{workflow_label}: {field_name} must be a list of strings.")

    items: list[str] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, str):
            raise ValueError(
                f"{workflow_label}: {field_name}[{index}] must be a string."
            )
        items.append(item)
    return tuple(items)


def workflow_filter_from_config(
    entry: dict[str, Any], workflow_label: str
) -> WorkflowJobFilter:
    return WorkflowJobFilter(
        include_exact=string_tuple_from_config(entry, "include_exact", workflow_label),
        include_prefixes=string_tuple_from_config(
            entry, "include_prefixes", workflow_label
        ),
        exclude_exact=string_tuple_from_config(entry, "exclude_exact", workflow_label),
        exclude_prefixes=string_tuple_from_config(
            entry, "exclude_prefixes", workflow_label
        ),
    )


def parse_workflow_url(value: str) -> tuple[str, str]:
    parsed = urlparse(value)
    path_parts = [part for part in parsed.path.split("/") if part]
    if not parsed.scheme or parsed.netloc != "github.com":
        raise ValueError(f"Expected a GitHub workflow URL, got: {value}")
    try:
        actions_index = path_parts.index("actions")
    except ValueError as exc:
        raise ValueError(f"Workflow URL is missing /actions/workflows/: {value}") from exc
    if (
        actions_index < 2
        or actions_index + 2 >= len(path_parts)
        or path_parts[actions_index + 1] != "workflows"
    ):
        raise ValueError(f"Workflow URL is missing /actions/workflows/: {value}")

    owner_repo = f"{path_parts[actions_index - 2]}/{path_parts[actions_index - 1]}"
    workflow_id = "/".join(path_parts[actions_index + 2 :])
    if not workflow_id:
        raise ValueError(f"Workflow URL is missing workflow file name: {value}")
    return owner_repo, workflow_id


def workflow_from_config_entry(
    entry: dict[str, Any], default_owner_repo: str | None, index: int
) -> WorkflowTarget | None:
    workflow_label = f"workflow #{index}"
    enabled = entry.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError(f"{workflow_label}: enabled must be true or false.")
    if not enabled:
        return None

    url = entry.get("url")
    workflow_id = entry.get("workflow_id")
    owner_repo = entry.get("repository") or default_owner_repo
    if url:
        if not isinstance(url, str):
            raise ValueError(f"{workflow_label}: url must be a string.")
        owner_repo, workflow_id = parse_workflow_url(url)
    elif not isinstance(workflow_id, str) or not workflow_id:
        raise ValueError(f"{workflow_label}: missing workflow_id or url.")
    elif not isinstance(owner_repo, str) or not owner_repo:
        raise ValueError(f"{workflow_label}: missing repository.")

    name = entry.get("name") or str(workflow_id).rsplit("/", 1)[-1]
    if not isinstance(name, str):
        raise ValueError(f"{workflow_label}: name must be a string.")

    return WorkflowTarget(
        owner_repo=owner_repo,
        workflow_id=str(workflow_id),
        name=name,
        source=str(url or workflow_id),
        job_filter=workflow_filter_from_config(entry, name),
    )


def load_workflows(config_path: Path) -> list[WorkflowTarget]:
    if yaml is None:
        raise RuntimeError("Missing dependency: install PyYAML to read workflow config.")
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Unable to read workflow config {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Unable to parse workflow config {config_path}: {exc}") from exc

    if not isinstance(config, dict):
        raise ValueError(f"Workflow config {config_path} must contain a mapping.")

    default_owner_repo = config.get("repository")
    if default_owner_repo is not None and not isinstance(default_owner_repo, str):
        raise ValueError("repository must be a string when set.")

    workflow_entries = config.get("workflows")
    if not isinstance(workflow_entries, list):
        raise ValueError(f"Workflow config {config_path} must contain workflows list.")

    workflows: list[WorkflowTarget] = []
    for index, entry in enumerate(workflow_entries, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"workflow #{index} must be a mapping.")
        workflow = workflow_from_config_entry(entry, default_owner_repo, index)
        if workflow is None:
            continue
        workflows.append(workflow)

    if not workflows:
        raise ValueError(f"Workflow config {config_path} has no enabled workflows.")
    return workflows


def matches_any_prefix(value: str, prefixes: tuple[str, ...]) -> bool:
    folded_value = value.casefold()
    return any(folded_value.startswith(prefix.casefold()) for prefix in prefixes)


def matches_any_exact(value: str, exact_values: tuple[str, ...]) -> bool:
    folded_value = value.casefold()
    return any(folded_value == exact_value.casefold() for exact_value in exact_values)


def job_allowed_by_filter(job_name: str, job_filter: WorkflowJobFilter) -> bool:
    has_includes = bool(job_filter.include_exact or job_filter.include_prefixes)
    if has_includes and not (
        matches_any_exact(job_name, job_filter.include_exact)
        or matches_any_prefix(job_name, job_filter.include_prefixes)
    ):
        return False
    if matches_any_exact(job_name, job_filter.exclude_exact):
        return False
    if matches_any_prefix(job_name, job_filter.exclude_prefixes):
        return False
    return True


def gh_env() -> dict[str, str]:
    env = os.environ.copy()
    if env.get("GITHUB_TOKEN") and not env.get("GH_TOKEN"):
        env["GH_TOKEN"] = env["GITHUB_TOKEN"]
    return env


def gh_api_json(endpoint: str, *, paginate: bool = False, timeout: int = 120) -> Any:
    cmd = ["gh", "api", "--method", "GET"]
    if paginate:
        cmd.extend(["--paginate", "--slurp"])
    cmd.append(endpoint)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
        env=gh_env(),
    )
    if result.returncode != 0:
        details = " ".join((result.stderr or result.stdout or "unknown gh api error").split())
        raise RuntimeError(f"gh api failed for {endpoint}: {details}")
    return json.loads(result.stdout or "{}")


def paginated_items(response: Any, key: str) -> list[dict[str, Any]]:
    if isinstance(response, dict):
        return list(response.get(key, []))

    items: list[dict[str, Any]] = []
    if isinstance(response, list):
        for page in response:
            if isinstance(page, dict):
                items.extend(page.get(key, []))
    return items


def workflow_runs_endpoint(workflow: WorkflowTarget, since: datetime) -> str:
    workflow_id = quote(workflow.workflow_id, safe="")
    query = urlencode(
        {
            "per_page": "100",
            "exclude_pull_requests": "true",
            "created": f">={format_utc(since)}",
        }
    )
    return f"repos/{workflow.owner_repo}/actions/workflows/{workflow_id}/runs?{query}"


def repository_runs_endpoint(owner_repo: str, since: datetime) -> str:
    query = urlencode(
        {
            "per_page": "100",
            "created": f">={format_utc(since)}",
        }
    )
    return f"repos/{owner_repo}/actions/runs?{query}"


def workflow_run_jobs_endpoint(owner_repo: str, run_id: str) -> str:
    query = urlencode({"filter": "all", "per_page": "100"})
    return f"repos/{owner_repo}/actions/runs/{run_id}/jobs?{query}"


def recent_job_from_api(
    *,
    owner_repo: str,
    workflow_name: str,
    workflow_id: str,
    run: dict[str, Any],
    job: dict[str, Any],
) -> RecentJob:
    return RecentJob(
        owner_repo=owner_repo,
        workflow=workflow_name,
        workflow_id=workflow_id,
        run_id=str(run.get("id") or ""),
        run_attempt=str(run.get("run_attempt") or ""),
        run_url=str(run.get("html_url") or ""),
        job_id=str(job.get("id") or ""),
        name=str(job.get("name") or ""),
        runner_name=str(job.get("runner_name") or ""),
        status=str(job.get("status") or ""),
        conclusion=str(job.get("conclusion") or ""),
        html_url=str(job.get("html_url") or ""),
        started_at=str(job.get("started_at") or ""),
        completed_at=str(job.get("completed_at") or ""),
    )


def list_recent_jobs(
    workflows: list[WorkflowTarget], since: datetime, gh_timeout: int
) -> list[RecentJob]:
    recent_jobs: list[RecentJob] = []
    seen_jobs: set[tuple[str, str]] = set()

    for workflow in workflows:
        kept_count = 0
        filtered_count = 0
        runs_response = gh_api_json(
            workflow_runs_endpoint(workflow, since),
            paginate=True,
            timeout=gh_timeout,
        )
        runs = paginated_items(runs_response, "workflow_runs")
        print(f"Found {len(runs)} recent run(s) for {workflow.name} ({workflow.source}).")

        for run in runs:
            run_id = str(run.get("id") or "")
            if not run_id:
                continue

            jobs_response = gh_api_json(
                workflow_run_jobs_endpoint(workflow.owner_repo, run_id),
                paginate=True,
                timeout=gh_timeout,
            )
            for job in paginated_items(jobs_response, "jobs"):
                job_id = str(job.get("id") or "")
                if not job_id:
                    continue

                job_name = str(job.get("name") or "")
                if not job_allowed_by_filter(job_name, workflow.job_filter):
                    filtered_count += 1
                    continue

                job_key = (workflow.owner_repo, job_id)
                if job_key in seen_jobs:
                    continue
                seen_jobs.add(job_key)
                kept_count += 1

                recent_jobs.append(
                    recent_job_from_api(
                        owner_repo=workflow.owner_repo,
                        workflow_name=workflow.name,
                        workflow_id=workflow.workflow_id,
                        run=run,
                        job=job,
                    )
                )

        print(
            f"Kept {kept_count} job(s) for {workflow.name}; "
            f"filtered out {filtered_count} job(s)."
        )

    return sorted(
        recent_jobs,
        key=lambda job: parse_github_time(job.started_at),
        reverse=True,
    )


def list_recent_jobs_for_runners(
    *,
    owner_repos: list[str],
    runner_names: set[str],
    since: datetime,
    gh_timeout: int,
) -> list[RecentJob]:
    if not runner_names:
        return []

    recent_jobs: list[RecentJob] = []
    seen_jobs: set[tuple[str, str]] = set()
    folded_runner_names = {runner.casefold() for runner in runner_names}

    for owner_repo in owner_repos:
        runs_response = gh_api_json(
            repository_runs_endpoint(owner_repo, since),
            paginate=True,
            timeout=gh_timeout,
        )
        runs = paginated_items(runs_response, "workflow_runs")
        kept_count = 0
        job_count = 0
        print(
            f"Found {len(runs)} recent repo run(s) in {owner_repo} "
            "for affected runner follow-up."
        )

        for run in runs:
            run_id = str(run.get("id") or "")
            if not run_id:
                continue

            jobs_response = gh_api_json(
                workflow_run_jobs_endpoint(owner_repo, run_id),
                paginate=True,
                timeout=gh_timeout,
            )
            workflow_name = str(
                run.get("name") or run.get("display_title") or "unknown workflow"
            )
            workflow_id = str(run.get("path") or run.get("workflow_id") or "")

            for job in paginated_items(jobs_response, "jobs"):
                job_id = str(job.get("id") or "")
                if not job_id:
                    continue
                job_count += 1

                runner_name = str(job.get("runner_name") or "")
                if runner_name.casefold() not in folded_runner_names:
                    continue

                job_key = (owner_repo, job_id)
                if job_key in seen_jobs:
                    continue
                seen_jobs.add(job_key)
                kept_count += 1
                recent_jobs.append(
                    recent_job_from_api(
                        owner_repo=owner_repo,
                        workflow_name=workflow_name,
                        workflow_id=workflow_id,
                        run=run,
                        job=job,
                    )
                )

        print(
            f"Kept {kept_count} affected-runner job(s) from {job_count} "
            f"recent job(s) in {owner_repo}."
        )

    return sorted(
        recent_jobs,
        key=lambda job: parse_github_time(job.started_at),
        reverse=True,
    )


def is_failed_job(job: RecentJob) -> bool:
    conclusion = job.conclusion.lower()
    return bool(conclusion) and conclusion not in NON_FAILED_CONCLUSIONS


def job_state_key(job: RecentJob) -> str:
    return f"{job.owner_repo}:{job.job_id}"


def signature_keys() -> list[str]:
    return [signature.key for signature in ERROR_SIGNATURES]


def empty_state() -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "signature_version": SIGNATURE_VERSION,
        "signature_keys": signature_keys(),
        "checked_jobs": {},
    }


def normalize_state(state: Any) -> dict[str, Any]:
    if not isinstance(state, dict):
        return empty_state()
    if state.get("schema_version") != STATE_SCHEMA_VERSION:
        return empty_state()
    if state.get("signature_version") != SIGNATURE_VERSION:
        print("Signature version changed; ignoring previous checked-job state.")
        return empty_state()
    if state.get("signature_keys") != signature_keys():
        print("Signature set changed; ignoring previous checked-job state.")
        return empty_state()
    if not isinstance(state.get("checked_jobs"), dict):
        state["checked_jobs"] = {}
    return state


def load_state(path: Path | None, force_fresh: bool) -> dict[str, Any]:
    if force_fresh or path is None or not path.is_file():
        return empty_state()
    try:
        return normalize_state(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        return empty_state()


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    temp_path.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)


def unchecked_jobs(jobs: list[RecentJob], state: dict[str, Any]) -> list[RecentJob]:
    checked_jobs = state.get("checked_jobs", {})
    return [job for job in jobs if job_state_key(job) not in checked_jobs]


def strip_terminal_sequences(value: str) -> str:
    return CSI_SEQUENCE_RE.sub("", OSC_SEQUENCE_RE.sub("", value))


def visible_len(value: str) -> int:
    return len(strip_terminal_sequences(value))


def clean_cell(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def pad_cell(value: str, width: int) -> str:
    return value + (" " * max(width - visible_len(value), 0))


def truncate(value: str, max_width: int) -> str:
    if visible_len(value) <= max_width:
        return value

    plain_value = strip_terminal_sequences(value)
    if max_width <= 3:
        return plain_value[:max_width]
    return f"{plain_value[: max_width - 3]}..."


def render_table(
    rows: list[dict[str, Any]], columns: list[tuple[str, str, int]]
) -> str:
    if not rows:
        return "(no rows)"

    rendered_rows = [
        [truncate(clean_cell(row.get(key)), max_width) for key, _, max_width in columns]
        for row in rows
    ]
    header_lines_by_column = [label.splitlines() or [""] for _, label, _ in columns]
    header_height = max(len(lines) for lines in header_lines_by_column)
    widths = [
        max(
            *(visible_len(line) for line in header_lines_by_column[index]),
            *(visible_len(row[index]) for row in rendered_rows),
        )
        for index in range(len(columns))
    ]

    header_rows = []
    for line_index in range(header_height):
        header_rows.append(
            " | ".join(
                pad_cell(
                    header_lines_by_column[column_index][line_index]
                    if line_index < len(header_lines_by_column[column_index])
                    else "",
                    widths[column_index],
                )
                for column_index in range(len(columns))
            )
        )
    separator_line = "-+-".join("-" * width for width in widths)
    body_lines = [
        " | ".join(pad_cell(row[index], widths[index]) for index in range(len(row)))
        for row in rendered_rows
    ]

    return "\n".join([*header_rows, separator_line, *body_lines])


def signature_found(log_text: str, signature: ErrorSignature) -> bool:
    if signature.pattern:
        flags = 0 if signature.case_sensitive else re.IGNORECASE
        return re.search(signature.pattern, log_text, flags=flags) is not None

    if not signature.needle:
        return False

    if signature.case_sensitive:
        return signature.needle in log_text
    return signature.needle.lower() in log_text.lower()


def format_fabric_node(mesh: str, device: str) -> str:
    return f"{mesh.lower()},{device.lower()}"


def extract_fabric_missing_links(log_text: str) -> str:
    links: list[str] = []
    seen_links: set[str] = set()
    plain_log_text = strip_terminal_sequences(log_text)
    for match in FABRIC_LINK_MISMATCH_RE.finditer(plain_log_text):
        source = format_fabric_node(match.group("src_mesh"), match.group("src_device"))
        destination = format_fabric_node(
            match.group("dst_mesh"), match.group("dst_device")
        )
        link = f"{source}>{destination}"
        if link not in seen_links:
            seen_links.add(link)
            links.append(link)
    return "; ".join(links)


def matching_signature_labels(log_text: str) -> list[str]:
    return [
        signature.label
        for signature in ERROR_SIGNATURES
        if signature_found(log_text, signature)
    ]


def fetch_github_job_log(job: RecentJob, timeout: int) -> LogLookupResult:
    endpoint = f"repos/{job.owner_repo}/actions/jobs/{job.job_id}/logs"
    try:
        result = subprocess.run(
            ["gh", "api", endpoint],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            env=gh_env(),
        )
    except subprocess.TimeoutExpired:
        return LogLookupResult(log_text=None, status=f"gh api timed out after {timeout}s")

    if result.returncode != 0:
        details = " ".join((result.stderr or result.stdout or "unknown gh api error").split())
        return LogLookupResult(log_text=None, status=f"gh api failed: {details}")

    return LogLookupResult(log_text=result.stdout, status="fetched")


def scan_job(job: RecentJob, timeout: int) -> JobScanResult:
    log_result = fetch_github_job_log(job, timeout=timeout)
    if log_result.log_text is None:
        return JobScanResult(
            job=job,
            log_status=log_result.status,
            log_checked=False,
            signature_labels=(),
            fabric_missing_links="",
        )

    signature_labels = matching_signature_labels(log_result.log_text)
    fabric_missing_links = ""
    if "Fabric link down (MGD topology)" in signature_labels:
        fabric_missing_links = extract_fabric_missing_links(log_result.log_text)

    return JobScanResult(
        job=job,
        log_status=log_result.status,
        log_checked=True,
        signature_labels=tuple(signature_labels),
        fabric_missing_links=fabric_missing_links,
    )


def scan_jobs(
    jobs: list[RecentJob], *, gh_timeout: int, log_workers: int
) -> list[JobScanResult]:
    if not jobs:
        return []

    worker_count = min(log_workers, len(jobs))
    print(f"Scanning {len(jobs)} job log(s) with {worker_count} worker(s).")
    results: list[JobScanResult] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(scan_job, job, gh_timeout) for job in jobs]
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:
                print(f"warning: job scan failed: {exc}", file=sys.stderr)
                continue
            results.append(result)
            if not result.log_checked:
                print(
                    f"warning: could not check {result.job.html_url}: "
                    f"{result.log_status}",
                    file=sys.stderr,
                )
            elif result.signature_labels:
                print(f"runner failure {result.job.html_url}")
    return sorted(
        results,
        key=lambda result: parse_github_time(result.job.started_at),
        reverse=True,
    )


def should_scan_log_for_full_table(job: RecentJob) -> bool:
    conclusion = job.conclusion.lower()
    return bool(job.job_id) and conclusion not in {"skipped", "cancelled"}


def full_runner_log_table_results(
    *,
    owner_repos: list[str],
    since: datetime,
    initial_scan_results: list[JobScanResult],
    failures: list[JobScanResult],
    gh_timeout: int,
    log_workers: int,
) -> list[JobScanResult]:
    affected_runners = {
        runner_name_for_job(result.job)
        for result in failures
        if runner_name_for_job(result.job) != UNKNOWN_RUNNER
    }
    if not affected_runners:
        return []

    table_jobs = list_recent_jobs_for_runners(
        owner_repos=owner_repos,
        runner_names=affected_runners,
        since=since,
        gh_timeout=gh_timeout,
    )
    table_job_keys = {job_state_key(job) for job in table_jobs}
    for result in initial_scan_results:
        if (
            runner_name_for_job(result.job) in affected_runners
            and job_state_key(result.job) not in table_job_keys
        ):
            table_jobs.append(result.job)
            table_job_keys.add(job_state_key(result.job))

    results_by_key = {
        job_state_key(result.job): result
        for result in initial_scan_results
        if runner_name_for_job(result.job) in affected_runners
    }
    jobs_to_scan = [
        job
        for job in table_jobs
        if job_state_key(job) not in results_by_key and should_scan_log_for_full_table(job)
    ]

    if jobs_to_scan:
        print(
            "Scanning "
            f"{len(jobs_to_scan)} additional job log(s) for full runner tables."
        )
        for result in scan_jobs(
            jobs_to_scan,
            gh_timeout=gh_timeout,
            log_workers=log_workers,
        ):
            results_by_key[job_state_key(result.job)] = result

    for job in table_jobs:
        key = job_state_key(job)
        if key not in results_by_key:
            conclusion = job.conclusion or job.status or "unknown"
            results_by_key[key] = JobScanResult(
                job=job,
                log_status=f"not checked: {conclusion}",
                log_checked=False,
                signature_labels=(),
                fabric_missing_links="",
            )

    return [
        results_by_key[job_state_key(job)]
        for job in sorted(
            table_jobs,
            key=lambda table_job: parse_github_time(table_job.started_at),
            reverse=True,
        )
    ]


def mark_job_checked(state: dict[str, Any], result: JobScanResult) -> None:
    state.setdefault("checked_jobs", {})[job_state_key(result.job)] = {
        "checked_at": format_utc(datetime.now(timezone.utc)),
        "workflow": result.job.workflow,
        "run_id": result.job.run_id,
        "run_attempt": result.job.run_attempt,
        "job_id": result.job.job_id,
        "job_name": result.job.name,
        "runner_name": result.job.runner_name,
        "conclusion": result.job.conclusion,
        "html_url": result.job.html_url,
        "log_checked": result.log_checked,
        "log_status": result.log_status,
        "signatures": list(result.signature_labels),
        "fabric_missing_links": result.fabric_missing_links,
    }


def job_to_dict(job: RecentJob) -> dict[str, Any]:
    return {
        "owner_repo": job.owner_repo,
        "workflow": job.workflow,
        "workflow_id": job.workflow_id,
        "run_id": job.run_id,
        "run_attempt": job.run_attempt,
        "run_url": job.run_url,
        "job_id": job.job_id,
        "name": job.name,
        "runner_name": job.runner_name,
        "status": job.status,
        "conclusion": job.conclusion,
        "html_url": job.html_url,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
    }


def result_to_dict(result: JobScanResult) -> dict[str, Any]:
    value = job_to_dict(result.job)
    value.update(
        {
            "log_checked": result.log_checked,
            "log_status": result.log_status,
            "signatures": list(result.signature_labels),
            "fabric_missing_links": result.fabric_missing_links,
        }
    )
    return value


def signature_counts(results: list[JobScanResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        for label in result.signature_labels:
            counts[label] = counts.get(label, 0) + 1
    return counts


def format_signature_summary(results: list[JobScanResult]) -> str:
    counts = signature_counts(results)
    return ", ".join(f"{count}x {label}" for label, count in sorted(counts.items()))


def runner_name_for_job(job: RecentJob) -> str:
    return job.runner_name or UNKNOWN_RUNNER


def group_results_by_runner(
    results: list[JobScanResult],
) -> dict[str, list[JobScanResult]]:
    grouped: dict[str, list[JobScanResult]] = {}
    for result in results:
        grouped.setdefault(runner_name_for_job(result.job), []).append(result)
    return grouped


def markdown_escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def markdown_link(label: str, url: str) -> str:
    if not url:
        return markdown_escape(label)
    return f"[{markdown_escape(label)}]({url})"


def build_markdown_report(
    *,
    generated_at: datetime,
    since: datetime,
    hours: int,
    workflows: list[WorkflowTarget],
    recent_jobs: list[RecentJob],
    failed_jobs: list[RecentJob],
    jobs_to_scan: list[RecentJob],
    scan_results: list[JobScanResult],
) -> str:
    failures = [result for result in scan_results if result.signature_labels]
    lines = [
        "# Runner Failure Scan",
        "",
        f"- Generated at: `{format_utc(generated_at)}`",
        f"- Window: last `{hours}` hour(s), since `{format_utc(since)}`",
        f"- Workflows: `{', '.join(workflow.name for workflow in workflows)}`",
        f"- Recent selected jobs: `{len(recent_jobs)}`",
        f"- Failed jobs: `{len(failed_jobs)}`",
        f"- Scanned jobs: `{len(jobs_to_scan)}`",
        f"- Runner-failure jobs: `{len(failures)}`",
        "",
    ]

    if not failures:
        lines.append("No runner-failure signatures found in scanned jobs.")
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "## Runner Failures",
            "",
            "| Runner | Count | Signatures | Jobs |",
            "| --- | ---: | --- | --- |",
        ]
    )
    for runner, runner_results in sorted(group_results_by_runner(failures).items()):
        jobs_text = "<br>".join(
            markdown_link(f"{result.job.workflow} #{result.job.job_id}", result.job.html_url)
            for result in runner_results[:10]
        )
        if len(runner_results) > 10:
            jobs_text += f"<br>... {len(runner_results) - 10} more"
        lines.append(
            "| "
            f"{markdown_escape(runner)} | "
            f"{len(runner_results)} | "
            f"{markdown_escape(format_signature_summary(runner_results))} | "
            f"{jobs_text} |"
        )

    lines.extend(["", "## Triggering Jobs", ""])
    for result in failures:
        signature_text = ", ".join(result.signature_labels)
        missing_links_text = (
            f"; missing links: {result.fabric_missing_links}"
            if result.fabric_missing_links
            else ""
        )
        lines.append(
            "- "
            f"{markdown_link(result.job.html_url or result.job.job_id, result.job.html_url)} "
            f"`{runner_name_for_job(result.job)}` "
            f"`{result.job.workflow}` "
            f"`{result.job.name}` "
            f"({signature_text}{missing_links_text})"
        )
    return "\n".join(lines) + "\n"


def build_json_report(
    *,
    generated_at: datetime,
    since: datetime,
    hours: int,
    workflows: list[WorkflowTarget],
    recent_jobs: list[RecentJob],
    failed_jobs: list[RecentJob],
    jobs_to_scan: list[RecentJob],
    scan_results: list[JobScanResult],
    runner_log_table_results: list[JobScanResult],
) -> dict[str, Any]:
    failures = [result for result in scan_results if result.signature_labels]
    return {
        "generated_at": format_utc(generated_at),
        "since": format_utc(since),
        "hours": hours,
        "signature_version": SIGNATURE_VERSION,
        "workflows": [
            {
                "name": workflow.name,
                "owner_repo": workflow.owner_repo,
                "workflow_id": workflow.workflow_id,
                "source": workflow.source,
            }
            for workflow in workflows
        ],
        "counts": {
            "recent_jobs": len(recent_jobs),
            "failed_jobs": len(failed_jobs),
            "jobs_to_scan": len(jobs_to_scan),
            "scanned_jobs": len(scan_results),
            "runner_log_table_jobs": len(runner_log_table_results),
            "runner_failure_jobs": len(failures),
        },
        "signature_counts": signature_counts(failures),
        "recent_jobs": [job_to_dict(job) for job in recent_jobs],
        "failed_jobs": [job_to_dict(job) for job in failed_jobs],
        "jobs_to_scan": [job_to_dict(job) for job in jobs_to_scan],
        "runner_failures": {
            runner: [result_to_dict(result) for result in results]
            for runner, results in sorted(group_results_by_runner(failures).items())
        },
        "scan_results": [result_to_dict(result) for result in scan_results],
        "runner_log_table_results": [
            result_to_dict(result) for result in runner_log_table_results
        ],
    }


def write_reports(
    *,
    report_json_path: Path,
    report_md_path: Path,
    report_json: dict[str, Any],
    report_md: str,
) -> None:
    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    report_md_path.parent.mkdir(parents=True, exist_ok=True)
    report_json_path.write_text(
        json.dumps(report_json, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_md_path.write_text(report_md, encoding="utf-8")


def slack_config_from_args(args: argparse.Namespace) -> SlackConfig | None:
    if not args.slack_channel:
        return None
    token = os.environ.get("SLACK_BOT_TOKEN") or os.environ.get(
        "RUNNER_FAILURE_SCAN_SLACK_BOT_TOKEN"
    )
    if not token:
        print(
            "warning: Slack channel is configured, but SLACK_BOT_TOKEN is not set; "
            "skipping Slack notifications.",
            file=sys.stderr,
        )
        return None
    return SlackConfig(token=token, channel=args.slack_channel)


def job_from_dict(value: dict[str, Any]) -> RecentJob:
    return RecentJob(
        owner_repo=str(value.get("owner_repo") or ""),
        workflow=str(value.get("workflow") or ""),
        workflow_id=str(value.get("workflow_id") or ""),
        run_id=str(value.get("run_id") or ""),
        run_attempt=str(value.get("run_attempt") or ""),
        run_url=str(value.get("run_url") or ""),
        job_id=str(value.get("job_id") or ""),
        name=str(value.get("name") or ""),
        runner_name=str(value.get("runner_name") or ""),
        status=str(value.get("status") or ""),
        conclusion=str(value.get("conclusion") or ""),
        html_url=str(value.get("html_url") or ""),
        started_at=str(value.get("started_at") or ""),
        completed_at=str(value.get("completed_at") or ""),
    )


def scan_result_from_dict(value: dict[str, Any]) -> JobScanResult:
    raw_signatures = value.get("signatures")
    signatures: tuple[str, ...] = ()
    if isinstance(raw_signatures, list):
        signatures = tuple(str(item) for item in raw_signatures if item)
    return JobScanResult(
        job=job_from_dict(value),
        log_status=str(value.get("log_status") or ""),
        log_checked=bool(value.get("log_checked")),
        signature_labels=signatures,
        fabric_missing_links=str(value.get("fabric_missing_links") or ""),
    )


def load_report_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(f"Unable to read report JSON {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Unable to parse report JSON {path}: {exc}") from exc

    if not isinstance(value, dict):
        raise RuntimeError(f"Report JSON {path} must contain an object.")
    return value


def scan_results_from_report(report: dict[str, Any]) -> list[JobScanResult]:
    values = report.get("scan_results")
    if not isinstance(values, list):
        return []
    return [
        scan_result_from_dict(value)
        for value in values
        if isinstance(value, dict)
    ]


def int_from_report(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def post_slack_alerts_from_report(
    slack_config: SlackConfig, report: dict[str, Any]
) -> None:
    scan_results = scan_results_from_report(report)
    failures = [result for result in scan_results if result.signature_labels]
    raw_runner_log_table_results = report.get("runner_log_table_results")
    runner_log_table_results = []
    if isinstance(raw_runner_log_table_results, list):
        runner_log_table_results = [
            scan_result_from_dict(value)
            for value in raw_runner_log_table_results
            if isinstance(value, dict)
        ]
    post_slack_alerts(
        slack_config,
        hours=int_from_report(report.get("hours"), default=24),
        failures=failures,
        scan_results=scan_results,
        runner_log_table_results=runner_log_table_results,
    )


def slack_escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def slack_link(url: str, label: str) -> str:
    if not url:
        return slack_escape(label)
    return f"<{url}|{slack_escape(label)}>"


def truncate_for_slack(value: str, max_length: int = MAX_SLACK_TEXT_LENGTH) -> str:
    if len(value) <= max_length:
        return value
    suffix = "\n... truncated ..."
    return value[: max_length - len(suffix)] + suffix


def slack_code_block(value: str) -> str:
    cleaned_value = value.replace("```", "'''")
    return f"```{truncate_for_slack(cleaned_value)}```"


def post_slack_message(
    slack_config: SlackConfig, text: str, *, thread_ts: str | None = None
) -> str:
    payload: dict[str, Any] = {
        "channel": slack_config.channel,
        "text": truncate_for_slack(text),
        "unfurl_links": False,
        "unfurl_media": False,
    }
    if thread_ts:
        payload["thread_ts"] = thread_ts

    slack_request = request.Request(
        SLACK_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {slack_config.token}",
            "Content-Type": "application/json; charset=utf-8",
        },
        method="POST",
    )
    try:
        with request.urlopen(slack_request, timeout=30) as response:
            response_body = response.read().decode("utf-8", errors="replace")
    except error.URLError as exc:
        raise RuntimeError(f"Slack post failed: {exc}") from exc

    try:
        response_json = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Slack returned non-JSON response: {response_body}") from exc

    if not response_json.get("ok"):
        slack_error = response_json.get("error") or "unknown_error"
        raise RuntimeError(f"Slack post failed: {slack_error}")

    ts = response_json.get("ts")
    if not isinstance(ts, str) or not ts:
        raise RuntimeError("Slack post succeeded but response did not include ts.")
    return ts


def format_triggering_jobs_for_slack(
    runner: str, results: list[JobScanResult]
) -> str:
    lines = [f"Triggering job failures for `{slack_escape(runner)}`:"]
    for result in results:
        signature_text = ", ".join(result.signature_labels)
        if result.fabric_missing_links:
            signature_text += f"; missing links: {result.fabric_missing_links}"
        lines.append(
            "* "
            f"{slack_link(result.job.html_url, result.job.job_id)} "
            f"{slack_escape(result.job.workflow)} / "
            f"{slack_escape(result.job.name)} "
            f"({slack_escape(signature_text)})"
        )
    return truncate_for_slack("\n".join(lines))


def signature_cell(result: JobScanResult, label: str) -> str:
    if not result.log_checked:
        return ""
    return "YES" if label in result.signature_labels else "NO"


def runner_log_table_rows(results: list[JobScanResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        job = result.job
        rows.append(
            {
                "GITHUB_RUN_ID": job.run_id,
                "GITHUB_JOB_ID": job.job_id,
                "GITHUB_JOB_LINK_DISPLAY": "open" if job.html_url else "",
                "WORKFLOW": job.workflow,
                "JOB_NAME": job.name,
                "JOB_STATUS": job.conclusion or job.status,
                "JOB_START_TS": job.started_at,
                "LOG_CHECKED": "YES" if result.log_checked else "NO",
                "TLB_ERROR_FOUND": signature_cell(result, "TLB error"),
                "MISSING_DEVICES_FOUND": signature_cell(result, "Missing devices"),
                "FABRIC_LINK_DOWN_MGD_TOPOLOGY_FOUND": signature_cell(
                    result, "Fabric link down (MGD topology)"
                ),
                "FABRIC_MISSING_LINKS": result.fabric_missing_links,
                "OUT_OF_DISK_FOUND": signature_cell(result, "Out of disk"),
            }
        )
    return rows


def format_runner_log_table_for_slack(
    runner: str, results: list[JobScanResult]
) -> str:
    table = render_table(
        runner_log_table_rows(results),
        [
            ("GITHUB_RUN_ID", "GH run", 13),
            ("GITHUB_JOB_ID", "GH job", 13),
            ("GITHUB_JOB_LINK_DISPLAY", "GH link", 7),
            ("WORKFLOW", "Workflow", 24),
            ("JOB_NAME", "Job name", 48),
            ("JOB_STATUS", "Status", 16),
            ("JOB_START_TS", "Started", 20),
            ("LOG_CHECKED", "Log checked", 11),
            ("TLB_ERROR_FOUND", "TLB error", 9),
            ("MISSING_DEVICES_FOUND", "Missing devices", 15),
            (
                "FABRIC_LINK_DOWN_MGD_TOPOLOGY_FOUND",
                "Fabric link\ndown (MGD\ntopology)",
                11,
            ),
            ("FABRIC_MISSING_LINKS", "Missing links", 44),
            ("OUT_OF_DISK_FOUND", "Out of disk", 11),
        ],
    )
    return (
        f"Full log signature table for `{slack_escape(runner)}` "
        "(`check-logs all` style, all recent repo jobs):\n"
        f"{slack_code_block(table)}"
    )


def post_slack_alerts(
    slack_config: SlackConfig,
    *,
    hours: int,
    failures: list[JobScanResult],
    scan_results: list[JobScanResult],
    runner_log_table_results: list[JobScanResult] | None = None,
) -> None:
    if not failures:
        print("No runner-failure signatures found; skipping Slack notifications.")
        return

    table_results = runner_log_table_results or scan_results
    table_results_by_runner = group_results_by_runner(table_results)
    for runner, runner_results in sorted(group_results_by_runner(failures).items()):
        signature_summary = format_signature_summary(runner_results)
        if not signature_summary:
            continue
        parent_text = (
            f"Detected runner failure on `{slack_escape(runner)}`: "
            f"in the last {hours}h {slack_escape(signature_summary)}. "
            "Details in thread \U0001F9F5"
        )
        thread_ts = post_slack_message(slack_config, parent_text)
        post_slack_message(
            slack_config,
            format_triggering_jobs_for_slack(runner, runner_results),
            thread_ts=thread_ts,
        )
        post_slack_message(
            slack_config,
            format_runner_log_table_for_slack(
                runner, table_results_by_runner.get(runner, runner_results)
            ),
            thread_ts=thread_ts,
        )
        print(f"Posted Slack summary for {runner}.")


def validate_args(args: argparse.Namespace) -> None:
    if args.hours <= 0:
        raise ValueError("--hours must be greater than zero.")
    if args.gh_timeout <= 0:
        raise ValueError("--gh-timeout must be greater than zero.")
    if args.log_workers <= 0:
        raise ValueError("--log-workers must be greater than zero.")


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        if args.post_slack_from_report:
            slack_config = slack_config_from_args(args)
            if slack_config is None:
                return 0
            report = load_report_json(args.post_slack_from_report)
            post_slack_alerts_from_report(slack_config, report)
            return 0

        ensure_gh_available()
        workflows = load_workflows(args.config)
        generated_at = datetime.now(timezone.utc)
        since = generated_at - timedelta(hours=args.hours)
        state = load_state(args.state_in, force_fresh=args.force_fresh)
        recent_jobs = list_recent_jobs(workflows, since, gh_timeout=args.gh_timeout)
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    failed = [job for job in recent_jobs if is_failed_job(job)]
    jobs_to_scan = unchecked_jobs(failed, state)
    print(
        f"Recent selected jobs: {len(recent_jobs)} total, {len(failed)} failed, "
        f"{len(jobs_to_scan)} to scan."
    )
    print(f"Workflow config: {args.config}")
    if args.state_in:
        print(f"State input: {args.state_in} ({'ignored' if args.force_fresh else 'used if present'})")
    print(f"State output: {args.state_out}")

    scan_results = scan_jobs(
        jobs_to_scan,
        gh_timeout=args.gh_timeout,
        log_workers=args.log_workers,
    )
    for result in scan_results:
        mark_job_checked(state, result)

    save_state(args.state_out, state)
    failures = [result for result in scan_results if result.signature_labels]
    runner_log_table_results = full_runner_log_table_results(
        owner_repos=sorted({workflow.owner_repo for workflow in workflows}),
        since=since,
        initial_scan_results=scan_results,
        failures=failures,
        gh_timeout=args.gh_timeout,
        log_workers=args.log_workers,
    )

    report_json = build_json_report(
        generated_at=generated_at,
        since=since,
        hours=args.hours,
        workflows=workflows,
        recent_jobs=recent_jobs,
        failed_jobs=failed,
        jobs_to_scan=jobs_to_scan,
        scan_results=scan_results,
        runner_log_table_results=runner_log_table_results,
    )
    report_md = build_markdown_report(
        generated_at=generated_at,
        since=since,
        hours=args.hours,
        workflows=workflows,
        recent_jobs=recent_jobs,
        failed_jobs=failed,
        jobs_to_scan=jobs_to_scan,
        scan_results=scan_results,
    )
    write_reports(
        report_json_path=args.report_json,
        report_md_path=args.report_md,
        report_json=report_json,
        report_md=report_md,
    )

    print(f"Runner-failure jobs found: {len(failures)}")
    for runner, runner_results in sorted(group_results_by_runner(failures).items()):
        print(
            f"runner {runner}: {len(runner_results)} job(s), "
            f"{format_signature_summary(runner_results)}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
