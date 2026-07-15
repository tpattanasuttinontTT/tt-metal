#!/usr/bin/env python3
"""Build a runner-specific failure report."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from runner_failure_common import (
    JobScanResult,
    RecentJob,
    SIGNATURE_VERSION,
    WorkflowTarget,
    ensure_gh_available,
    format_signature_summary,
    format_utc,
    job_state_key,
    job_to_dict,
    list_recent_jobs,
    load_workflows,
    markdown_escape,
    markdown_link,
    parse_github_time,
    result_to_dict,
    runner_name_for_job,
    scan_jobs,
    signature_counts,
    write_reports,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a check-logs-all-style report for one GitHub Actions runner."
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
        "--runner-name",
        help="Runner/host name to report on.",
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
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.hours <= 0:
        raise ValueError("--hours must be greater than zero.")
    if args.gh_timeout <= 0:
        raise ValueError("--gh-timeout must be greater than zero.")
    if args.log_workers <= 0:
        raise ValueError("--log-workers must be greater than zero.")
    if not args.runner_name or not args.runner_name.strip():
        raise ValueError("--runner-name is required.")


def build_runner_markdown_report(
    *,
    generated_at: datetime,
    since: datetime,
    hours: int,
    runner_name: str,
    workflows: list[WorkflowTarget],
    runner_jobs: list[RecentJob],
    scan_results: list[JobScanResult],
) -> str:
    failures = [result for result in scan_results if result.signature_labels]
    lines = [
        "# Runner Failure Report",
        "",
        f"- Generated at: `{format_utc(generated_at)}`",
        f"- Runner: `{runner_name}`",
        f"- Window: last `{hours}` hour(s), since `{format_utc(since)}`",
        f"- Workflows: `{', '.join(workflow.name for workflow in workflows)}`",
        f"- Recent configured-workflow jobs on runner: `{len(runner_jobs)}`",
        f"- Scanned jobs: `{sum(1 for result in scan_results if result.log_checked)}`",
        f"- Runner-failure jobs: `{len(failures)}`",
        "",
    ]

    if not scan_results:
        lines.append("No configured-workflow jobs found for this runner.")
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "| GH job | Workflow | Job name | Status | Started | Log checked | Signatures |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for result in scan_results:
        signatures = ", ".join(result.signature_labels)
        if result.fabric_missing_links:
            signatures += f"; missing links: {result.fabric_missing_links}"
        lines.append(
            "| "
            f"{markdown_link(result.job.job_id or 'open', result.job.html_url)} | "
            f"{markdown_escape(result.job.workflow)} | "
            f"{markdown_escape(result.job.name)} | "
            f"{markdown_escape(result.job.conclusion or result.job.status or 'unknown')} | "
            f"{markdown_escape(result.job.started_at)} | "
            f"{'YES' if result.log_checked else 'NO'} | "
            f"{markdown_escape(signatures)} |"
        )
    return "\n".join(lines) + "\n"


def build_runner_json_report(
    *,
    generated_at: datetime,
    since: datetime,
    hours: int,
    runner_name: str,
    workflows: list[WorkflowTarget],
    runner_jobs: list[RecentJob],
    scan_results: list[JobScanResult],
) -> dict[str, Any]:
    failures = [result for result in scan_results if result.signature_labels]
    return {
        "type": "runner_report",
        "generated_at": format_utc(generated_at),
        "since": format_utc(since),
        "hours": hours,
        "runner_name": runner_name,
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
            "runner_jobs": len(runner_jobs),
            "scanned_jobs": len(scan_results),
            "log_checked_jobs": sum(
                1 for result in scan_results if result.log_checked
            ),
            "runner_failure_jobs": len(failures),
        },
        "signature_counts": signature_counts(failures),
        "recent_jobs": [job_to_dict(job) for job in runner_jobs],
        "scan_results": [result_to_dict(result) for result in scan_results],
        "runner_log_table_results": [
            result_to_dict(result) for result in scan_results
        ],
    }


def should_scan_log_for_full_table(job: RecentJob) -> bool:
    conclusion = job.conclusion.lower()
    return bool(job.job_id) and conclusion not in {"skipped", "cancelled"}


def runner_report_results(
    *,
    runner_jobs: list[RecentJob],
    gh_timeout: int,
    log_workers: int,
) -> list[JobScanResult]:
    jobs_to_scan = [
        job for job in runner_jobs if should_scan_log_for_full_table(job)
    ]
    results_by_key: dict[str, JobScanResult] = {}

    if jobs_to_scan:
        for result in scan_jobs(
            jobs_to_scan,
            gh_timeout=gh_timeout,
            log_workers=log_workers,
        ):
            results_by_key[job_state_key(result.job)] = result

    for job in runner_jobs:
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
            runner_jobs,
            key=lambda runner_job: parse_github_time(runner_job.started_at),
            reverse=True,
        )
    ]


def build_runner_report(args: argparse.Namespace) -> int:
    runner_name = args.runner_name.strip()
    ensure_gh_available()
    workflows = load_workflows(args.config)
    generated_at = datetime.now(timezone.utc)
    since = generated_at - timedelta(hours=args.hours)
    recent_jobs = list_recent_jobs(workflows, since, gh_timeout=args.gh_timeout)
    runner_jobs = [
        job
        for job in recent_jobs
        if runner_name_for_job(job).casefold() == runner_name.casefold()
    ]
    print(
        f"Found {len(runner_jobs)} configured-workflow job(s) on runner "
        f"{runner_name!r} in the last {args.hours} hour(s)."
    )

    scan_results = runner_report_results(
        runner_jobs=runner_jobs,
        gh_timeout=args.gh_timeout,
        log_workers=args.log_workers,
    )
    report_json = build_runner_json_report(
        generated_at=generated_at,
        since=since,
        hours=args.hours,
        runner_name=runner_name,
        workflows=workflows,
        runner_jobs=runner_jobs,
        scan_results=scan_results,
    )
    report_md = build_runner_markdown_report(
        generated_at=generated_at,
        since=since,
        hours=args.hours,
        runner_name=runner_name,
        workflows=workflows,
        runner_jobs=runner_jobs,
        scan_results=scan_results,
    )
    write_reports(
        report_json_path=args.report_json,
        report_md_path=args.report_md,
        report_json=report_json,
        report_md=report_md,
    )

    failures = [result for result in scan_results if result.signature_labels]
    print(f"Runner-failure jobs found: {len(failures)}")
    if failures:
        print(f"runner {runner_name}: {format_signature_summary(failures)}")
    return 0


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        return build_runner_report(args)
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
