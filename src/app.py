from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from src.monitoring import queries as q
from src.monitoring.collector import load_mock_runs
from src.monitoring.models import RunRecord

DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "mock_runs.json"


def fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "n/a"
    total = int(seconds)
    return f"{total // 60}m{total % 60:02d}s"


def fmt_time(dt: Optional[datetime]) -> str:
    if dt is None:
        return "n/a"
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def fmt_run(r: RunRecord) -> str:
    return (
        f"{r.run_id}  {r.job_name or r.job_id}  {r.result_state.value}  "
        f"{fmt_time(r.start_time)}  {fmt_duration(r.duration_seconds)}"
    )


def print_runs(runs: Sequence[RunRecord], empty_message: str) -> None:
    if not runs:
        print(empty_message)
        return
    for r in runs:
        print(fmt_run(r))


def print_run_detail(r: RunRecord) -> None:
    print(f"Run:        {r.run_id}  (job {r.job_id}, {r.job_name or 'unnamed'})")
    print(f"Source:     {r.source}")
    print(f"Result:     {r.result_state.value}")
    print(f"Lifecycle:  {r.lifecycle_state or 'n/a'}")
    print(f"Started:    {fmt_time(r.start_time)}")
    print(f"Ended:      {fmt_time(r.end_time)}")
    print(f"Duration:   {fmt_duration(r.duration_seconds)}")
    print(f"Error:      {r.error_message or '(no error message recorded)'}")
    if r.tasks:
        print("Tasks:")
        for t in r.tasks:
            deps = ",".join(t.depends_on) if t.depends_on else "-"
            print(f"  - {t.task_key}  {t.result_state.value}  depends_on=[{deps}]")
            if t.error_message:
                print(f"      error: {t.error_message}")
    else:
        print("Tasks:      (none recorded)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python3 -m src.app")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA, help="mock runs JSON file")
    p.add_argument("--now", help="ISO time with timezone; default is the current time")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("summary", help="counts per result state")

    s = sub.add_parser("latest", help="most recent run")
    s.add_argument("--job", help="job_id filter")

    s = sub.add_parser("failed", help="failed or timed-out runs")
    s.add_argument("--hours", type=float, help="only the last N hours before --now")
    s.add_argument("--job", help="job_id filter")

    s = sub.add_parser("slow", help="runs longer than a threshold")
    s.add_argument("--threshold", type=float, required=True, help="seconds")

    sub.add_parser("repeats", help="exactly repeated error messages")

    s = sub.add_parser("run", help="details of one run")
    s.add_argument("run_id")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        now = datetime.fromisoformat(args.now) if args.now else datetime.now(timezone.utc)
        if now.tzinfo is None:
            raise ValueError("--now must include a timezone, e.g. 2026-10-05T12:00:00+00:00")
        loaded = load_mock_runs(args.data)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    runs = loaded.runs
    print(
        f"[MOCK DATA] {len(runs)} runs loaded | "
        f"{loaded.duplicates_dropped} duplicate dropped | "
        f"{len(loaded.issues)} records rejected"
    )
    for issue in loaded.issues:
        print(f"  load issue at index {issue.index} ({issue.run_id}): {issue.message}")
    print()

    cmd = args.command
    if cmd == "summary":
        for state, n in sorted(q.status_counts(runs).items(), key=lambda kv: kv[0].value):
            print(f"  {state.value:<10} {n}")

    elif cmd == "latest":
        r = q.latest_run(runs, args.job)
        if r is None:
            print("No run found.")
            return 1
        print_run_detail(r)

    elif cmd == "failed":
        if args.hours is not None:
            try:
                hits = q.failed_in_last(runs, now, args.hours, args.job)
            except ValueError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
        else:
            hits = q.failed_runs(runs, job_id=args.job)
        print_runs(hits, "No failed runs in selection.")

    elif cmd == "slow":
        try:
            hits = q.runs_exceeding(runs, args.threshold)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print_runs(hits, f"No runs exceed {args.threshold:g} seconds.")

    elif cmd == "repeats":
        groups = q.repeated_errors(runs)
        if not groups:
            print("No exactly repeated error messages.")
        for g in groups:
            print(f"{g.count}x  runs={list(g.run_ids)}  {g.message}")

    elif cmd == "run":
        found = q.find_runs(runs, args.run_id)
        if not found:
            print(f"No run found with id {args.run_id}.")
            return 1
        for r in found:
            print_run_detail(r)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())