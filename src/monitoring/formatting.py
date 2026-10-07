from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from src.monitoring.models import RunRecord


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


def run_detail_lines(r: RunRecord) -> list[str]:
    lines = [
        f"Run:        {r.run_id}  (job {r.job_id}, {r.job_name or 'unnamed'})",
        f"Source:     {r.source}",
        f"Result:     {r.result_state.value}",
        f"Lifecycle:  {r.lifecycle_state or 'n/a'}",
        f"Started:    {fmt_time(r.start_time)}",
        f"Ended:      {fmt_time(r.end_time)}",
        f"Duration:   {fmt_duration(r.duration_seconds)}",
    ]
    if r.queue_seconds is not None:
        lines.append(f"Queued:     {fmt_duration(r.queue_seconds)} (may be included in the duration above)")
    lines.append(f"Error:      {r.error_message or '(no error message recorded)'}")
    if r.tasks:
        lines.append("Tasks:")
        for t in r.tasks:
            deps = ",".join(t.depends_on) if t.depends_on else "-"
            retries = f"  attempts={t.attempts}" if t.attempts > 1 else ""
            lines.append(f"  - {t.task_key}  {t.result_state.value}  depends_on=[{deps}]{retries}")
            if t.error_message:
                lines.append(f"      error: {t.error_message}")
    else:
        lines.append("Tasks:      (none recorded)")
    return lines