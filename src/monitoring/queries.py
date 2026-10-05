from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence

from src.monitoring.models import ResultState, RunRecord

FAILURE_STATES = frozenset({ResultState.FAILED, ResultState.TIMED_OUT})

_MIN_TIME = datetime.min.replace(tzinfo=timezone.utc)


def _check_aware(name: str, dt: Optional[datetime]) -> None:
    if dt is not None and dt.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")


def _sort_key(run: RunRecord) -> tuple:
    # Runs without a start time sort last; run_id breaks ties deterministically.
    return (run.start_time is None, run.start_time or _MIN_TIME, run.run_id)


def find_runs(runs: Sequence[RunRecord], run_id: str) -> list[RunRecord]:
    """All runs with this run_id. A list, because ids could collide across jobs."""
    return [r for r in runs if r.run_id == run_id]


def latest_run(runs: Sequence[RunRecord], job_id: Optional[str] = None) -> Optional[RunRecord]:
    """Run with the most recent start time. None if nothing qualifies."""
    candidates = [
        r for r in runs
        if r.start_time is not None and (job_id is None or r.job_id == job_id)
    ]
    return max(candidates, key=lambda r: (r.start_time, r.run_id), default=None)


def runs_in_window(
    runs: Sequence[RunRecord],
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> list[RunRecord]:
    """Runs whose start_time is in [start, end). With no bounds, returns all runs."""
    _check_aware("start", start)
    _check_aware("end", end)
    if start is not None and end is not None and start > end:
        raise ValueError("start must not be after end")
    if start is None and end is None:
        return list(runs)

    selected = []
    for r in runs:
        t = r.start_time
        if t is None:
            continue
        if start is not None and t < start:
            continue
        if end is not None and t >= end:
            continue
        selected.append(r)
    return selected


def failed_runs(
    runs: Sequence[RunRecord],
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    job_id: Optional[str] = None,
    states: frozenset[ResultState] = FAILURE_STATES,
) -> list[RunRecord]:
    """Failed runs, oldest first, optionally within a window and/or for one job."""
    selected = runs_in_window(runs, start, end)
    hits = [
        r for r in selected
        if r.result_state in states and (job_id is None or r.job_id == job_id)
    ]
    return sorted(hits, key=_sort_key)


def count_failures(
    runs: Sequence[RunRecord],
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    job_id: Optional[str] = None,
) -> int:
    return len(failed_runs(runs, start, end, job_id))


def failed_in_last(
    runs: Sequence[RunRecord],
    now: datetime,
    hours: float,
    job_id: Optional[str] = None,
) -> list[RunRecord]:
    """Failures that started in [now - hours, now). `now` is passed in, never read."""
    _check_aware("now", now)
    if hours <= 0:
        raise ValueError("hours must be positive")
    return failed_runs(runs, now - timedelta(hours=hours), now, job_id)


def runs_exceeding(runs: Sequence[RunRecord], threshold_seconds: float) -> list[RunRecord]:
    """Runs with duration strictly greater than the threshold, longest first.
    Runs with unknown duration are excluded, not guessed."""
    if threshold_seconds < 0:
        raise ValueError("threshold_seconds must be >= 0")
    hits = [
        r for r in runs
        if r.duration_seconds is not None and r.duration_seconds > threshold_seconds
    ]
    return sorted(hits, key=lambda r: (-r.duration_seconds, r.run_id))


def status_counts(runs: Sequence[RunRecord]) -> dict[ResultState, int]:
    return dict(Counter(r.result_state for r in runs))


@dataclass(frozen=True)
class ErrorGroup:
    message: str
    run_ids: tuple[str, ...]
    count: int


def _normalize(message: str) -> str:
    return " ".join(message.lower().split())


def repeated_errors(runs: Sequence[RunRecord], min_count: int = 2) -> list[ErrorGroup]:
    """Run-level error messages that repeat EXACTLY (ignoring case and whitespace).
    Similar-but-different wording is NOT grouped here; that is semantic retrieval's job."""
    if min_count < 2:
        raise ValueError("min_count must be at least 2")
    groups: dict[str, list[RunRecord]] = defaultdict(list)
    for r in sorted(runs, key=_sort_key):
        if r.error_message:
            groups[_normalize(r.error_message)].append(r)

    result = [
        ErrorGroup(
            message=rs[0].error_message or "",
            run_ids=tuple(x.run_id for x in rs),
            count=len(rs),
        )
        for rs in groups.values()
        if len(rs) >= min_count
    ]
    return sorted(result, key=lambda g: (-g.count, g.message))