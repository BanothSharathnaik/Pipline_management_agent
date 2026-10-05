from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from src.monitoring.models import RunRecord


@dataclass(frozen=True)
class LoadIssue:
    index: int
    run_id: str | None
    message: str


@dataclass
class LoadResult:
    runs: list[RunRecord] = field(default_factory=list)
    issues: list[LoadIssue] = field(default_factory=list)
    duplicates_dropped: int = 0
    total_records: int = 0


def load_mock_runs(path: str | Path) -> LoadResult:
    """Load mock runs. Bad records are reported in `issues`, never silently dropped."""
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"mock data file not found: {p}") from exc

    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {p}: {exc}") from exc

    if not isinstance(raw, list):
        raise ValueError(f"{p}: top level must be a JSON list of run records")

    result = LoadResult(total_records=len(raw))
    seen: dict[tuple[str, str], dict] = {}

    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            result.issues.append(LoadIssue(i, None, "record is not a JSON object"))
            continue

        try:
            run = RunRecord.from_dict(item)
        except ValueError as exc:
            rid = item.get("run_id")
            result.issues.append(LoadIssue(i, str(rid) if rid else None, str(exc)))
            continue

        if run.source != "mock":
            result.issues.append(
                LoadIssue(i, run.run_id, f"source {run.source!r} not allowed in mock file")
            )
            continue

        key = (run.job_id, run.run_id)
        if key in seen:
            result.duplicates_dropped += 1
            if seen[key] != item:
                result.issues.append(
                    LoadIssue(i, run.run_id, "conflicting duplicate; kept first occurrence")
                )
            continue

        seen[key] = item
        result.runs.append(run)

    return result