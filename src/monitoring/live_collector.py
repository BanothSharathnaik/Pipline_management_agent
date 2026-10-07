from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence

from src.monitoring.databricks_client import DatabricksError, RunsFetch
from src.monitoring.models import ResultState, RunRecord
from src.monitoring.queries import FAILURE_STATES

SNAPSHOT_PATH = Path(__file__).resolve().parents[2] / "data" / "live" / "runs_snapshot.json"
SNAPSHOT_VERSION = 1
_MIN_TIME = datetime.min.replace(tzinfo=timezone.utc)

# Observed in a real workspace: SUCCESS, FAILED. Everything else here is from memory of the
# Jobs API docs and UNVERIFIED; unknown values map to UNKNOWN with a warning.
RESULT_STATE_MAP = {
    "SUCCESS": ResultState.SUCCESS,
    "FAILED": ResultState.FAILED,
    "TIMEDOUT": ResultState.TIMED_OUT,
    "CANCELED": ResultState.CANCELLED,
    "UPSTREAM_FAILED": ResultState.SKIPPED,
    "UPSTREAM_CANCELED": ResultState.SKIPPED,
    "EXCLUDED": ResultState.SKIPPED,
    "DISABLED": ResultState.SKIPPED,
    "MAXIMUM_CONCURRENT_RUNS_REACHED": ResultState.SKIPPED,
}
TERMINAL_LIFECYCLE = frozenset({"TERMINATED", "INTERNAL_ERROR", "SKIPPED"})
TERMINAL_STATUS = frozenset({"TERMINATED", "SKIPPED"})
FETCH_OUTPUT_FOR = frozenset({"FAILED", "TIMEDOUT"})
_FAILURE_VALUES = {s.value for s in FAILURE_STATES}


class Source(Protocol):
    def list_runs(self, **kwargs) -> RunsFetch: ...
    def get_run_output(self, task_run_id: int) -> dict: ...


@dataclass(frozen=True)
class LiveCollection:
    runs: tuple[RunRecord, ...]
    warnings: tuple[str, ...]
    fetched_at: datetime
    truncated: bool
    raw_runs: tuple[dict, ...]            # sanitised raw payloads (for the snapshot)
    raw_outputs: dict[str, dict]          # task run id -> {"error": str | None}
    fetch_warnings: tuple[str, ...]


# --- small helpers --------------------------------------------------------------------------

def _dt(ms: Any) -> Optional[datetime]:
    if isinstance(ms, bool) or not isinstance(ms, (int, float)) or ms <= 0:
        return None  # Databricks uses 0 for "not set yet"
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


def _text(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    return value.strip() or None


def _dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _is_terminal(state: dict, status: dict) -> bool:
    return state.get("life_cycle_state") in TERMINAL_LIFECYCLE or status.get("state") in TERMINAL_STATUS


def _map_state(raw: Any, label: str, terminal: bool, warnings: list[str]) -> ResultState:
    if raw is None or raw == "":
        if terminal:
            warnings.append(f"{label}: finished but has no result_state; recorded as UNKNOWN")
        return ResultState.UNKNOWN
    mapped = RESULT_STATE_MAP.get(str(raw))
    if mapped is None:
        warnings.append(f"{label}: unrecognized result_state {raw!r}; recorded as UNKNOWN")
        return ResultState.UNKNOWN
    return mapped


def _attempt(task: dict) -> int:
    value = task.get("attempt_number")
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _latest_attempts(tasks: Sequence[Any]) -> list[tuple[dict, int]]:
    """One entry per task_key: the highest attempt, and how many entries that key had."""
    order: list[str] = []
    groups: dict[str, list[dict]] = {}
    for t in tasks:
        if not isinstance(t, dict) or not isinstance(t.get("task_key"), str):
            continue
        groups.setdefault(t["task_key"], [])
        if len(groups[t["task_key"]]) == 0:
            order.append(t["task_key"])
        groups[t["task_key"]].append(t)
    result = []
    for key in order:
        entries = groups[key]
        latest = max(enumerate(entries), key=lambda p: (_attempt(p[1]), p[0]))[1]
        result.append((latest, len(entries)))
    return result


def output_targets(run: Mapping[str, Any]) -> list[int]:
    """Task run ids whose error text we want: latest attempt of each failed task."""
    tasks = run.get("tasks")
    if isinstance(tasks, list) and tasks:
        return [
            t["run_id"] for t, _ in _latest_attempts(tasks)
            if _dict(t.get("state")).get("result_state") in FETCH_OUTPUT_FOR
            and isinstance(t.get("run_id"), int)
        ]
    if _dict(run.get("state")).get("result_state") in FETCH_OUTPUT_FOR and isinstance(run.get("run_id"), int):
        return [run["run_id"]]  # a run with no tasks is its own task
    return []


# --- mapping ------------------------------------------------------------------------------------

def map_run(run: Mapping[str, Any], outputs: Mapping[str, Mapping[str, Any]]) -> tuple[RunRecord, list[str]]:
    warnings: list[str] = []
    rid = run.get("run_id")
    label = f"run {rid}"
    state, status = _dict(run.get("state")), _dict(run.get("status"))
    result = _map_state(state.get("result_state"), label, _is_terminal(state, status), warnings)

    def fetched_error(task_run_id: Any) -> Optional[str]:
        entry = outputs.get(str(task_run_id))
        return _text(entry.get("error")) if isinstance(entry, Mapping) else None

    tasks: list[dict] = []
    for t, count in _latest_attempts(run.get("tasks") if isinstance(run.get("tasks"), list) else []):
        ts, tstatus = _dict(t.get("state")), _dict(t.get("status"))
        tlabel = f"{label} task {t['task_key']}"
        t_result = _map_state(ts.get("result_state"), tlabel, _is_terminal(ts, tstatus), warnings)
        error = None
        if t_result in FAILURE_STATES:
            error = fetched_error(t.get("run_id"))
            if error is None:
                error = (_text(ts.get("state_message"))
                         or _text(_dict(tstatus.get("termination_details")).get("message")))
                warnings.append(f"{tlabel}: no fetched error output; used the state message")
        raw_deps = t.get("depends_on") if isinstance(t.get("depends_on"), list) else []
        deps = [d.get("task_key") if isinstance(d, dict) else d for d in raw_deps]
        tasks.append({
            "task_key": t["task_key"],
            "result_state": t_result.value,
            "start_time": _iso(_dt(t.get("start_time"))),
            "end_time": _iso(_dt(t.get("end_time"))),
            "error_message": error,
            "depends_on": [d for d in deps if isinstance(d, str) and d],
            "attempts": count,
        })

    error = None
    if result in FAILURE_STATES:
        error = next((t["error_message"] for t in tasks
                      if t["result_state"] in _FAILURE_VALUES and t["error_message"]), None)
        if error is None and not tasks:
            error = fetched_error(rid)
        if error is None:
            error = (_text(_dict(status.get("termination_details")).get("message"))
                     or _text(state.get("state_message")))
            warnings.append(f"{label}: no task error text available; used the run's termination message"
                            if error else f"{label}: failed with no error text available")

    queue_ms = run.get("queue_duration")
    queue_seconds = (queue_ms / 1000 if isinstance(queue_ms, (int, float))
                     and not isinstance(queue_ms, bool) and queue_ms >= 0 else None)

    record = RunRecord.from_dict({
        "run_id": str(rid) if rid is not None else None,
        "job_id": str(run.get("job_id")) if run.get("job_id") is not None else None,
        "job_name": _text(run.get("run_name")),
        "source": "live",
        "result_state": result.value,
        "lifecycle_state": _text(status.get("state")) or _text(state.get("life_cycle_state")),
        "start_time": _iso(_dt(run.get("start_time"))),
        "end_time": _iso(_dt(run.get("end_time"))),
        "error_message": error,
        "tasks": tasks,
        "queue_seconds": queue_seconds,
    })
    return record, warnings


def build_collection(
    raw_runs: Sequence[dict],
    raw_outputs: Mapping[str, dict],
    fetched_at: datetime,
    truncated: bool,
    fetch_warnings: Sequence[str],
) -> LiveCollection:
    runs: list[RunRecord] = []
    warnings = list(fetch_warnings)
    for raw in raw_runs:
        try:
            record, w = map_run(raw, raw_outputs)
        except ValueError as exc:
            warnings.append(f"run {raw.get('run_id')}: skipped, {exc}")
            continue
        runs.append(record)
        warnings.extend(w)
    runs.sort(key=lambda r: (r.start_time is None, r.start_time or _MIN_TIME, r.run_id))
    return LiveCollection(
        runs=tuple(runs), warnings=tuple(warnings), fetched_at=fetched_at, truncated=truncated,
        raw_runs=tuple(raw_runs), raw_outputs=dict(raw_outputs), fetch_warnings=tuple(fetch_warnings),
    )


# --- collecting ---------------------------------------------------------------------------------

def _sanitize(run: Mapping[str, Any]) -> dict:
    clean = dict(run)
    clean.pop("creator_user_name", None)  # an email address; not needed
    return clean


def collect_live(
    client: Source,
    max_runs: int = 100,
    max_output_fetches: int = 20,
    start_time_from_ms: Optional[int] = None,
    now: Optional[datetime] = None,
) -> LiveCollection:
    if max_output_fetches < 0:
        raise ValueError("max_output_fetches must be >= 0")
    fetched_at = now or datetime.now(timezone.utc)
    if fetched_at.tzinfo is None:
        raise ValueError("now must be timezone-aware")

    fetch = client.list_runs(max_runs=max_runs, start_time_from_ms=start_time_from_ms)
    raw_runs = [_sanitize(r) for r in fetch.runs]
    fetch_warnings: list[str] = []
    if fetch.truncated:
        fetch_warnings.append(f"only the {len(raw_runs)} most recent runs were fetched; more exist")

    newest_first = sorted(raw_runs, key=lambda r: (r.get("start_time") or 0, r.get("run_id") or 0), reverse=True)
    targets: list[int] = []
    for run in newest_first:
        for tid in output_targets(run):
            if tid not in targets:
                targets.append(tid)

    outputs: dict[str, dict] = {}
    for tid in targets[:max_output_fetches]:
        try:
            out = client.get_run_output(tid)
        except DatabricksError as exc:
            fetch_warnings.append(f"error output of task run {tid} not available: {exc}")
            continue
        error = out.get("error") if isinstance(out, dict) else None
        outputs[str(tid)] = {"error": error if isinstance(error, str) else None}
    skipped = len(targets) - min(len(targets), max_output_fetches)
    if skipped:
        fetch_warnings.append(f"{skipped} task output(s) not fetched (limit {max_output_fetches})")

    return build_collection(raw_runs, outputs, fetched_at, fetch.truncated, fetch_warnings)


# --- snapshot -----------------------------------------------------------------------------------

def save_snapshot(collection: LiveCollection, path: str | Path = SNAPSHOT_PATH) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": SNAPSHOT_VERSION,
        "fetched_at": collection.fetched_at.isoformat(),
        "truncated": collection.truncated,
        "fetch_warnings": list(collection.fetch_warnings),
        "runs": list(collection.raw_runs),
        "outputs": collection.raw_outputs,
    }
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def load_snapshot(path: str | Path = SNAPSHOT_PATH) -> LiveCollection:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"live snapshot not found: {p} (run the collector first)")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"unreadable live snapshot {p}: {exc}") from exc
    if not isinstance(data, dict) or data.get("version") != SNAPSHOT_VERSION:
        raise ValueError(f"unsupported live snapshot format in {p}")
    try:
        fetched_at = datetime.fromisoformat(data["fetched_at"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"live snapshot {p} has no valid fetched_at") from exc
    if fetched_at.tzinfo is None:
        raise ValueError(f"live snapshot {p}: fetched_at must include a timezone")
    runs, outputs = data.get("runs"), data.get("outputs")
    if not isinstance(runs, list) or not all(isinstance(r, dict) for r in runs) or not isinstance(outputs, dict):
        raise ValueError(f"live snapshot {p} is malformed")
    return build_collection(runs, outputs, fetched_at, bool(data.get("truncated")),
                            [str(w) for w in data.get("fetch_warnings") or []])


def describe_age(fetched_at: datetime, now: datetime) -> str:
    """How old a snapshot is, relative to an explicit 'now'."""
    seconds = (now - fetched_at).total_seconds()
    if seconds < 0:
        return "newer than the reference time"
    if seconds < 60:
        return "less than a minute before now"
    minutes = int(seconds // 60)
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''} before now"
    hours = minutes // 60
    if hours < 48:
        return f"{hours} hour{'s' if hours != 1 else ''} before now"
    days = hours // 24
    return f"{days} day{'s' if days != 1 else ''} before now"