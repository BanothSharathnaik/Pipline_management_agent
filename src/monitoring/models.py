from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional

VALID_SOURCES = {"mock", "live"}


class ResultState(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    SKIPPED = "SKIPPED"
    TIMED_OUT = "TIMED_OUT"
    UNKNOWN = "UNKNOWN"


def _parse_time(value: Any) -> Optional[datetime]:
    """ISO-8601 string -> aware datetime. None/'' -> None. Bad input -> ValueError."""
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ValueError(f"timestamp must be a string, got {type(value).__name__}")
    try:
        dt = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"bad timestamp: {value!r}") from exc
    if dt.tzinfo is None:
        raise ValueError(f"timestamp must include a timezone: {value!r}")
    return dt


def _parse_state(value: Any) -> ResultState:
    if not value:
        return ResultState.UNKNOWN
    try:
        return ResultState(str(value).upper())
    except ValueError:
        return ResultState.UNKNOWN


def _clean_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _require(data: dict, key: str) -> str:
    value = data.get(key)
    if value is None or str(value).strip() == "":
        raise ValueError(f"missing required field: {key}")
    return str(value)


def _duration(start: Optional[datetime], end: Optional[datetime]) -> Optional[float]:
    if start is None or end is None:
        return None
    seconds = (end - start).total_seconds()
    return seconds if seconds >= 0 else None  # end before start = bad data


@dataclass(frozen=True)
class TaskRecord:
    task_key: str
    result_state: ResultState
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    error_message: Optional[str]
    depends_on: tuple[str, ...]
    attempts: int = 1

    @property
    def duration_seconds(self) -> Optional[float]:
        return _duration(self.start_time, self.end_time)

    @classmethod
    def from_dict(cls, data: dict) -> "TaskRecord":
        attempts = data.get("attempts", 1)
        if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
            raise ValueError("attempts must be a positive integer")
        return cls(
            task_key=_require(data, "task_key"),
            result_state=_parse_state(data.get("result_state")),
            start_time=_parse_time(data.get("start_time")),
            end_time=_parse_time(data.get("end_time")),
            error_message=_clean_text(data.get("error_message")),
            depends_on=tuple(data.get("depends_on") or ()),
            attempts=attempts,
        )


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    job_id: str
    job_name: Optional[str]
    result_state: ResultState
    lifecycle_state: Optional[str]
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    error_message: Optional[str]
    tasks: tuple[TaskRecord, ...]
    source: str  # "mock" or "live"
    queue_seconds: Optional[float] = None

    @property
    def duration_seconds(self) -> Optional[float]:
        return _duration(self.start_time, self.end_time)

    @classmethod
    def from_dict(cls, data: dict) -> "RunRecord":
        source = _require(data, "source")
        if source not in VALID_SOURCES:
            raise ValueError(f"source must be one of {sorted(VALID_SOURCES)}, got {source!r}")
        queue = data.get("queue_seconds")
        if queue is not None and (isinstance(queue, bool) or not isinstance(queue, (int, float)) or queue < 0):
            raise ValueError("queue_seconds must be a non-negative number")
        return cls(
            run_id=_require(data, "run_id"),
            job_id=_require(data, "job_id"),
            job_name=_clean_text(data.get("job_name")),
            result_state=_parse_state(data.get("result_state")),
            lifecycle_state=_clean_text(data.get("lifecycle_state")),
            start_time=_parse_time(data.get("start_time")),
            end_time=_parse_time(data.get("end_time")),
            error_message=_clean_text(data.get("error_message")),
            tasks=tuple(TaskRecord.from_dict(t) for t in data.get("tasks") or ()),
            source=source,
            queue_seconds=float(queue) if queue is not None else None,
        )