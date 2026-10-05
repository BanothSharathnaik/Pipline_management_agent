import pytest

from src.monitoring.models import ResultState, RunRecord


def base(**overrides):
    d = {
        "run_id": "r1",
        "job_id": "j1",
        "source": "mock",
        "result_state": "failed",
        "start_time": "2026-10-01T02:00:00+00:00",
        "end_time": "2026-10-01T02:10:30+00:00",
        "error_message": "AnalysisException: column x not found",
    }
    d.update(overrides)
    return d


def test_valid_run_parses():
    r = RunRecord.from_dict(base())
    assert r.result_state == ResultState.FAILED
    assert r.duration_seconds == 630.0
    assert r.source == "mock"


def test_missing_end_time_gives_none_duration():
    r = RunRecord.from_dict(base(end_time=None))
    assert r.duration_seconds is None


def test_end_before_start_gives_none_duration():
    r = RunRecord.from_dict(base(end_time="2026-10-01T01:00:00+00:00"))
    assert r.duration_seconds is None


def test_missing_run_id_raises():
    with pytest.raises(ValueError):
        RunRecord.from_dict(base(run_id=None))


def test_invalid_source_raises():
    with pytest.raises(ValueError):
        RunRecord.from_dict(base(source="prod"))


def test_naive_timestamp_raises():
    with pytest.raises(ValueError):
        RunRecord.from_dict(base(start_time="2026-10-01T02:00:00"))


def test_unknown_state_maps_to_unknown():
    assert RunRecord.from_dict(base(result_state="weird")).result_state == ResultState.UNKNOWN
    assert RunRecord.from_dict(base(result_state=None)).result_state == ResultState.UNKNOWN


def test_blank_error_becomes_none():
    assert RunRecord.from_dict(base(error_message="   ")).error_message is None


def test_tasks_parse_with_dependencies():
    r = RunRecord.from_dict(
        base(tasks=[
            {"task_key": "ingest", "result_state": "success"},
            {"task_key": "transform", "result_state": "failed", "depends_on": ["ingest"]},
        ])
    )
    assert len(r.tasks) == 2
    assert r.tasks[1].depends_on == ("ingest",)