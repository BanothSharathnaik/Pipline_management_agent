import json
from pathlib import Path

import pytest

from src.monitoring.collector import load_mock_runs
from src.monitoring.models import ResultState

MOCK = Path(__file__).resolve().parents[1] / "data" / "mock_runs.json"


def by_id(result, run_id):
    return next(r for r in result.runs if r.run_id == run_id)


def test_counts_on_real_mock_file():
    res = load_mock_runs(MOCK)
    assert res.total_records == 20
    assert len(res.runs) == 16
    assert len(res.issues) == 3
    assert res.duplicates_dropped == 1


def test_all_loaded_runs_are_mock():
    assert {r.source for r in load_mock_runs(MOCK).runs} == {"mock"}


def test_running_run_has_no_end_or_duration():
    r = by_id(load_mock_runs(MOCK), "r1005")
    assert r.end_time is None
    assert r.duration_seconds is None
    assert r.lifecycle_state == "RUNNING"


def test_failed_run_with_no_error_text():
    r = by_id(load_mock_runs(MOCK), "r2005")
    assert r.result_state == ResultState.FAILED
    assert r.error_message is None


def test_task_dependencies_preserved():
    r = by_id(load_mock_runs(MOCK), "r2002")
    assert [t.task_key for t in r.tasks] == [
        "ingest_customers", "transform_customers", "publish"
    ]
    assert r.tasks[2].depends_on == ("transform_customers",)


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_mock_runs("does_not_exist.json")


def test_invalid_json_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    with pytest.raises(ValueError):
        load_mock_runs(p)


def test_non_list_top_level_raises(tmp_path):
    p = tmp_path / "obj.json"
    p.write_text('{"run_id": "x"}')
    with pytest.raises(ValueError):
        load_mock_runs(p)


def test_conflicting_duplicate_keeps_first_and_reports(tmp_path):
    a = {"run_id": "x1", "job_id": "j", "source": "mock", "result_state": "SUCCESS"}
    b = {"run_id": "x1", "job_id": "j", "source": "mock", "result_state": "FAILED"}
    p = tmp_path / "dup.json"
    p.write_text(json.dumps([a, b]))
    res = load_mock_runs(p)
    assert len(res.runs) == 1
    assert res.runs[0].result_state == ResultState.SUCCESS
    assert any("conflicting" in i.message for i in res.issues)


def test_live_record_rejected_from_mock_file(tmp_path):
    rec = {"run_id": "L1", "job_id": "j", "source": "live", "result_state": "SUCCESS"}
    p = tmp_path / "live.json"
    p.write_text(json.dumps([rec]))
    res = load_mock_runs(p)
    assert res.runs == []
    assert len(res.issues) == 1