from datetime import datetime, timezone

import pytest

from src.app import main
from src.monitoring.databricks_client import DatabricksError, RunsFetch
from src.monitoring.live_collector import collect_live, describe_age, save_snapshot

UTC = timezone.utc
FETCHED = datetime(2026, 10, 7, 5, 58, tzinfo=UTC)
NOW = "--now=2026-10-07T06:30:00+00:00"
BASE = int(datetime(2026, 10, 7, 3, 0, tzinfo=UTC).timestamp() * 1000)
ERROR = "Exception: Deliberate failure: Testing Databricks job error handling."


def at(seconds):
    return BASE + int(seconds * 1000)


def task(run_id, attempt, result):
    return {"run_id": run_id, "task_key": "task_failure", "attempt_number": attempt,
            "state": {"life_cycle_state": "TERMINATED", "result_state": result, "state_message": "x"},
            "status": {"state": "TERMINATED"}, "start_time": at(25), "end_time": at(100)}


RUN_OK = {
    "job_id": 568, "run_id": 799790931046816, "run_name": "JOB-SUCCESS",
    "state": {"life_cycle_state": "TERMINATED", "result_state": "SUCCESS", "state_message": ""},
    "status": {"state": "TERMINATED"}, "start_time": at(43), "end_time": at(57),
    "tasks": [task(1, 0, "SUCCESS")],
}
RUN_FAIL = {
    "job_id": 658, "run_id": 437531692562553, "run_name": "JOB-FAILURA",
    "state": {"life_cycle_state": "INTERNAL_ERROR", "result_state": "FAILED", "state_message": "x"},
    "status": {"state": "TERMINATED"}, "start_time": at(25), "end_time": at(127),
    "tasks": [task(10, 0, "FAILED"), task(11, 1, "FAILED")],
}


class FakeClient:
    def __init__(self, error=None):
        self.error = error

    def list_runs(self, **kwargs):
        if self.error:
            raise self.error
        return RunsFetch([dict(RUN_OK), dict(RUN_FAIL)], 1, False, 0)

    def get_run_output(self, task_run_id):
        return {"error": ERROR}


@pytest.fixture
def snap(tmp_path):
    path = tmp_path / "snap.json"
    save_snapshot(collect_live(FakeClient(), now=FETCHED), path)
    return path


def run(capsys, snap, *args):
    code = main([NOW, "--source", "live", "--snapshot", str(snap), *args])
    return code, capsys.readouterr()


def test_live_summary_and_banner(capsys, snap):
    code, out = run(capsys, snap, "summary")
    assert code == 0
    assert "[LIVE DATA] 2 runs from a snapshot fetched 2026-10-07 05:58 UTC (32 minutes before now)" in out.out
    assert "  FAILED     1" in out.out and "  SUCCESS    1" in out.out


def test_live_failed_lists_only_the_failed_run(capsys, snap):
    _, out = run(capsys, snap, "failed")
    assert "437531692562553" in out.out and "799790931046816" not in out.out


def test_live_run_detail(capsys, snap):
    _, out = run(capsys, snap, "run", "437531692562553")
    assert "Source:     live" in out.out and "attempts=2" in out.out and ERROR in out.out


def test_missing_snapshot(capsys, tmp_path):
    code, out = run(capsys, tmp_path / "none.json", "summary")
    assert code == 2 and "collector" in out.err


def test_source_flags_cannot_cross(capsys, snap):
    code = main([NOW, "--source", "live", "--snapshot", str(snap), "--data", "x.json", "summary"])
    assert code == 2 and "--data only applies" in capsys.readouterr().err
    code = main([NOW, "--snapshot", str(snap), "summary"])
    assert code == 2 and "--snapshot only applies" in capsys.readouterr().err


def test_collect_writes_snapshot_that_live_commands_can_read(capsys, tmp_path, monkeypatch):
    class Fake:
        @staticmethod
        def from_env(**kwargs):
            return FakeClient()

    monkeypatch.setattr("src.monitoring.databricks_client.DatabricksClient", Fake)
    path = tmp_path / "out" / "snap.json"
    code = main([NOW, "--snapshot", str(path), "collect"])
    out = capsys.readouterr().out
    assert code == 0 and "[LIVE DATA] 2 runs fetched" in out and path.is_file()
    code = main([NOW, "--source", "live", "--snapshot", str(path), "summary"])
    assert code == 0 and "[LIVE DATA] 2 runs from a snapshot" in capsys.readouterr().out


def test_collect_without_credentials(capsys, monkeypatch, tmp_path):
    monkeypatch.delenv("DATABRICKS_HOST", raising=False)
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)
    path = tmp_path / "snap.json"
    code = main([NOW, "--snapshot", str(path), "collect"])
    assert code == 2 and "must be set" in capsys.readouterr().err and not path.exists()


def test_collect_api_error_saves_nothing(capsys, tmp_path, monkeypatch):
    class Fake:
        @staticmethod
        def from_env(**kwargs):
            return FakeClient(error=DatabricksError("Databricks returned HTTP 401: nope", 401))

    monkeypatch.setattr("src.monitoring.databricks_client.DatabricksClient", Fake)
    path = tmp_path / "snap.json"
    code = main([NOW, "--snapshot", str(path), "collect"])
    assert code == 2 and "401" in capsys.readouterr().err and not path.exists()


def test_ask_on_live_data_with_a_structured_question(capsys, snap):
    code, out = run(capsys, snap, "ask", "--no-llm", "Which pipelines failed today?")
    assert code == 0
    assert "437531692562553" in out.out and "799790931046816" not in out.out.split("Failed runs")[1]
    assert "Data note: live snapshot fetched" in out.out
    assert "not written about these live runs" in out.out


@pytest.mark.parametrize("seconds,expected", [
    (30, "less than a minute before now"), (60, "1 minute before now"),
    (59 * 60, "59 minutes before now"), (90 * 60, "1 hour before now"),
    (5 * 3600, "5 hours before now"), (50 * 3600, "2 days before now"),
    (-5, "newer than the reference time"),
])
def test_describe_age(seconds, expected):
    from datetime import timedelta
    assert describe_age(FETCHED, FETCHED + timedelta(seconds=seconds)) == expected
