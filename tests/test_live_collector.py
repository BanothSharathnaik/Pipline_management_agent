import copy
import json
import os
from datetime import datetime, timezone

import pytest

from src.monitoring.databricks_client import DatabricksError, RunsFetch
from src.monitoring.live_collector import (
    build_collection, collect_live, load_snapshot, map_run, output_targets, save_snapshot,
)
from src.monitoring.models import ResultState, RunRecord, TaskRecord

BASE = datetime(2026, 10, 7, 3, 0, tzinfo=timezone.utc)
BASE_MS = int(BASE.timestamp() * 1000)
UTC = timezone.utc


def at(seconds):
    return BASE_MS + round(seconds * 1000)


def state(lc, result=None, msg=""):
    d = {"life_cycle_state": lc, "state_message": msg, "user_cancelled_or_timedout": False}
    if result:
        d["result_state"] = result
    return d


def status(s, code=None, msg=""):
    d = {"state": s}
    if code:
        d["termination_details"] = {"code": code, "type": "CLIENT_ERROR", "message": msg}
    return d


def task(run_id, key, lc, result=None, attempt=0, msg="", st="TERMINATED", **extra):
    t = {"run_id": run_id, "task_key": key, "attempt_number": attempt,
         "state": state(lc, result, msg), "status": status(st), "start_time": at(25), "end_time": at(100)}
    t.update(extra)
    return t


RUN_OK = {
    "job_id": 568297559042585, "run_id": 799790931046816, "run_name": "JOB-SUCCESS",
    "creator_user_name": "someone@example.com",
    "state": state("TERMINATED", "SUCCESS"), "status": status("TERMINATED", "SUCCESS"),
    "start_time": at(43), "end_time": at(57), "queue_duration": 0, "run_duration": 14000,
    "tasks": [task(689493286361158, "task_success", "TERMINATED", "SUCCESS", start_time=at(43), end_time=at(57))],
}

TASK_MSG = "Workload failed, see run output for details"
RUN_FAIL = {
    "job_id": 658020425585260, "run_id": 437531692562553, "run_name": "JOB-FAILURA",
    "creator_user_name": "someone@example.com",
    "state": state("INTERNAL_ERROR", "FAILED", "Task task_failure failed with message: " + TASK_MSG + "."),
    "status": status("TERMINATED", "RUN_EXECUTION_ERROR", "Task task_failure failed with message: " + TASK_MSG + "."),
    "start_time": at(25), "end_time": at(127.643), "queue_duration": 0,
    "tasks": [
        task(584755969648631, "task_failure", "TERMINATED", "FAILED", attempt=0, msg=TASK_MSG),
        task(731249776086533, "task_failure", "TERMINATED", "FAILED", attempt=1, msg=TASK_MSG),
    ],
}

RUN_RUNNING = {
    "job_id": 658020425585260, "run_id": 111, "run_name": "JOB-FAILURA",
    "state": state("RUNNING"), "status": status("RUNNING"),
    "start_time": at(25), "end_time": 0,
    "tasks": [
        task(1, "task_failure", "TERMINATED", "FAILED", attempt=0, msg=TASK_MSG),
        task(2, "task_failure", "PENDING", None, attempt=1, st="QUEUED"),
    ],
}

RUN_QUEUED = {
    "job_id": 658020425585260, "run_id": 222, "run_name": "JOB-FAILURA",
    "state": state("QUEUED"), "status": status("QUEUED"), "start_time": at(54), "end_time": 0,
    "tasks": [task(3, "task_failure", "QUEUED", None, st="QUEUED")],
}

OUTPUTS = {
    "731249776086533": {"error": "Exception: Deliberate failure: Testing Databricks job error handling."},
    "584755969648631": {"error": "OLD attempt error"},
}


def run_with(base, **overrides):
    r = copy.deepcopy(base)
    r.update(overrides)
    return r


# --- model additions -----------------------------------------------------------------------

def test_model_defaults_and_validation():
    t = TaskRecord.from_dict({"task_key": "a"})
    assert t.attempts == 1
    assert TaskRecord.from_dict({"task_key": "a", "attempts": 3}).attempts == 3
    for bad in (0, -1, True, "2"):
        with pytest.raises(ValueError):
            TaskRecord.from_dict({"task_key": "a", "attempts": bad})
    base = {"run_id": "1", "job_id": "j", "source": "live"}
    assert RunRecord.from_dict(base).queue_seconds is None
    assert RunRecord.from_dict({**base, "queue_seconds": 3}).queue_seconds == 3.0
    for bad in (-1, True, "x"):
        with pytest.raises(ValueError):
            RunRecord.from_dict({**base, "queue_seconds": bad})


# --- mapping --------------------------------------------------------------------------------

def test_success_run():
    rec, warnings = map_run(RUN_OK, {})
    assert warnings == []
    assert (rec.run_id, rec.job_id, rec.job_name) == ("799790931046816", "568297559042585", "JOB-SUCCESS")
    assert (rec.source, rec.result_state, rec.lifecycle_state) == ("live", ResultState.SUCCESS, "TERMINATED")
    assert rec.duration_seconds == 14.0 and rec.error_message is None
    assert rec.start_time == datetime(2026, 10, 7, 3, 0, 43, tzinfo=UTC)
    assert [(t.task_key, t.result_state) for t in rec.tasks] == [("task_success", ResultState.SUCCESS)]


def test_failed_retry_run_uses_latest_attempt_and_its_output():
    rec, warnings = map_run(RUN_FAIL, OUTPUTS)
    assert warnings == []
    assert rec.result_state == ResultState.FAILED
    assert rec.lifecycle_state == "TERMINATED"           # status.state, not INTERNAL_ERROR
    assert rec.duration_seconds == pytest.approx(102.643, abs=1e-3)
    assert len(rec.tasks) == 1 and rec.tasks[0].attempts == 2
    expected = "Exception: Deliberate failure: Testing Databricks job error handling."
    assert rec.tasks[0].error_message == expected and rec.error_message == expected


def test_missing_output_falls_back_to_state_message_with_warning():
    rec, warnings = map_run(RUN_FAIL, {})
    assert rec.tasks[0].error_message == TASK_MSG and rec.error_message == TASK_MSG
    assert any("no fetched error output" in w for w in warnings)


def test_running_run_has_no_end_duration_or_result():
    rec, warnings = map_run(RUN_RUNNING, {})
    assert warnings == []
    assert rec.end_time is None and rec.duration_seconds is None
    assert rec.result_state == ResultState.UNKNOWN and rec.lifecycle_state == "RUNNING"
    assert rec.tasks[0].attempts == 2 and rec.tasks[0].result_state == ResultState.UNKNOWN


def test_queued_run_has_start_but_no_duration():
    rec, warnings = map_run(RUN_QUEUED, {})
    assert warnings == [] and rec.lifecycle_state == "QUEUED"
    assert rec.start_time is not None and rec.duration_seconds is None


def test_lifecycle_falls_back_to_life_cycle_state():
    run = run_with(RUN_OK)
    del run["status"]
    assert map_run(run, {})[0].lifecycle_state == "TERMINATED"


def test_unrecognized_result_state_is_unknown_with_warning():
    rec, warnings = map_run(run_with(RUN_OK, state=state("TERMINATED", "SOMETHING_NEW")), {})
    assert rec.result_state == ResultState.UNKNOWN
    assert any("SOMETHING_NEW" in w for w in warnings)


def test_finished_run_without_result_warns():
    rec, warnings = map_run(run_with(RUN_OK, state=state("TERMINATED")), {})
    assert rec.result_state == ResultState.UNKNOWN
    assert any("no result_state" in w for w in warnings)


@pytest.mark.parametrize("raw,expected", [
    ("TIMEDOUT", ResultState.TIMED_OUT), ("CANCELED", ResultState.CANCELLED),
    ("UPSTREAM_FAILED", ResultState.SKIPPED), ("EXCLUDED", ResultState.SKIPPED),
])
def test_unverified_state_table(raw, expected):
    rec, _ = map_run(run_with(RUN_OK, state=state("TERMINATED", raw)), {})
    assert rec.result_state == expected


def test_queue_seconds():
    assert map_run(run_with(RUN_OK, queue_duration=74000), {})[0].queue_seconds == 74.0
    assert map_run(RUN_OK, {})[0].queue_seconds == 0.0
    run = run_with(RUN_OK)
    del run["queue_duration"]
    assert map_run(run, {})[0].queue_seconds is None


def test_depends_on_parsing_ignores_garbage():
    t = task(5, "b", "TERMINATED", "SUCCESS")
    t["depends_on"] = [{"task_key": "a"}, {"nope": 1}, "c", 5]
    rec, _ = map_run(run_with(RUN_OK, tasks=[t]), {})
    assert rec.tasks[0].depends_on == ("a", "c")


def test_failed_run_without_tasks_uses_its_own_output():
    run = {"job_id": 1, "run_id": 11, "run_name": "legacy", "state": state("TERMINATED", "FAILED"),
           "status": status("TERMINATED"), "start_time": at(1), "end_time": at(5)}
    assert output_targets(run) == [11]
    rec, warnings = map_run(run, {"11": {"error": "boom"}})
    assert rec.error_message == "boom" and rec.tasks == () and warnings == []


def test_end_before_start_gives_no_duration():
    assert map_run(run_with(RUN_OK, end_time=at(1)), {})[0].duration_seconds is None


def test_missing_job_id_raises():
    run = run_with(RUN_OK)
    del run["job_id"]
    with pytest.raises(ValueError, match="job_id"):
        map_run(run, {})


def test_output_targets_only_latest_failed_attempt():
    assert output_targets(RUN_FAIL) == [731249776086533]
    assert output_targets(RUN_OK) == []
    assert output_targets(RUN_RUNNING) == []   # latest attempt is still pending
    assert output_targets(RUN_QUEUED) == []


def test_build_collection_skips_bad_runs_with_warning_and_sorts():
    bad = run_with(RUN_OK)
    del bad["job_id"]
    col = build_collection([RUN_OK, bad, RUN_FAIL], {}, BASE, False, ["earlier warning"])
    assert [r.run_id for r in col.runs] == ["437531692562553", "799790931046816"]  # oldest first
    assert col.warnings[0] == "earlier warning"
    assert any("skipped" in w for w in col.warnings)


# --- collecting ---------------------------------------------------------------------------------

class FakeClient:
    def __init__(self, runs, outputs=None, fail=(), truncated=False):
        self.runs, self.outputs, self.fail, self.truncated = runs, outputs or {}, set(fail), truncated
        self.calls, self.list_kwargs = [], None

    def list_runs(self, **kwargs):
        self.list_kwargs = kwargs
        return RunsFetch(copy.deepcopy(self.runs), 1, self.truncated, 0)

    def get_run_output(self, task_run_id):
        self.calls.append(task_run_id)
        if task_run_id in self.fail:
            raise DatabricksError("boom", 500)
        return self.outputs.get(task_run_id, {"error": None})


NOW = datetime(2026, 10, 7, 4, 0, tzinfo=UTC)
FAIL_OUT = {731249776086533: {"error": "Exception: X", "error_trace": "long trace"}}


def test_collect_fetches_only_needed_outputs_and_maps():
    client = FakeClient([RUN_OK, RUN_FAIL, RUN_QUEUED], FAIL_OUT)
    col = collect_live(client, now=NOW)
    assert client.calls == [731249776086533]
    assert [r.run_id for r in col.runs] == ["437531692562553", "799790931046816", "222"]
    assert col.runs[0].error_message == "Exception: X" and col.fetched_at == NOW and not col.truncated
    assert col.raw_outputs == {"731249776086533": {"error": "Exception: X"}}   # trace not kept


def test_collect_passes_limits_to_the_client():
    client = FakeClient([RUN_OK])
    collect_live(client, max_runs=10, start_time_from_ms=5, now=NOW)
    assert client.list_kwargs == {"max_runs": 10, "start_time_from_ms": 5}


def test_output_cap_spends_calls_on_the_newest_failure():
    newer = run_with(RUN_FAIL, run_id=999, start_time=at(60), end_time=at(160),
                     tasks=[task(900, "task_failure", "TERMINATED", "FAILED", msg=TASK_MSG)])
    client = FakeClient([RUN_FAIL, newer], {900: {"error": "new error"}})
    col = collect_live(client, max_output_fetches=1, now=NOW)
    assert client.calls == [900]
    assert any("1 task output(s) not fetched" in w for w in col.warnings)


def test_zero_output_cap_makes_no_calls():
    client = FakeClient([RUN_FAIL])
    col = collect_live(client, max_output_fetches=0, now=NOW)
    assert client.calls == [] and any("not fetched" in w for w in col.warnings)
    with pytest.raises(ValueError):
        collect_live(client, max_output_fetches=-1, now=NOW)


def test_output_failure_becomes_warning_and_fallback():
    client = FakeClient([RUN_FAIL], fail={731249776086533})
    col = collect_live(client, now=NOW)
    assert any("not available" in w for w in col.warnings)
    assert col.runs[0].error_message == TASK_MSG


def test_truncation_is_reported():
    col = collect_live(FakeClient([RUN_OK], truncated=True), now=NOW)
    assert col.truncated and any("more exist" in w for w in col.warnings)


def test_naive_now_rejected():
    with pytest.raises(ValueError):
        collect_live(FakeClient([RUN_OK]), now=datetime(2026, 10, 7))


def test_collect_does_not_mutate_inputs_and_strips_user_name():
    client = FakeClient([RUN_OK])
    col = collect_live(client, now=NOW)
    assert "creator_user_name" in RUN_OK
    assert all("creator_user_name" not in r for r in col.raw_runs)


# --- snapshot -----------------------------------------------------------------------------------

def test_snapshot_roundtrip(tmp_path):
    col = collect_live(FakeClient([RUN_OK, RUN_FAIL, RUN_RUNNING], FAIL_OUT, truncated=True), now=NOW)
    path = tmp_path / "live" / "snap.json"
    save_snapshot(col, path)
    assert os.listdir(path.parent) == ["snap.json"]
    assert "someone@example.com" not in path.read_text()
    loaded = load_snapshot(path)
    assert loaded.runs == col.runs
    assert (loaded.fetched_at, loaded.truncated, loaded.warnings) == (NOW, True, col.warnings)


def test_snapshot_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_snapshot(tmp_path / "nope.json")
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    with pytest.raises(ValueError, match="unreadable"):
        load_snapshot(p)
    p.write_text(json.dumps({"version": 99}))
    with pytest.raises(ValueError, match="unsupported"):
        load_snapshot(p)
    p.write_text(json.dumps({"version": 1, "fetched_at": "2026-10-07T04:00:00", "runs": [], "outputs": {}}))
    with pytest.raises(ValueError, match="timezone"):
        load_snapshot(p)
    p.write_text(json.dumps({"version": 1, "fetched_at": NOW.isoformat(), "runs": "x", "outputs": {}}))
    with pytest.raises(ValueError, match="malformed"):
        load_snapshot(p)