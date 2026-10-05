from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.monitoring.collector import load_mock_runs
from src.monitoring.models import ResultState, RunRecord
from src.monitoring.queries import (
    count_failures,
    failed_in_last,
    failed_runs,
    find_runs,
    latest_run,
    repeated_errors,
    runs_exceeding,
    runs_in_window,
    status_counts,
)

MOCK = Path(__file__).resolve().parents[1] / "data" / "mock_runs.json"
RUNS = load_mock_runs(MOCK).runs


def t(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def ids(runs):
    return [r.run_id for r in runs]


def mk(run_id, state, start=None, end=None, error=None, job="j"):
    return RunRecord.from_dict({
        "run_id": run_id, "job_id": job, "source": "mock",
        "result_state": state, "start_time": start, "end_time": end,
        "error_message": error,
    })


# --- latest / find ---------------------------------------------------------

def test_latest_run_overall():
    assert latest_run(RUNS).run_id == "r1005"


def test_latest_run_per_job():
    assert latest_run(RUNS, "j_cust").run_id == "r2005"
    assert latest_run(RUNS, "j_sales").run_id == "r3005"  # r3006 has no start time
    assert latest_run(RUNS, "j_orders").run_id == "r1005"


def test_latest_run_unknown_job_and_empty():
    assert latest_run(RUNS, "nope") is None
    assert latest_run([]) is None


def test_find_runs():
    assert ids(find_runs(RUNS, "r2001")) == ["r2001"]
    assert find_runs(RUNS, "missing") == []


# --- failures --------------------------------------------------------------

def test_failed_runs_default_includes_timeouts_oldest_first():
    assert ids(failed_runs(RUNS)) == ["r2002", "r3002", "r2003", "r1004", "r3004", "r2005"]


def test_failed_runs_failed_only():
    got = failed_runs(RUNS, states=frozenset({ResultState.FAILED}))
    assert set(ids(got)) == {"r1004", "r2002", "r2003", "r2005", "r3004"}


def test_failed_runs_job_filter():
    assert ids(failed_runs(RUNS, job_id="j_cust")) == ["r2002", "r2003", "r2005"]


def test_count_failures_oct4():
    n = count_failures(RUNS, t("2026-10-04T00:00:00"), t("2026-10-05T00:00:00"))
    assert n == 3  # r1004, r3004, r2005


def test_window_is_half_open():
    got = runs_in_window(RUNS, t("2026-10-04T01:00:00"), t("2026-10-04T20:00:00"))
    assert "r1004" in ids(got)      # starts exactly at window start -> included
    assert "r2005" not in ids(got)  # starts exactly at window end -> excluded


def test_timeout_inside_oct2_window():
    got = failed_runs(RUNS, t("2026-10-02T00:00:00"), t("2026-10-03T00:00:00"))
    assert ids(got) == ["r2002", "r3002"]


def test_failed_in_last_24h():
    got = failed_in_last(RUNS, t("2026-10-05T12:00:00"), 24)
    assert ids(got) == ["r2005"]


def test_naive_datetime_rejected():
    with pytest.raises(ValueError):
        runs_in_window(RUNS, datetime(2026, 10, 4))
    with pytest.raises(ValueError):
        failed_in_last(RUNS, datetime(2026, 10, 5), 24)


def test_start_after_end_rejected():
    with pytest.raises(ValueError):
        runs_in_window(RUNS, t("2026-10-05T00:00:00"), t("2026-10-04T00:00:00"))


def test_run_without_start_excluded_from_window_only():
    runs = [mk("x1", "FAILED")]  # no start time
    assert ids(failed_runs(runs)) == ["x1"]
    assert failed_runs(runs, start=t("2026-10-01T00:00:00")) == []


# --- durations -------------------------------------------------------------

def test_runs_exceeding_1000s():
    assert ids(runs_exceeding(RUNS, 1000)) == ["r3002", "r1003"]  # 1800s, 1300s


def test_exceeding_is_strictly_greater():
    assert ids(runs_exceeding(RUNS, 1300)) == ["r3002"]


def test_exceeding_edge_cases():
    assert runs_exceeding(RUNS, 5000) == []
    with pytest.raises(ValueError):
        runs_exceeding(RUNS, -1)


# --- counts and repeats ----------------------------------------------------

def test_status_counts():
    c = status_counts(RUNS)
    assert c[ResultState.SUCCESS] == 7
    assert c[ResultState.FAILED] == 5
    assert c[ResultState.TIMED_OUT] == 1
    assert c[ResultState.CANCELLED] == 1
    assert c[ResultState.SKIPPED] == 1
    assert c[ResultState.UNKNOWN] == 1
    assert sum(c.values()) == 16


def test_no_exact_repeats_in_mock_data():
    # r2002/r2003 are similar but worded differently: exact matching must NOT group them.
    assert repeated_errors(RUNS) == []


def test_repeated_errors_groups_ignoring_case_and_whitespace():
    runs = [
        mk("a", "FAILED", "2026-10-01T00:00:00+00:00", error="Disk  FULL on node 1"),
        mk("b", "FAILED", "2026-10-02T00:00:00+00:00", error="disk full on node 1"),
        mk("c", "FAILED", "2026-10-03T00:00:00+00:00", error="something else"),
    ]
    groups = repeated_errors(runs)
    assert len(groups) == 1
    assert groups[0].run_ids == ("a", "b")
    assert groups[0].count == 2