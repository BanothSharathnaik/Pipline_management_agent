from datetime import datetime, timedelta, timezone

import pytest

from src.routing.router import (
    COUNT_FAILURES, EXPLAIN, FAILED_LIST, HISTORY, INTENT_NEEDS, LATEST_DETAILS,
    LATEST_FAILED_DIAGNOSIS, LONGEST, REPEATS, RUN_DETAILS, RUN_DIAGNOSIS,
    SLOW_THRESHOLD, SUMMARY, UNROUTED, UNSUPPORTED_DURATION,
    parse_threshold, parse_window, route_question,
)

NOW = datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc)


def utc(s):
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


CASES = [
    ("Why did my latest pipeline run fail?", LATEST_FAILED_DIAGNOSIS),
    ("Which pipelines failed during the last 24 hours?", FAILED_LIST),
    ("What was the error in run r2002?", RUN_DIAGNOSIS),
    ("Have similar errors occurred previously?", HISTORY),
    ("What are the possible causes of this failure?", EXPLAIN),
    ("Which pipelines have unusually long execution times?", UNSUPPORTED_DURATION),
    ("How has pipeline execution duration changed over time?", UNSUPPORTED_DURATION),
    ("Are there recurring errors across multiple pipeline runs?", REPEATS),
    ("Which upstream pipeline might be related to the failure of run r3004?", RUN_DIAGNOSIS),
    ("What happened during run r2002?", RUN_DETAILS),
    ("Which pipelines failed yesterday?", FAILED_LIST),
    ("Why did run 123 fail?", RUN_DIAGNOSIS),
    ("Have we seen this error before?", HISTORY),
    ("Which pipeline had the longest runtime last week?", LONGEST),
    ("Explain why execution times may have increased.", UNSUPPORTED_DURATION),
    ("How many failures occurred in the last 24 hours?", COUNT_FAILURES),
    ("Which runs exceeded 10 minutes?", SLOW_THRESHOLD),
    ("Which runs took longer than 1800 seconds in the last 7 days?", SLOW_THRESHOLD),
    ("Give me a summary of pipeline health", SUMMARY),
    ("Why is it so slow?", EXPLAIN),
    ("how do I bake bread", UNROUTED),
    ("Why did the pipeline fail?", EXPLAIN),
    ("What is the most recent run?", LATEST_DETAILS),
    ("What is the latest run?", LATEST_DETAILS),
    ("How many runs failed in 2026?", COUNT_FAILURES),
    ("compare run r2002 and run r2003", RUN_DETAILS),
]


@pytest.mark.parametrize("question,intent", CASES)
def test_intent(question, intent):
    r = route_question(question, NOW)
    assert r.intent == intent
    assert (r.needs_structured, r.needs_retrieval) == INTENT_NEEDS[intent]
    assert r.matched_rule


def test_flag_table_covers_every_intent_and_unsupported_calls_nothing():
    assert INTENT_NEEDS[UNSUPPORTED_DURATION] == (False, False)
    assert INTENT_NEEDS[RUN_DIAGNOSIS] == (True, True)
    assert INTENT_NEEDS[HISTORY] == (False, True)
    assert INTENT_NEEDS[FAILED_LIST] == (True, False)


def test_run_ids_extracted():
    assert route_question("Why did run r2002 fail?", NOW).run_ids == ("r2002",)
    assert route_question("Why did run 123 fail?", NOW).run_ids == ("123",)
    assert route_question("compare run r2002 and run r2003", NOW).run_ids == ("r2002", "r2003")
    assert route_question("Which pipelines failed yesterday?", NOW).run_ids == ()


def test_windows_attached_to_windowed_intents():
    r = route_question("Which pipelines failed during the last 24 hours?", NOW)
    assert r.window.label == "last 24 hours"
    assert (r.window.start, r.window.end) == (utc("2026-10-04T12:00:00"), NOW)
    r = route_question("Which pipelines failed yesterday?", NOW)
    assert (r.window.start, r.window.end) == (utc("2026-10-04T00:00:00"), utc("2026-10-05T00:00:00"))
    r = route_question("Which pipeline had the longest runtime last week?", NOW)
    assert r.window.start == utc("2026-09-28T12:00:00")
    assert any("time window" in n for n in r.notes)


def test_window_ignored_for_intents_that_do_not_use_it():
    r = route_question("Why did the pipeline fail yesterday?", NOW)
    assert r.intent == EXPLAIN and r.window is None
    r = route_question("Why did run r2002 fail yesterday?", NOW)
    assert r.intent == RUN_DIAGNOSIS and r.window is None


def test_out_of_range_window_is_reported_not_applied():
    r = route_question("Which runs failed in the last 0 hours?", NOW)
    assert r.intent == FAILED_LIST and r.window is None
    assert any("ignored time window" in n for n in r.notes)


def test_threshold_values():
    assert route_question("Which runs exceeded 10 minutes?", NOW).threshold_seconds == 600.0
    r = route_question("Which runs took longer than 1800 seconds in the last 7 days?", NOW)
    assert r.threshold_seconds == 1800.0 and r.window.label == "last 7 days"


def test_notes_for_special_cases():
    assert any("no dependency lookup" in n for n in
               route_question("Which upstream pipeline might be related to the failure of run r3004?", NOW).notes)
    assert any("most recent FAILED run" in n for n in
               route_question("Why did my latest pipeline run fail?", NOW).notes)
    assert any("not implemented" in n for n in
               route_question("How has pipeline execution duration changed over time?", NOW).notes)
    assert any("threshold" in n for n in
               route_question("Which pipelines have unusually long execution times?", NOW).notes)
    assert any("no run ID" in n for n in route_question("Why did the pipeline fail?", NOW).notes)
    assert any("no routing rule" in n for n in route_question("how do I bake bread", NOW).notes)


def test_routing_is_deterministic():
    a = route_question("Which pipelines failed yesterday?", NOW)
    assert a == route_question("Which pipelines failed yesterday?", NOW)


def test_invalid_input_raises():
    with pytest.raises(ValueError):
        route_question("   ", NOW)
    with pytest.raises(ValueError):
        route_question(None, NOW)
    with pytest.raises(ValueError):
        route_question("Which pipelines failed?", datetime(2026, 10, 5, 12, 0))


# --- parse_window / parse_threshold --------------------------------------------------

def test_parse_window_units():
    w, _ = parse_window("last 2 days", NOW)
    assert (w.start, w.end, w.label) == (utc("2026-10-03T12:00:00"), NOW, "last 2 days")
    w, _ = parse_window("failed in the last 24h", NOW)
    assert w.label == "last 24 hours"
    w, _ = parse_window("past 1 week", NOW)
    assert w.label == "last 1 week" and w.start == NOW - timedelta(weeks=1)
    w, _ = parse_window("today", NOW)
    assert (w.start, w.end) == (utc("2026-10-05T00:00:00"), utc("2026-10-06T00:00:00"))


def test_parse_window_no_window():
    assert parse_window("which runs failed", NOW) == (None, None)
    assert parse_window("how many failed in 2026", NOW) == (None, None)


def test_parse_window_uses_utc_calendar_days():
    ist = timezone(timedelta(hours=5, minutes=30))
    now = datetime(2026, 10, 5, 1, 0, tzinfo=ist)   # 2026-10-04 19:30 UTC
    w, _ = parse_window("yesterday", now)
    assert (w.start, w.end) == (utc("2026-10-03T00:00:00"), utc("2026-10-04T00:00:00"))


def test_parse_window_naive_now_raises():
    with pytest.raises(ValueError):
        parse_window("yesterday", datetime(2026, 10, 5))


def test_parse_threshold_units():
    assert parse_threshold("exceeded 30 seconds") == 30.0
    assert parse_threshold("longer than 2 hours") == 7200.0
    assert parse_threshold("over 1.5 minutes") == 90.0
    assert parse_threshold("which runs failed") is None
    assert parse_threshold("exceeded the timeout") is None