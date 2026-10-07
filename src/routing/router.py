from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from src.retrieval.search import extract_run_ids

RUN_DIAGNOSIS = "run_diagnosis"
RUN_DETAILS = "run_details"
LATEST_FAILED_DIAGNOSIS = "latest_failed_diagnosis"
LATEST_DETAILS = "latest_details"
FAILED_LIST = "failed_list"
COUNT_FAILURES = "count_failures"
LONGEST = "longest"
SLOW_THRESHOLD = "slow_threshold"
REPEATS = "repeats"
SUMMARY = "summary"
HISTORY = "history"
EXPLAIN = "explain"
UNSUPPORTED_DURATION = "unsupported_duration"
UNROUTED = "unrouted"

# intent -> (needs_structured, needs_retrieval)
INTENT_NEEDS: dict[str, tuple[bool, bool]] = {
    RUN_DIAGNOSIS: (True, True),
    RUN_DETAILS: (True, False),
    LATEST_FAILED_DIAGNOSIS: (True, True),
    LATEST_DETAILS: (True, False),
    FAILED_LIST: (True, False),
    COUNT_FAILURES: (True, False),
    LONGEST: (True, False),
    SLOW_THRESHOLD: (True, False),
    REPEATS: (True, False),
    SUMMARY: (True, False),
    HISTORY: (False, True),
    EXPLAIN: (False, True),
    UNSUPPORTED_DURATION: (False, False),
    UNROUTED: (False, True),
}

_WORD = re.compile(r"[a-z0-9']+")
FAILURE_WORDS = frozenset({"fail", "failed", "fails", "failing", "failure", "failures"})
DIAG_WORDS = FAILURE_WORDS | {
    "why", "cause", "causes", "caused", "reason", "reasons", "explain", "fix",
    "investigate", "error", "errors", "exception", "wrong", "broke", "broken",
}
EXPLAIN_WORDS = frozenset({"why", "cause", "causes", "caused", "reason", "reasons",
                           "explain", "fix", "investigate", "diagnose"})
REPEAT_WORDS = frozenset({"recurring", "repeated", "repeatedly", "repeat", "repeats", "recurrent"})
HISTORY_WORDS = frozenset({"seen", "before", "previously", "earlier", "again", "similar"})
ERRORISH = frozenset({"error", "errors", "failure", "failures", "failed", "fail", "issue",
                      "issues", "exception", "exceptions", "problem", "problems"})
DURATION_WORDS = frozenset({"slow", "slower", "slowest", "longest", "duration", "durations",
                            "runtime", "runtimes", "exceed", "exceeds", "exceeded", "unusually", "took"})
TREND_WORDS = frozenset({"changed", "change", "trend", "increase", "increased", "increasing",
                         "growing", "degraded"})
SUMMARY_WORDS = frozenset({"summary", "overview", "health", "status", "statuses"})
DEPENDENCY_WORDS = frozenset({"upstream", "downstream", "dependency", "dependencies", "depends"})

_LATEST = re.compile(r"\b(?:latest|most recent|last)\b(?:\s+\w+){0,2}?\s+run\b")
_NUM_WINDOW = re.compile(r"\b(?:last|past)\s+(\d+)\s*(hours?|hrs?|h|days?|d|weeks?|w)\b")
_WEEK = re.compile(r"\b(?:last|past)\s+week\b")
_THRESHOLD = re.compile(
    r"\b(?:exceed(?:s|ed|ing)?|longer than|more than|over|above|greater than|beyond)\s+"
    r"(\d+(?:\.\d+)?)\s*(seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h)\b"
)
_UNITS = {"h": ("hour", timedelta(hours=1)), "d": ("day", timedelta(days=1)),
          "w": ("week", timedelta(weeks=1))}
_SECONDS = {"s": 1.0, "m": 60.0, "h": 3600.0}


@dataclass(frozen=True)
class TimeWindow:
    start: datetime   # inclusive, UTC
    end: datetime     # exclusive, UTC
    label: str


@dataclass(frozen=True)
class Route:
    intent: str
    needs_structured: bool
    needs_retrieval: bool
    matched_rule: str
    run_ids: tuple[str, ...] = ()
    window: Optional[TimeWindow] = None
    threshold_seconds: Optional[float] = None
    notes: tuple[str, ...] = ()


def parse_window(text: str, now: datetime) -> tuple[Optional[TimeWindow], Optional[str]]:
    """Time window named in the text, relative to `now`. Returns (window, note about an ignored window)."""
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    q = text.lower()
    now_utc = now.astimezone(timezone.utc)

    m = _NUM_WINDOW.search(q)
    if m:
        n = int(m.group(1))
        name, unit = _UNITS[m.group(2)[0]]
        if n < 1 or n > 3650:
            return None, f"ignored time window 'last {n} {name}s': out of range"
        label = f"last {n} {name}{'s' if n != 1 else ''}"
        return TimeWindow(now_utc - n * unit, now_utc, label), None
    if _WEEK.search(q):
        return TimeWindow(now_utc - timedelta(weeks=1), now_utc, "last week (rolling 7 days)"), None

    midnight = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    if re.search(r"\byesterday\b", q):
        return TimeWindow(midnight - timedelta(days=1), midnight, "yesterday (UTC)"), None
    if re.search(r"\btoday\b", q):
        return TimeWindow(midnight, midnight + timedelta(days=1), "today (UTC)"), None
    return None, None


def parse_threshold(text: str) -> Optional[float]:
    """Duration threshold in seconds, e.g. 'exceeded 10 minutes' -> 600.0."""
    m = _THRESHOLD.search(text.lower())
    if not m:
        return None
    return float(m.group(1)) * _SECONDS[m.group(2)[0]]


def _fmt(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M")


def route_question(question: str, now: datetime) -> Route:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")

    q = question.lower()
    words = set(_WORD.findall(q))
    run_ids = tuple(extract_run_ids(question))
    window, window_note = parse_window(q, now)
    threshold = parse_threshold(q)
    base_notes = [window_note] if window_note else []

    def make(intent: str, rule: str, use_window: bool = False, ids: tuple[str, ...] = (),
             thr: Optional[float] = None, notes: tuple[str, ...] = ()) -> Route:
        used = window if use_window else None
        out = list(base_notes)
        if used is not None:
            out.append(f"time window: {used.label} ({_fmt(used.start)} to {_fmt(used.end)} UTC, "
                       "start inclusive, end exclusive)")
        out.extend(notes)
        structured, retrieval = INTENT_NEEDS[intent]
        return Route(intent, structured, retrieval, rule, run_ids=ids, window=used,
                     threshold_seconds=thr, notes=tuple(out))

    # R1: a run ID in the question
    if run_ids:
        if words & DIAG_WORDS or words & HISTORY_WORDS:
            extra = ()
            if words & DEPENDENCY_WORDS:
                extra = ("no dependency lookup between runs exists yet: related evidence is "
                         "similarity only and does not establish an upstream link",)
            return make(RUN_DIAGNOSIS, "R1 run id + diagnostic words", ids=run_ids, notes=extra)
        return make(RUN_DETAILS, "R1 run id", ids=run_ids)

    # R2: "latest run"
    if _LATEST.search(q):
        if words & DIAG_WORDS:
            return make(LATEST_FAILED_DIAGNOSIS, "R2 latest run + diagnostic words",
                        notes=("'latest' means the most recent FAILED run, not the most recent run overall",))
        return make(LATEST_DETAILS, "R2 latest run")

    # R3: recurring errors
    if words & REPEAT_WORDS:
        return make(REPEATS, "R3 recurring errors",
                    notes=("only exactly repeated error messages are grouped; similar wording is "
                           "found by asking 'have we seen this error before?'",))

    # R4: history of similar errors
    if words & HISTORY_WORDS and words & ERRORISH:
        return make(HISTORY, "R4 earlier occurrences")

    # R5: durations
    is_duration = bool(words & DURATION_WORDS) or "execution time" in q or "how long" in q
    if threshold is not None:
        return make(SLOW_THRESHOLD, "R5 duration threshold", use_window=True, thr=threshold)
    if is_duration:
        if words & {"longest", "slowest"}:
            return make(LONGEST, "R5 longest run", use_window=True)
        if words & TREND_WORDS or "over time" in q:
            return make(UNSUPPORTED_DURATION, "R5 duration trend",
                        notes=("duration trends over time are not implemented yet; ask for the longest "
                               "run, or give a threshold such as 'exceeded 20 minutes'",))
        if "why" not in words:
            return make(UNSUPPORTED_DURATION, "R5 duration without a threshold",
                        notes=("no threshold given: say e.g. 'exceeded 20 minutes', or ask for the longest run",))
        # "why ... slow" falls through to the explain rule

    # R6: how many failures
    if words & FAILURE_WORDS and ("how many" in q or "number of" in q or "count" in words):
        return make(COUNT_FAILURES, "R6 failure count", use_window=True)

    # R7: explain without a run ID
    if words & EXPLAIN_WORDS:
        return make(EXPLAIN, "R7 explain, no run id",
                    notes=("no run ID in the question: answering from retrieved evidence only; "
                           "name a run (e.g. 'run r2002') for run-specific facts",))

    # R8: which runs failed
    if words & FAILURE_WORDS:
        return make(FAILED_LIST, "R8 failed runs", use_window=True)

    # R9: summary
    if words & SUMMARY_WORDS:
        return make(SUMMARY, "R9 summary", use_window=True)

    return make(UNROUTED, "R10 no rule matched",
                notes=("no routing rule matched: searching the diagnostic evidence only",))