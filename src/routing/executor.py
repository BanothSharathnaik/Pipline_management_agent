from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional, Sequence

from src.generation.answer import generate_answer
from src.generation.llm_client import LLM, LLMError
from src.ingestion.parser import Document
from src.monitoring.formatting import fmt_run, fmt_time, run_detail_lines
from src.monitoring.models import RunRecord
from src.monitoring.queries import (
    count_failures,
    failed_runs,
    latest_run,
    repeated_errors,
    runs_exceeding,
    runs_in_window,
    status_counts,
)
from src.retrieval.run_context import RunContext, format_context, retrieve_for_question
from src.retrieval.search import RetrievalPipeline
from src.routing.router import (
    COUNT_FAILURES, EXPLAIN, FAILED_LIST, HISTORY, LATEST_DETAILS, LATEST_FAILED_DIAGNOSIS,
    LONGEST, REPEATS, RUN_DETAILS, RUN_DIAGNOSIS, SLOW_THRESHOLD, SUMMARY,
    UNROUTED, UNSUPPORTED_DURATION, Route, route_question,
)


class DisabledLLM:
    """Stands in for the model when the user passes --no-llm."""
    model = "disabled"

    def generate(self, system: str, user: str):
        raise LLMError("disabled with --no-llm")


@dataclass(frozen=True)
class Answer:
    text: str
    intent: str
    llm_used: bool
    warnings: tuple[str, ...]
    resolved_run_id: Optional[str]


def _banner(runs: Sequence[RunRecord]) -> str:
    sources = sorted({r.source for r in runs}) or ["no"]
    return f"[{'/'.join(s.upper() for s in sources)} DATA] {len(runs)} runs loaded"


def _bounds(route: Route):
    return (route.window.start, route.window.end) if route.window else (None, None)


def _label(route: Route) -> str:
    return route.window.label if route.window else "all loaded runs"


def _coverage(runs: Sequence[RunRecord]) -> str:
    starts = [r.start_time for r in runs if r.start_time is not None]
    if not starts:
        return "data coverage: no runs with start times"
    return (f"data coverage: {len(runs)} runs, starts from {fmt_time(min(starts))} "
            f"to {fmt_time(max(starts))}")


def _find(runs: Sequence[RunRecord], run_id: str) -> list[RunRecord]:
    return [r for r in runs if r.run_id.lower() == run_id.lower()]


def _latest_failed(runs: Sequence[RunRecord]) -> Optional[RunRecord]:
    candidates = [r for r in failed_runs(runs) if r.start_time is not None]
    return candidates[-1] if candidates else None  # failed_runs is oldest first


# --- structured handlers: return lines, never touch the model or the embedder -------------

def _failed_list(route: Route, runs) -> list[str]:
    start, end = _bounds(route)
    hits = failed_runs(runs, start, end)
    lines = ["Failed runs (FAILED or TIMED_OUT), oldest first:"]
    lines += [fmt_run(r) for r in hits] if hits else [f"No failed runs in {_label(route)}."]
    return lines + [_coverage(runs)]


def _count(route: Route, runs) -> list[str]:
    start, end = _bounds(route)
    n = count_failures(runs, start, end)
    return [f"Failures: {n} (FAILED or TIMED_OUT) in {_label(route)}", _coverage(runs)]


def _longest(route: Route, runs) -> list[str]:
    start, end = _bounds(route)
    ranked = runs_exceeding(runs_in_window(runs, start, end), 0.0)[:3]
    if not ranked:
        return [f"No runs with a known duration in {_label(route)}.", _coverage(runs)]
    return (["Longest runs (known durations only; runs without an end time are excluded):"]
            + [fmt_run(r) for r in ranked] + [_coverage(runs)])


def _slow(route: Route, runs) -> list[str]:
    start, end = _bounds(route)
    thr = route.threshold_seconds or 0.0
    hits = runs_exceeding(runs_in_window(runs, start, end), thr)
    if not hits:
        return [f"No runs exceed {thr:g} seconds in {_label(route)}.", _coverage(runs)]
    return ([f"Runs longer than {thr:g} seconds, longest first:"]
            + [fmt_run(r) for r in hits] + [_coverage(runs)])


def _summary(route: Route, runs) -> list[str]:
    start, end = _bounds(route)
    selected = runs_in_window(runs, start, end)
    counts = status_counts(selected)
    lines = [f"Run counts by result state in {_label(route)}:"]
    lines += [f"  {s.value:<10} {n}" for s, n in sorted(counts.items(), key=lambda kv: kv[0].value)]
    return lines + [f"Total: {len(selected)} runs", _coverage(runs)]


def _repeats(route: Route, runs) -> list[str]:
    groups = repeated_errors(runs)
    if not groups:
        return ["No exactly repeated error messages."]
    return [f"{g.count}x  runs={list(g.run_ids)}  {g.message}" for g in groups]


def _run_details(route: Route, runs) -> list[str]:
    lines: list[str] = []
    for rid in route.run_ids:
        found = _find(runs, rid)
        if not found:
            lines.append(f"No run found with id {rid}.")
        for r in found:
            lines += run_detail_lines(r) + [""]
    return lines


def _latest_details(route: Route, runs) -> list[str]:
    r = latest_run(runs)
    return run_detail_lines(r) if r else ["No run with a start time was found."]


def _unsupported(route: Route, runs) -> list[str]:
    return ["This question cannot be answered yet; see the notes above."]


STRUCTURED = {
    FAILED_LIST: _failed_list, COUNT_FAILURES: _count, LONGEST: _longest,
    SLOW_THRESHOLD: _slow, SUMMARY: _summary, REPEATS: _repeats,
    RUN_DETAILS: _run_details, LATEST_DETAILS: _latest_details,
    UNSUPPORTED_DURATION: _unsupported,
}


def _verified_records(ctx: RunContext, runs: Sequence[RunRecord]) -> list[str]:
    results = ctx.report.related.results
    cited: list[str] = []
    for h in results:
        rid = h.chunk.run_id
        if rid and rid.lower() not in {c.lower() for c in cited}:
            cited.append(rid)
    lines = [""]
    if cited:
        lines.append("Verified monitoring records for runs cited by the evidence:")
        for rid in cited:
            found = _find(runs, rid)
            lines.append("  " + (fmt_run(found[0]) if found else f"{rid}: not found in monitoring data"))
    if any(not h.chunk.run_id for h in results):
        lines.append("Documents without a run link have no monitoring record to verify.")
    if not results:
        lines.append("No evidence to verify.")
    return lines


def answer_question(
    question: str,
    now: datetime,
    runs: Sequence[RunRecord],
    get_pipeline: Callable[[], RetrievalPipeline],
    docs: Sequence[Document],
    llm: LLM,
    top_k: int = 5,
    data_note: Optional[str] = None,
) -> Answer:
    route = route_question(question, now)
    head = [_banner(runs)]
    if data_note:
        head.append(f"Data note: {data_note}")
    if any(r.source == "live" for r in runs) and any(d.source == "mock" for d in docs):
        head.append("NOTE: the diagnostic documents are source=mock sample documents, not written about "
                    "these live runs; read any match as general reference only")
    head += [
        f"Question: {question}",
        f"Route: {route.intent} (rule: {route.matched_rule}) | "
        f"structured={route.needs_structured} retrieval={route.needs_retrieval}",
    ] + [f"NOTE: {n}" for n in route.notes]

    llm_used, warnings, resolved = False, (), None
    intent = route.intent

    if intent in STRUCTURED:
        body = "\n".join(STRUCTURED[intent](route, runs))

    elif intent == RUN_DIAGNOSIS or intent == EXPLAIN:
        ctx = retrieve_for_question(get_pipeline(), question, runs, top_k=top_k, docs=docs)
        final = generate_answer(ctx, llm)
        body, llm_used, warnings = final.text, final.llm_used, final.warnings

    elif intent == LATEST_FAILED_DIAGNOSIS:
        target = _latest_failed(runs)
        if target is None:
            body = "No failed run was found in the monitoring data."
        else:
            resolved = target.run_id
            overall = latest_run(runs)
            lines = [f"Resolved 'latest failed run' to {target.run_id}."]
            if overall is not None and overall.run_id != target.run_id:
                lines.append(
                    f"The most recent run overall is {overall.run_id} "
                    f"(lifecycle {overall.lifecycle_state or 'n/a'}, result {overall.result_state.value}); "
                    "it is not a failure, so it is not diagnosed."
                )
            ctx = retrieve_for_question(get_pipeline(), f"Why did run {target.run_id} fail?",
                                        runs, top_k=top_k, docs=docs)
            final = generate_answer(ctx, llm)
            body = "\n".join(lines) + "\n\n" + final.text
            llm_used, warnings = final.llm_used, final.warnings

    else:  # HISTORY, UNROUTED: evidence plus verified records, never the model
        ctx = retrieve_for_question(get_pipeline(), question, runs, top_k=top_k, docs=docs)
        body = format_context(ctx) + "\n".join(_verified_records(ctx, runs))
        if intent == HISTORY:
            body += ("\nThe question does not say which error it means, so the closest evidence to the "
                     "question text is shown. Name a run (e.g. 'run r2002') to search with that run's error.")

    return Answer("\n".join(head) + "\n\n" + body, intent, llm_used, tuple(warnings), resolved)