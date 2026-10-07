from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Sequence

from src.ingestion.parser import Document
from src.monitoring.models import ResultState, RunRecord
from src.monitoring.queries import FAILURE_STATES
from src.retrieval.conflicts import Conflict, find_conflicts, object_names, relevant_conflicts
from src.retrieval.search import RetrievalPipeline, RetrievalReport, _line, extract_run_ids


@dataclass(frozen=True)
class RunFacts:
    """Verified facts copied from the structured monitoring record. No retrieval involved."""
    run_id: str
    job_id: str
    job_name: Optional[str]
    result_state: str
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    duration_seconds: Optional[float]
    error_message: Optional[str]
    failed_tasks: tuple[str, ...]
    skipped_tasks: tuple[str, ...]
    source: str


@dataclass(frozen=True)
class RunContext:
    report: RetrievalReport
    facts: tuple[RunFacts, ...]
    query_note: Optional[str]
    query_used: str
    conflicts: tuple[Conflict, ...] = ()


def facts_from_run(run: RunRecord) -> RunFacts:
    return RunFacts(
        run_id=run.run_id,
        job_id=run.job_id,
        job_name=run.job_name,
        result_state=run.result_state.value,
        start_time=run.start_time,
        end_time=run.end_time,
        duration_seconds=run.duration_seconds,
        error_message=run.error_message,
        failed_tasks=tuple(t.task_key for t in run.tasks if t.result_state in FAILURE_STATES),
        skipped_tasks=tuple(t.task_key for t in run.tasks if t.result_state == ResultState.SKIPPED),
        source=run.source,
    )


def relation_to_run(ts: Optional[datetime], facts: RunFacts) -> str:
    """Where a piece of evidence sits in time relative to the run. Timestamp comparison only."""
    if ts is None:
        return "undated"
    if facts.start_time is None:
        return "run start unknown"
    if ts < facts.start_time:
        return "before this run"
    if facts.end_time is None:
        return "on or after run start (run end unknown)"
    if ts > facts.end_time:
        return "after this run"
    return "within this run's time window"


def plan_related_query(facts: Sequence[RunFacts]) -> tuple[Optional[str], Optional[str]]:
    """Returns (query text or None to use the question, note explaining the choice)."""
    if not facts:
        return None, None
    if len(facts) > 1:
        return None, "several runs mentioned: similarity search used the question text"
    f = facts[0]
    if f.error_message:
        return f.error_message, (
            f"similarity search used the recorded error message of run {f.run_id} "
            "as the query, not the question"
        )
    return None, (
        f"run {f.run_id} has no recorded error message: similarity search used the "
        "question text, which carries little error information"
    )


def _conflicts_for_runs(conflicts: Sequence[Conflict], facts: Sequence[RunFacts]) -> tuple[Conflict, ...]:
    """With runs in the question, keep only conflicts about objects those runs' errors name."""
    if not facts:
        return tuple(conflicts)
    asked: set[str] = set()
    for f in facts:
        if f.error_message:
            asked |= object_names(f.error_message)
    return tuple(c for c in conflicts if asked & set(c.shared_objects))


def retrieve_for_question(
    pipeline: RetrievalPipeline,
    question: str,
    runs: Sequence[RunRecord],
    top_k: int = 5,
    docs: Optional[Sequence[Document]] = None,
    min_score: Optional[float] = None,
) -> RunContext:
    wanted = set(extract_run_ids(question))
    facts = tuple(facts_from_run(r) for r in runs if r.run_id.lower() in wanted)
    query, note = plan_related_query(facts)
    report = pipeline.retrieve(question, top_k=top_k, related_query=query, min_score=min_score)
    conflicts = (
        _conflicts_for_runs(relevant_conflicts(report, find_conflicts(docs, runs)), facts)
        if docs else ()
    )
    return RunContext(report=report, facts=facts, query_note=note,
                      query_used=query or question, conflicts=conflicts)


def _fmt_time(dt: Optional[datetime]) -> str:
    return "n/a" if dt is None else dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "n/a"
    total = int(seconds)
    return f"{total // 60}m{total % 60:02d}s"


def _facts_block(f: RunFacts) -> list[str]:
    return [
        f"Verified run facts (monitoring data, source={f.source}):",
        f"  run {f.run_id} | job {f.job_name or 'unnamed'} ({f.job_id}) | {f.result_state}",
        f"  started {_fmt_time(f.start_time)} | ended {_fmt_time(f.end_time)} | "
        f"duration {_fmt_duration(f.duration_seconds)}",
        f"  error: {f.error_message or '(none recorded)'}",
        f"  failed tasks: {', '.join(f.failed_tasks) or '(none)'} | "
        f"skipped tasks: {', '.join(f.skipped_tasks) or '(none)'}",
    ]


def format_context(ctx: RunContext) -> str:
    rep = ctx.report
    lines = [f"Question: {rep.question}"]
    for f in ctx.facts:
        lines.extend(_facts_block(f))
    lines.extend(f"NOTE: {n}" for n in rep.notes)
    if ctx.query_note:
        lines.append(f"NOTE: {ctx.query_note}")

    for rid, results in rep.run_evidence.items():
        lines.append(f"\nEvidence linked to run {rid} (by run_id metadata; score is not relevance):")
        if results:
            lines.extend(_line(r) for r in results)
        else:
            lines.append("  (none)")

    lines.append("\nRelated evidence (similarity only; may belong to OTHER runs; not proof):")
    single = ctx.facts[0] if len(ctx.facts) == 1 else None
    if rep.related.results:
        for r in rep.related.results:
            lines.append(_line(r))
            if single is not None:
                lines.append(
                    f"       relation to run {single.run_id}: "
                    f"{relation_to_run(r.chunk.timestamp, single)}"
                )
    else:
        best = rep.related.best_score
        detail = f" (best match scored {best:.3f})" if best is not None else ""
        lines.append(f"  no evidence above min score {rep.related.min_score:.2f}{detail}")

    if ctx.conflicts:
        lines.append("\nPOSSIBLE CONFLICTS (rule-based check of documents against monitoring data):")
        for c in ctx.conflicts:
            lines.append(
                f"  {c.doc_id} (file {c.origin}) claims resolved at {_fmt_time(c.claim_time)}, "
                f"but run {c.run_id} started {_fmt_time(c.run_start)}, ended in a failure state, "
                f"and its error names {', '.join(c.shared_objects)}."
            )
        lines.append(
            "  This does not prove the fix failed (the cause may differ). "
            "It means the claim is not confirmed by later runs."
        )
    return "\n".join(lines)