from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Sequence

from src.generation.answer import generate_answer
from src.generation.llm_client import LLM
from src.ingestion.parser import Document
from src.monitoring.formatting import fmt_duration, fmt_time
from src.monitoring.models import ResultState, RunRecord
from src.monitoring.queries import FAILURE_STATES
from src.retrieval.run_context import RunContext, retrieve_for_question
from src.retrieval.search import RetrievalPipeline


@dataclass(frozen=True)
class IncidentReport:
    run_id: str
    job_id: str
    job_name: Optional[str]
    result_state: str
    text: str
    has_prior_occurrences: bool
    evidence_count: int
    llm_used: bool = False


def build_incident_report(
    target_run: RunRecord,
    all_runs: Sequence[RunRecord],
    pipeline: Optional[RetrievalPipeline] = None,
    docs: Optional[Sequence[Document]] = None,
    llm: Optional[LLM] = None,
    now: Optional[datetime] = None,
    data_note: Optional[str] = None,
    top_k: int = 5,
) -> IncidentReport:
    """Builds an incident report for a run with deterministic facts and optional unverified LLM hypotheses."""
    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        raise ValueError("now must be timezone-aware")

    lines: list[str] = []

    # 1. Header
    lines.append(f"# INCIDENT REPORT: Run {target_run.run_id}")
    lines.append("")
    lines.append("## 1. Incident Header")
    lines.append(f"- Incident / Run ID:   {target_run.run_id}")
    lines.append(f"- Job ID:               {target_run.job_id}")
    lines.append(f"- Job Name:             {target_run.job_name or 'unnamed'}")
    lines.append(f"- Data Source:          {target_run.source}")
    lines.append(f"- Report Generated At:  {fmt_time(current_time)}")
    if data_note:
        lines.append(f"- Data Freshness:       {data_note}")
    if target_run.source == "live" and docs and any(d.source == "mock" for d in docs):
        lines.append(
            "- Diagnostic Corpus:    NOTE: diagnostic documents are sample mock references, "
            "not specifically authored for this live run"
        )

    # 2. Executive Summary
    lines.append("")
    lines.append("## 2. Executive Summary")
    lines.append(f"- Overall Result:       {target_run.result_state.value}")
    lines.append(f"- Lifecycle State:      {target_run.lifecycle_state or 'n/a'}")
    lines.append(f"- Start Time:           {fmt_time(target_run.start_time)}")
    lines.append(f"- End Time:             {fmt_time(target_run.end_time)}")
    lines.append(f"- Total Duration:       {fmt_duration(target_run.duration_seconds)}")
    if target_run.queue_seconds is not None:
        lines.append(
            f"- Queue Duration:       {fmt_duration(target_run.queue_seconds)} "
            "(may be included in duration)"
        )

    # 3. Impact Assessment
    lines.append("")
    lines.append("## 3. Impact Assessment")
    failed_tasks = [t for t in target_run.tasks if t.result_state in FAILURE_STATES]
    skipped_tasks = [t for t in target_run.tasks if t.result_state == ResultState.SKIPPED]
    success_tasks = [t for t in target_run.tasks if t.result_state == ResultState.SUCCESS]
    other_tasks = [
        t for t in target_run.tasks
        if t not in failed_tasks and t not in skipped_tasks and t not in success_tasks
    ]
    retried_tasks = [t for t in target_run.tasks if t.attempts > 1]

    if not target_run.tasks:
        lines.append("- Tasks: No individual tasks recorded (single-workload run).")
    else:
        lines.append(f"- Total Tasks:          {len(target_run.tasks)}")
        lines.append(f"- Successful Tasks:     {len(success_tasks)}")
        lines.append(f"- Failed Tasks:         {len(failed_tasks)}")
        for t in failed_tasks:
            retry_info = f" ({t.attempts} attempts)" if t.attempts > 1 else ""
            lines.append(f"    * {t.task_key}{retry_info}")
        lines.append(f"- Skipped / Blocked:    {len(skipped_tasks)}")
        for t in skipped_tasks:
            deps = f" [depends on: {', '.join(t.depends_on)}]" if t.depends_on else ""
            lines.append(f"    * {t.task_key}{deps}")
        if other_tasks:
            lines.append(f"- Other Task States:    {len(other_tasks)}")
            for t in other_tasks:
                lines.append(f"    * {t.task_key} ({t.result_state.value})")
        if retried_tasks:
            lines.append(
                f"- Tasks with Retries:   {len(retried_tasks)} "
                f"({', '.join(f'{t.task_key} [{t.attempts}x]' for t in retried_tasks)})"
            )

    # 4. Execution Timeline
    lines.append("")
    lines.append("## 4. Execution Timeline")
    timeline_events: list[str] = []
    if target_run.start_time is not None:
        timeline_events.append(f"- {fmt_time(target_run.start_time)}: Run started")
    else:
        timeline_events.append("- Time unknown: Run started")

    # Sort tasks by start_time if available
    sorted_tasks = sorted(
        target_run.tasks,
        key=lambda t: (t.start_time is None, t.start_time or datetime.min.replace(tzinfo=timezone.utc)),
    )
    for t in sorted_tasks:
        t_start = fmt_time(t.start_time) if t.start_time else "Time unknown"
        t_end = fmt_time(t.end_time) if t.end_time else "Time unknown"
        dur = f"duration {fmt_duration(t.duration_seconds)}" if t.duration_seconds is not None else "duration unknown"
        attempts_note = f", {t.attempts} attempts" if t.attempts > 1 else ""
        timeline_events.append(
            f"- {t_start} -> {t_end}: Task '{t.task_key}' finished with state {t.result_state.value} "
            f"({dur}{attempts_note})"
        )

    if target_run.end_time is not None:
        timeline_events.append(
            f"- {fmt_time(target_run.end_time)}: Run terminated with state {target_run.result_state.value}"
        )
    else:
        timeline_events.append(
            f"- Run lifecycle state is '{target_run.lifecycle_state or 'unknown'}' (end time not set)"
        )
    lines.extend(timeline_events)

    # 5. Verbatim Error Message
    lines.append("")
    lines.append("## 5. Verbatim Error Message")
    if target_run.error_message:
        lines.append("```text")
        lines.append(target_run.error_message.strip())
        lines.append("```")
    else:
        lines.append("No error message recorded in run or task details.")

    # 6. Diagnostic Evidence
    lines.append("")
    lines.append("## 6. Diagnostic Evidence")
    evidence_count = 0
    ctx: Optional[RunContext] = None
    if pipeline is not None:
        query_text = f"Why did run {target_run.run_id} fail?"
        ctx = retrieve_for_question(pipeline, query_text, all_runs, top_k=top_k, docs=docs)
        linked = ctx.report.run_evidence.get(target_run.run_id, [])
        related = ctx.report.related.results

        evidence_count = len(linked) + len(related)
        if linked:
            lines.append("### Linked Evidence (explicitly tagged with this run ID):")
            for h in linked:
                c = h.chunk
                title = c.title or c.origin
                lines.append(f"- [{c.doc_id}] {title} (origin: {c.origin}):")
                lines.append(f"  {c.text.strip()}")
        else:
            lines.append("No documents explicitly linked to this run ID.")

        if related:
            lines.append("")
            lines.append("### Related Reference Documents (retrieved via semantic similarity):")
            for idx, h in enumerate(related, start=1):
                c = h.chunk
                title = c.title or c.origin
                lines.append(f"{idx}. [{c.doc_id}] {title} (score: {h.score:.3f}, origin: {c.origin}):")
                lines.append(f"   {c.text.strip()}")
        else:
            lines.append("No related diagnostic reference documents found above threshold.")
    else:
        lines.append("No retrieval pipeline provided; diagnostic document matching skipped.")

    # 7. Earlier Occurrences
    lines.append("")
    lines.append("## 7. Earlier Occurrences & Recurrence")
    prior_same_job: list[RunRecord] = []
    prior_same_error: list[RunRecord] = []

    for r in all_runs:
        if r.run_id == target_run.run_id:
            continue
        if r.job_id == target_run.job_id and r.result_state in FAILURE_STATES:
            prior_same_job.append(r)
        if (
            target_run.error_message
            and r.error_message
            and r.error_message.strip() == target_run.error_message.strip()
        ):
            prior_same_error.append(r)

    has_prior_occurrences = bool(prior_same_job or prior_same_error)

    if prior_same_error:
        lines.append(f"- Identical Error Repeated: {len(prior_same_error)} other run(s) had this exact error message:")
        for r in prior_same_error:
            lines.append(f"    * Run {r.run_id} (job {r.job_id}) at {fmt_time(r.start_time)}")
    else:
        lines.append("- Identical Error Repeated: None (this error message is unique in the loaded data).")

    if prior_same_job:
        lines.append(f"- Prior Failures for Job '{target_run.job_id}': {len(prior_same_job)} previous failure(s):")
        for r in prior_same_job[-5:]:  # show up to 5
            lines.append(f"    * Run {r.run_id} at {fmt_time(r.start_time)} (state: {r.result_state.value})")
    else:
        lines.append(f"- Prior Failures for Job '{target_run.job_id}': None recorded in loaded data.")

    # 8. Unconfirmed Claims & Unknowns
    lines.append("")
    lines.append("## 8. Unconfirmed Claims & Unknowns")
    lines.append("- Root cause is UNCONFIRMED: The monitoring record captures symptoms and error outputs,")
    lines.append("  not code or infrastructure root cause verification.")
    if target_run.source == "live":
        lines.append("- Upstream data quality and network conditions at run time have not been verified.")
    lines.append("- Full stack trace / cluster log inspectability depends on Databricks retention policy.")

    # 9. Limits & Scope Bounds
    lines.append("")
    lines.append("## 9. Limitations & Scope Bounds")
    lines.append("- The factual sections (1-8) were generated deterministically from structured records.")
    if llm is not None:
        lines.append("- Section 10 below was synthesized by a language model as hypotheses and is UNVERIFIED.")
    else:
        lines.append("- No generative language model was used in this report; all facts and text are literal.")
    lines.append(f"- Total historical runs analyzed: {len(all_runs)}.")
    if target_run.source == "live":
        lines.append("- Data reflects the state at snapshot collection time, not real-time stream.")

    # 10. Potential Root Causes & Hypotheses (Generated by LLM — UNVERIFIED)
    llm_used = False
    if llm is not None:
        lines.append("")
        lines.append("## 10. Potential Root Causes & Hypotheses (Language Model — UNVERIFIED)")
        lines.append(
            "> [!WARNING]\n"
            "> UNVERIFIED HYPOTHESES: The explanations below were generated by a local language model "
            f"({getattr(llm, 'model', 'Ollama')}) using retrieved reference documents.\n"
            "> They are hypotheses to guide investigation, NOT confirmed causes. The facts above are authoritative.\n"
        )
        if ctx is not None:
            try:
                llm_response = generate_answer(ctx, llm)
                if llm_response.llm_used and llm_response.model_text:
                    llm_used = True
                    lines.append(llm_response.model_text.strip())
                    if llm_response.warnings:
                        lines.append("\n### Model Output Checks:")
                        for w in llm_response.warnings:
                            lines.append(f"- WARNING: {w}")
                else:
                    lines.append(
                        "No explanation generated: verified facts alone or available evidence "
                        "were insufficient to establish an explanation."
                    )
            except Exception as exc:
                lines.append(f"Language model generation failed: {exc}")
        else:
            lines.append("Language model could not be called because no retrieval context was available.")

    report_text = "\n".join(lines)
    return IncidentReport(
        run_id=target_run.run_id,
        job_id=target_run.job_id,
        job_name=target_run.job_name,
        result_state=target_run.result_state.value,
        text=report_text,
        has_prior_occurrences=has_prior_occurrences,
        evidence_count=evidence_count,
        llm_used=llm_used,
    )
