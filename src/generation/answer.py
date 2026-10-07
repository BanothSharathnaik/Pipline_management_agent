import dataclasses
from src.generation.llm_client import PROMPT_BUDGET_CHARS, LLM, LLMError
from src.generation.prompt_builder import build_prompt
from src.generation.response_formatter import FinalResponse, check_answer, render_response
from src.ingestion.chunker import Chunk
from src.retrieval.run_context import RunContext
from src.retrieval.vector_index import VectorResult


def _with_direct_error_evidence(ctx: RunContext) -> RunContext:
    """If no evidence was retrieved but run facts contain an explicit error message, synthesize direct exception evidence."""
    rep = ctx.report
    has_evidence = any(rep.run_evidence.values()) or bool(rep.related.results)
    if has_evidence:
        return ctx

    error_runs = [f for f in ctx.facts if f.error_message and f.error_message.strip()]
    if not error_runs:
        return ctx

    new_run_evidence = dict(rep.run_evidence)
    for f in error_runs:
        msg = f.error_message.strip()
        chunk = Chunk(
            chunk_id="FACT-ERR-01",
            doc_id="FACT-ERR-01",
            index=0,
            text=f"Direct Exception Output: {msg}",
            start_char=0,
            end_char=len(msg),
            doc_type="error_log",
            source=f.source,
            origin="telemetry_stdout",
            title="Direct Exception Output",
            run_id=f.run_id,
            job_id=f.job_id,
            timestamp=f.start_time,
        )
        new_run_evidence[f.run_id] = (VectorResult(rank=1, score=1.0, chunk=chunk),)

    new_report = dataclasses.replace(rep, run_evidence=new_run_evidence)
    return dataclasses.replace(ctx, report=new_report)


def generate_answer(
    ctx: RunContext,
    llm: LLM,
    max_chars: int = PROMPT_BUDGET_CHARS,
    fallback_direct_error: bool = False,
) -> FinalResponse:
    """Facts and evidence are always shown. The model is called only when there is something to explain."""
    if fallback_direct_error:
        ctx = _with_direct_error_evidence(ctx)
    rep = ctx.report
    has_evidence = any(rep.run_evidence.values()) or bool(rep.related.results)
    if not has_evidence:  # verified facts alone are not enough to explain anything
        text = render_response(ctx, None, (), None, nothing_to_explain=True)
        return FinalResponse(text, False, None, (), None)

    bundle = build_prompt(ctx, max_chars=max_chars)  # ValueError (prompt too large) is not swallowed
    try:
        response = llm.generate(bundle.system, bundle.user)
    except LLMError as exc:
        message = str(exc)
        return FinalResponse(render_response(ctx, None, (), message), False, None, (), message)

    warnings = check_answer(response.text, bundle, ctx, response)
    return FinalResponse(
        text=render_response(ctx, response.text, warnings, None),
        llm_used=True,
        model_text=response.text,
        warnings=warnings,
        llm_error=None,
    )