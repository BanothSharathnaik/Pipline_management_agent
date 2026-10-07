from __future__ import annotations

from src.generation.llm_client import PROMPT_BUDGET_CHARS, LLM, LLMError
from src.generation.prompt_builder import build_prompt
from src.generation.response_formatter import FinalResponse, check_answer, render_response
from src.retrieval.run_context import RunContext


def generate_answer(ctx: RunContext, llm: LLM, max_chars: int = PROMPT_BUDGET_CHARS) -> FinalResponse:
    """Facts and evidence are always shown. The model is called only when there is something to explain."""
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