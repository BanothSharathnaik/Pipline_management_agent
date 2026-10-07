from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence

from src.generation.llm_client import LLMResponse
from src.generation.prompt_builder import PromptBundle
from src.retrieval.conflicts import object_names
from src.retrieval.run_context import RunContext, format_context
from src.retrieval.search import extract_run_ids

_CITATION = re.compile(r"\b([EF])(\d{1,3})\b")
_LIST_LINE = re.compile(r"\s*(\d+[.)]|[-*\u2022])\s+")
REQUIRED_SECTIONS = ("verified facts", "possible explanations", "what to investigate next", "limits")


@dataclass(frozen=True)
class FinalResponse:
    text: str
    llm_used: bool
    model_text: Optional[str]
    warnings: tuple[str, ...]
    llm_error: Optional[str]


def _supplied_run_ids(ctx: RunContext) -> set[str]:
    ids = {f.run_id.lower() for f in ctx.facts}
    ids |= {r.lower() for r in ctx.report.run_ids_mentioned}
    for hits in ctx.report.run_evidence.values():
        ids |= {h.chunk.run_id.lower() for h in hits if h.chunk.run_id}
    ids |= {h.chunk.run_id.lower() for h in ctx.report.related.results if h.chunk.run_id}
    return ids


def _own_objects(ctx: RunContext) -> frozenset[str]:
    """Object names that belong to this question: its text, the runs' own errors, linked evidence."""
    names = object_names(ctx.report.question)
    for f in ctx.facts:
        if f.error_message:
            names = names | object_names(f.error_message)
    for hits in ctx.report.run_evidence.values():
        for h in hits:
            names = names | object_names(f"{h.chunk.title or ''}\n{h.chunk.text}")
    return names


def _has_section(text_low: str, section: str) -> bool:
    if section == "verified facts":
        return "verified facts" in text_low
    if section == "possible explanations":
        return "possible explanations" in text_low or "hypotheses" in text_low
    if section == "what to investigate next":
        return any(k in text_low for k in ("what to investigate next", "next steps", "recommended actions"))
    if section == "limits":
        return any(k in text_low for k in ("limits", "limitations", "investigation scope", "scope bounds"))
    return section in text_low


def _explanation_lines(answer: str) -> list[str]:
    low = answer.lower()
    start = low.find("possible explanations")
    if start == -1:
        start = low.find("hypotheses")
    if start == -1:
        return []
    newline = answer.find("\n", start)
    if newline == -1:
        return []
    body_start = newline + 1
    end_markers = (
        "what to investigate next", "next steps", "recommended actions",
        "limits", "limitations", "investigation scope", "## 11", "## 12",
    )
    ends = [low.find(h, body_start) for h in end_markers]
    ends = [e for e in ends if e != -1]
    body = answer[body_start: min(ends) if ends else len(answer)]
    return [ln.strip() for ln in body.splitlines() if _LIST_LINE.match(ln)]


def check_answer(
    answer: str,
    bundle: PromptBundle,
    ctx: RunContext,
    response: Optional[LLMResponse] = None,
) -> tuple[str, ...]:
    """Cheap, deterministic checks on generated text. Returns warnings; empty means none found."""
    warnings: list[str] = []
    supplied = set(bundle.evidence_ids) | set(bundle.fact_ids)

    cited = [f"{a}{n}" for a, n in _CITATION.findall(answer)]
    unknown = sorted(set(c for c in cited if c not in supplied))
    if unknown:
        warnings.append(f"cites IDs that were not supplied: {', '.join(unknown)}")
    if supplied and not cited:
        warnings.append("the answer contains no citations")

    extra_runs = [r for r in extract_run_ids(answer) if r not in _supplied_run_ids(ctx)]
    if extra_runs:
        warnings.append(f"mentions run IDs that were not in the supplied data: {', '.join(extra_runs)}")

    if ctx.facts:
        stray = sorted(object_names(answer) - _own_objects(ctx))
        if stray:
            warnings.append(
                "names objects that are not in this run's own facts or linked evidence "
                f"(they may come from other runs' evidence, or be invented): {', '.join(stray)}"
            )

    lines = _explanation_lines(answer)
    uncited = [ln for ln in lines if not _CITATION.search(ln)]
    if uncited:
        warnings.append(f"{len(uncited)} of {len(lines)} explanation lines cite no evidence or fact ID")

    low = answer.lower()
    missing = [s for s in REQUIRED_SECTIONS if not _has_section(low, s)]
    if missing:
        warnings.append(f"the answer is missing sections: {', '.join(missing)}")

    if response is not None:
        if response.cut_off:
            warnings.append("the answer was cut off at the length limit")
        if response.context_nearly_full:
            warnings.append("prompt plus answer filled the context window; the prompt may have been truncated")

    return tuple(warnings)


def render_response(
    ctx: RunContext,
    model_text: Optional[str],
    warnings: Sequence[str],
    llm_error: Optional[str],
    nothing_to_explain: bool = False,
) -> str:
    lines = ["=== VERIFIED MONITORING DATA AND RETRIEVED EVIDENCE ===", format_context(ctx), ""]
    if model_text is not None:
        lines += [
            "=== GENERATED EXPLANATION (written by a language model; NOT verified; "
            "the data above is authoritative) ===",
            model_text,
            "",
            "=== AUTOMATIC CHECKS ===",
        ]
        if warnings:
            lines += [f"WARNING: {w}" for w in warnings]
        else:
            lines.append(
                "no problems found by the automatic checks (the checks are limited and "
                "do not prove the explanation is correct)"
            )
    elif llm_error is not None:
        lines += [
            f"=== NO GENERATED EXPLANATION: language model unavailable ({llm_error}) ===",
            "The verified data and retrieved evidence above are still valid.",
        ]
    elif nothing_to_explain:
        if ctx.facts:
            reason = ("The monitoring data above is verified, but no diagnostic evidence matched it, "
                      "so no cause can be established from the available information and the "
                      "language model was not called.")
        else:
            reason = ("Nothing in the monitoring data or retrieved evidence supports an explanation, "
                      "so the language model was not called.")
        lines += ["=== NO EXPLANATION GENERATED ===", reason]
    return "\n".join(lines)