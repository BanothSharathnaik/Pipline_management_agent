from pathlib import Path

import pytest

from src.generation.answer import generate_answer
from src.generation.llm_client import LLMError, LLMResponse
from src.generation.prompt_builder import SYSTEM_PROMPT
from src.ingestion.chunker import chunk_documents
from src.ingestion.parser import load_documents
from src.monitoring.collector import load_mock_runs
from src.retrieval.chunk_embeddings import embed_chunks
from src.retrieval.embeddings import HashingEmbedder
from src.retrieval.run_context import retrieve_for_question
from src.retrieval.search import RetrievalPipeline
from src.retrieval.vector_index import VectorIndex

ROOT = Path(__file__).resolve().parents[1]
RUNS = load_mock_runs(ROOT / "data" / "mock_runs.json").runs
CORPUS = load_documents(ROOT / "data" / "diagnostic_documents").docs

GOOD = """Verified facts
- run r2003 failed [F1]
Possible explanations (hypotheses)
1. The principal lacks SELECT on main.crm.customers [E1]
What to investigate next
- check the grants [E1]
Limits
- E2 is a similar case [E2]"""


class FakeLLM:
    model = "fake"

    def __init__(self, text=GOOD, error=None):
        self.text, self.error, self.calls = text, error, []

    def generate(self, system, user):
        self.calls.append((system, user))
        if self.error:
            raise LLMError(self.error)
        return LLMResponse(text=self.text, model="fake", prompt_tokens=100,
                           completion_tokens=50, done_reason="stop", elapsed_s=0.1, num_ctx=4096)


def ctx_for(question, min_score=0.15):
    e = HashingEmbedder(dim=4096)
    chunks = chunk_documents(CORPUS)
    idx = VectorIndex(e.dim, e.model_name)
    idx.add(chunks, embed_chunks(chunks, e))
    pipeline = RetrievalPipeline(idx, e, known_run_ids=[r.run_id for r in RUNS], min_score=min_score)
    return retrieve_for_question(pipeline, question, RUNS, docs=CORPUS)


def test_known_run_calls_model_with_the_built_prompt():
    llm = FakeLLM()
    out = generate_answer(ctx_for("Why did run r2003 fail?"), llm)
    assert len(llm.calls) == 1
    assert llm.calls[0][0] == SYSTEM_PROMPT
    assert "<monitoring_facts>" in llm.calls[0][1]
    assert out.llm_used is True and out.model_text == GOOD and out.llm_error is None


def test_verified_section_comes_before_generated_text():
    out = generate_answer(ctx_for("Why did run r2003 fail?"), FakeLLM())
    t = out.text
    assert "Verified run facts (monitoring data, source=mock):" in t
    assert t.index("VERIFIED MONITORING DATA") < t.index("GENERATED EXPLANATION") < t.index("AUTOMATIC CHECKS")
    assert "NOT verified" in t


def test_clean_answer_reports_limited_checks():
    out = generate_answer(ctx_for("Why did run r2003 fail?"), FakeLLM())
    assert out.warnings == ()
    assert "no problems found" in out.text and "do not prove" in out.text


def test_bad_answer_shows_warnings():
    bad = GOOD.replace("main.crm.customers", "main.sales.orders_clean")
    out = generate_answer(ctx_for("Why did run r2003 fail?"), FakeLLM(text=bad))
    assert out.warnings and "WARNING:" in out.text and "main.sales.orders_clean" in out.text


def test_unknown_run_never_calls_the_model():
    llm = FakeLLM()
    out = generate_answer(ctx_for("Why did run r9999 fail?"), llm)
    assert llm.calls == []
    assert out.llm_used is False and out.model_text is None
    assert "not found in monitoring data" in out.text
    assert "language model was not called" in out.text


def test_no_evidence_above_threshold_never_calls_the_model():
    llm = FakeLLM()
    out = generate_answer(ctx_for("disk heap memory", min_score=0.99), llm)
    assert llm.calls == []
    assert "language model was not called" in out.text


def test_model_down_falls_back_to_evidence_only():
    llm = FakeLLM(error="down")
    out = generate_answer(ctx_for("Why did run r2003 fail?"), llm)
    assert out.llm_used is False and out.llm_error == "down" and out.model_text is None
    assert "language model unavailable (down)" in out.text
    assert "Verified run facts" in out.text and "ERR-003#0" in out.text
    assert "=== NO GENERATED EXPLANATION:" in out.text  # fallback header present
    assert "=== GENERATED EXPLANATION ===" not in out.text  # no positive explanation


def test_prompt_budget_error_is_not_swallowed():
    with pytest.raises(ValueError, match="exceeds"):
        generate_answer(ctx_for("Why did run r2003 fail?"), FakeLLM(), max_chars=100)


def test_question_without_run_still_uses_the_model_when_evidence_exists():
    llm = FakeLLM(text="Verified facts\nnone\nPossible explanations\nWhat to investigate next\nLimits")
    out = generate_answer(ctx_for("disk heap memory join executor"), llm)
    assert len(llm.calls) == 1 and out.llm_used