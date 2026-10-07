from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.generation.llm_client import LLMResponse
from src.ingestion.chunker import chunk_documents
from src.ingestion.parser import load_documents
from src.monitoring.collector import load_mock_runs
from src.monitoring.models import RunRecord
from src.retrieval.chunk_embeddings import embed_chunks
from src.retrieval.embeddings import HashingEmbedder
from src.retrieval.search import RetrievalPipeline
from src.retrieval.vector_index import VectorIndex
from src.routing.executor import DisabledLLM, answer_question

ROOT = Path(__file__).resolve().parents[1]
RUNS = load_mock_runs(ROOT / "data" / "mock_runs.json").runs
CORPUS = load_documents(ROOT / "data" / "diagnostic_documents").docs
NOW = datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc)

GOOD = """Verified facts
- ok [F1]
Possible explanations (hypotheses)
1. something [E1]
What to investigate next
- look [E1]
Limits
- none"""


class FakeLLM:
    model = "fake"

    def __init__(self):
        self.calls = []

    def generate(self, system, user):
        self.calls.append((system, user))
        return LLMResponse(text=GOOD, model="fake", prompt_tokens=100, completion_tokens=50,
                           done_reason="stop", elapsed_s=0.1, num_ctx=4096)


def no_pipeline():
    raise AssertionError("the retrieval pipeline must not be built for this question")


def hashing_pipeline():
    e = HashingEmbedder(dim=4096)
    chunks = chunk_documents(CORPUS)
    idx = VectorIndex(e.dim, e.model_name)
    idx.add(chunks, embed_chunks(chunks, e))
    return RetrievalPipeline(idx, e, known_run_ids=[r.run_id for r in RUNS], min_score=0.15)


def ask(question, llm=None, factory=no_pipeline):
    llm = llm or FakeLLM()
    return answer_question(question, NOW, RUNS, factory, CORPUS, llm), llm


# --- structured questions: no pipeline, no model -------------------------------------

@pytest.mark.parametrize("question", [
    "Which pipelines failed yesterday?",
    "How many failures occurred in the last 24 hours?",
    "What is the latest run?",
    "Which pipeline had the longest runtime last week?",
    "Which runs exceeded 10 minutes?",
    "Give me a summary of pipeline health",
    "Are there recurring errors across multiple pipeline runs?",
    "What happened during run r2002?",
    "How has pipeline execution duration changed over time?",
])
def test_structured_questions_use_neither_pipeline_nor_model(question):
    answer, llm = ask(question)
    assert llm.calls == [] and answer.llm_used is False
    assert answer.text.startswith("[MOCK DATA] 16 runs loaded")


def test_failed_yesterday():
    t = ask("Which pipelines failed yesterday?")[0].text
    assert t.index("r1004") < t.index("r3004") < t.index("r2005")
    assert "r2002" not in t
    assert "data coverage: 16 runs" in t


def test_failed_in_empty_window_explains_coverage():
    t = ask("Which pipelines failed in the last 1 hours?")[0].text
    assert "No failed runs in last 1 hour." in t and "data coverage" in t


def test_count_in_window_and_overall():
    assert "Failures: 1 (FAILED or TIMED_OUT) in last 24 hours" in ask(
        "How many failures occurred in the last 24 hours?")[0].text
    assert "Failures: 6 (FAILED or TIMED_OUT) in all loaded runs" in ask(
        "How many runs failed in 2026?")[0].text


def test_longest_last_week():
    t = ask("Which pipeline had the longest runtime last week?")[0].text
    assert t.index("r3002") < t.index("r1003")
    assert "30m00s" in t


def test_slow_threshold_order_and_exclusions():
    t = ask("Which runs exceeded 10 minutes?")[0].text
    assert t.index("r3002") < t.index("r1003") < t.index("r2001") < t.index("r2004")
    assert "r3003" not in t


def test_slow_threshold_empty():
    assert "No runs exceed 300 seconds in today (UTC)." in ask(
        "Which runs exceeded 5 minutes today?")[0].text


def test_summary_counts():
    t = ask("Give me a summary of pipeline health")[0].text
    assert "  SUCCESS    7" in t and "  FAILED     5" in t and "Total: 16 runs" in t


def test_repeats_none():
    assert "No exactly repeated error messages." in ask(
        "Are there recurring errors across multiple pipeline runs?")[0].text


def test_run_details_and_unknown_run():
    t = ask("compare run r2002 and run r2003")[0].text
    assert "Run:        r2002" in t and "Run:        r2003" in t
    assert "ingest_customers  FAILED" in t
    assert "No run found with id r9999." in ask("What happened during run r9999?")[0].text


def test_latest_details():
    t = ask("What is the latest run?")[0].text
    assert "Run:        r1005" in t and "Lifecycle:  RUNNING" in t


def test_unsupported_duration_says_so():
    t = ask("How has pipeline execution duration changed over time?")[0].text
    assert "not implemented yet" in t and "cannot be answered yet" in t


# --- retrieval questions -------------------------------------------------------------

def test_run_diagnosis_calls_model_once():
    answer, llm = ask("Why did run r2002 fail?", factory=hashing_pipeline)
    assert answer.llm_used is True and len(llm.calls) == 1
    assert "Verified run facts" in answer.text and "GENERATED EXPLANATION" in answer.text


def test_latest_failed_resolves_to_r2005_and_names_the_running_run():
    answer, llm = ask("Why did my latest pipeline run fail?", factory=hashing_pipeline)
    assert answer.resolved_run_id == "r2005"
    assert "Resolved 'latest failed run' to r2005." in answer.text
    assert "most recent run overall is r1005" in answer.text
    assert "[F1] run r2005" in llm.calls[0][1]


def test_explain_without_run_id_uses_model_with_no_facts():
    answer, llm = ask("What are the possible causes of this permission denied failure?",
                      factory=hashing_pipeline)
    assert answer.intent == "explain" and len(llm.calls) == 1
    assert "No monitoring facts are available" in llm.calls[0][1]


def test_history_never_calls_model_and_verifies_cited_runs():
    answer, llm = ask(
        "Have we seen this permission denied error on main.crm.customers before?",
        factory=hashing_pipeline)
    assert llm.calls == [] and answer.llm_used is False
    t = answer.text
    assert "Verified monitoring records for runs cited by the evidence:" in t
    assert "r2002  customer_etl  FAILED" in t and "r2003  customer_etl  FAILED" in t
    assert "does not say which error" in t


def test_unrouted_question_gets_evidence_only():
    answer, llm = ask("how do I bake bread", factory=hashing_pipeline)
    assert llm.calls == [] and "no evidence above min score" in answer.text


def test_disabled_llm_keeps_facts_and_says_why():
    answer, _ = ask("Why did run r2002 fail?", llm=DisabledLLM(), factory=hashing_pipeline)
    assert answer.llm_used is False
    assert "language model unavailable (disabled with --no-llm)" in answer.text
    assert "Verified run facts" in answer.text


def test_blank_question_raises():
    with pytest.raises(ValueError):
        ask("   ")


def test_banner_reflects_source_labels():
    live = [r for r in RUNS[:1]]
    assert answer_question("Which pipelines failed?", NOW, live, no_pipeline, CORPUS,
                           FakeLLM()).text.startswith("[MOCK DATA] 1 runs loaded")


LIVE = RunRecord.from_dict({
    "run_id": "1001", "job_id": "9", "job_name": "JOB-X", "source": "live",
    "result_state": "FAILED", "start_time": "2026-10-07T03:00:00+00:00",
    "end_time": "2026-10-07T03:01:00+00:00", "error_message": "Exception: Deliberate failure",
})


def empty_pipeline():
    e = HashingEmbedder(dim=64)
    return RetrievalPipeline(VectorIndex(e.dim, e.model_name), e, known_run_ids=["1001"])


def test_facts_without_any_evidence_do_not_call_the_model():
    llm = FakeLLM()
    out = answer_question("Why did run 1001 fail?", NOW, [LIVE], empty_pipeline, CORPUS, llm)
    assert llm.calls == [] and out.llm_used is False
    assert out.text.startswith("[LIVE DATA] 1 runs loaded")
    assert "Exception: Deliberate failure" in out.text
    assert "no diagnostic evidence matched it" in out.text
    assert "language model was not called" in out.text


def test_live_runs_get_the_mock_documents_note():
    text = answer_question("Which pipelines failed?", NOW, [LIVE], no_pipeline, CORPUS, FakeLLM()).text
    assert "not written about these live runs" in text
    mock_text = answer_question("Which pipelines failed?", NOW, RUNS, no_pipeline, CORPUS, FakeLLM()).text
    assert "not written about these live runs" not in mock_text


def test_mixed_sources_are_labelled_in_the_banner():
    text = answer_question("Which pipelines failed?", NOW, RUNS[:1] + [LIVE], no_pipeline,
                           CORPUS, FakeLLM()).text
    assert text.startswith("[LIVE/MOCK DATA] 2 runs loaded")


def test_data_note_is_shown():
    text = answer_question("Which pipelines failed?", NOW, [LIVE], no_pipeline, CORPUS, FakeLLM(),
                           data_note="live snapshot fetched earlier").text
    assert "Data note: live snapshot fetched earlier" in text