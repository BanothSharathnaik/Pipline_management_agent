from pathlib import Path

from src.generation.llm_client import LLMResponse
from src.generation.prompt_builder import build_prompt
from src.generation.response_formatter import check_answer
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


def ctx_for(question):
    e = HashingEmbedder(dim=4096)
    chunks = chunk_documents(CORPUS)
    idx = VectorIndex(e.dim, e.model_name)
    idx.add(chunks, embed_chunks(chunks, e))
    pipeline = RetrievalPipeline(idx, e, known_run_ids=[r.run_id for r in RUNS])
    return retrieve_for_question(pipeline, question, RUNS, docs=CORPUS)


CTX = ctx_for("Why did run r2003 fail?")
BUNDLE = build_prompt(CTX)

GOOD = """Verified facts
- run r2003 failed [F1]
Possible explanations (hypotheses)
1. The principal lacks SELECT on main.crm.customers [E1]
Next
What to investigate next
- check the grants [E1]
Limits
- E2 is a similar case from another run [E2]"""


def resp(**kw):
    base = dict(text="t", model="m", prompt_tokens=100, completion_tokens=50,
                done_reason="stop", elapsed_s=1.0, num_ctx=4096)
    base.update(kw)
    return LLMResponse(**base)


def test_clean_answer_has_no_warnings():
    assert BUNDLE.evidence_ids["E1"] == "ERR-003#0"
    assert check_answer(GOOD, BUNDLE, CTX) == ()


def test_unknown_citation_flagged():
    w = check_answer(GOOD + "\n- extra [E9]", BUNDLE, CTX)
    assert any("not supplied" in x and "E9" in x for x in w)


def test_no_citations_flagged():
    text = "Verified facts\nPossible explanations\n1. a\nWhat to investigate next\nLimits"
    w = check_answer(text, BUNDLE, CTX)
    assert any("no citations" in x for x in w)


def test_invented_run_id_flagged_but_supplied_one_is_not():
    w = check_answer(GOOD + "\n- see run r7777 and r2002 [E3]", BUNDLE, CTX)
    assert any("r7777" in x for x in w)
    assert not any("r2002" in x for x in w)


def test_object_from_other_run_flagged():
    text = GOOD.replace("main.crm.customers", "main.sales.orders_clean")
    w = check_answer(text, BUNDLE, CTX)
    assert any("main.sales.orders_clean" in x for x in w)


def test_own_object_not_flagged():
    assert not any("objects" in x for x in check_answer(GOOD, BUNDLE, CTX))


def test_object_check_skipped_without_run_facts():
    ctx = ctx_for("disk heap memory join executor")
    assert ctx.facts == ()
    w = check_answer("see main.foo.bar", build_prompt(ctx), ctx)
    assert not any("objects" in x for x in w)


def test_missing_sections_flagged():
    w = check_answer("Verified facts\n- ok [F1]", BUNDLE, CTX)
    assert any("missing sections" in x and "limits" in x for x in w)


def test_all_uncited_explanations_flagged():
    text = ("Verified facts\n- x [F1]\nPossible explanations (hypotheses)\n1. a\n2. b\n"
            "What to investigate next\n- c\nLimits\n- d")
    assert any("2 of 2 explanation lines" in x for x in check_answer(text, BUNDLE, CTX))


def test_partly_cited_explanations_flagged():
    text = ("Verified facts\n- x [F1]\nPossible explanations (hypotheses)\n1. a [E1]\n2. b\n"
            "What to investigate next\n- c\nLimits\n- d")
    assert any("1 of 2 explanation lines" in x for x in check_answer(text, BUNDLE, CTX))


def test_cut_off_and_full_context_flagged():
    w = check_answer(GOOD, BUNDLE, CTX, resp(done_reason="length"))
    assert any("cut off" in x for x in w)
    w = check_answer(GOOD, BUNDLE, CTX, resp(prompt_tokens=4000, completion_tokens=96))
    assert any("context window" in x for x in w)


def test_normal_response_adds_no_flags():
    assert check_answer(GOOD, BUNDLE, CTX, resp()) == ()