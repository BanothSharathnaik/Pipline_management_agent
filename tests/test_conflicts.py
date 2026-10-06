from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.ingestion.chunker import chunk_documents
from src.ingestion.parser import Document, load_documents, parse_document
from src.monitoring.collector import load_mock_runs
from src.monitoring.models import RunRecord
from src.retrieval.chunk_embeddings import embed_chunks
from src.retrieval.conflicts import find_conflicts, object_names
from src.retrieval.embeddings import HashingEmbedder
from src.retrieval.run_context import format_context, retrieve_for_question
from src.retrieval.search import RetrievalPipeline
from src.retrieval.vector_index import VectorIndex

ROOT = Path(__file__).resolve().parents[1]
RUNS = load_mock_runs(ROOT / "data" / "mock_runs.json").runs
CORPUS = load_documents(ROOT / "data" / "diagnostic_documents").docs
DEMO = load_documents(ROOT / "data" / "conflict_demo")
T0 = datetime(2026, 10, 2, 9, 0, tzinfo=timezone.utc)


def mkdoc(claim="resolved", ts=T0, body="SELECT granted on main.crm.customers", doc_id="H"):
    return Document(doc_id=doc_id, doc_type="historical_failure", source="mock",
                    body=body, timestamp=ts, claim=claim, origin=f"{doc_id}.txt")


def mkrun(start="2026-10-03T02:00:00+00:00", state="FAILED",
          error="PERMISSION_DENIED on main.crm.customers", run_id="x"):
    return RunRecord.from_dict({"run_id": run_id, "job_id": "j", "source": "mock",
                                "result_state": state, "start_time": start,
                                "error_message": error})


# --- parser: claim header ----------------------------------------------------

def test_claim_header_parsed():
    d = parse_document("doc_id: T\ndoc_type: historical_failure\nsource: mock\nclaim: resolved\n---\nbody")
    assert d.claim == "resolved"


def test_claim_absent_is_none():
    assert parse_document("doc_id: T\ndoc_type: error_log\nsource: mock\n---\nbody").claim is None


def test_invalid_claim_raises():
    with pytest.raises(ValueError, match="claim"):
        parse_document("doc_id: T\ndoc_type: error_log\nsource: mock\nclaim: maybe\n---\nbody")


# --- object names --------------------------------------------------------------

def test_object_names_basic():
    assert object_names("no access to main.crm.customers.") == {"main.crm.customers"}
    assert object_names("table `Main.Sales.Orders_Clean` missing") == {"main.sales.orders_clean"}
    assert object_names("two names a.b.c and d.e.f") == {"a.b.c", "d.e.f"}


def test_object_names_ignores_two_part_and_plain_text():
    assert object_names("crm.customers only") == frozenset()
    assert object_names("") == frozenset()


# --- find_conflicts ------------------------------------------------------------

def test_conflict_detected():
    c = find_conflicts([mkdoc()], [mkrun()])
    assert len(c) == 1
    assert (c[0].doc_id, c[0].run_id, c[0].shared_objects) == ("H", "x", ("main.crm.customers",))


def test_run_before_claim_is_not_a_conflict():
    assert find_conflicts([mkdoc()], [mkrun(start="2026-10-02T02:00:00+00:00")]) == []


def test_run_at_exactly_claim_time_is_not_a_conflict():
    assert find_conflicts([mkdoc()], [mkrun(start="2026-10-02T09:00:00+00:00")]) == []


def test_later_success_is_not_a_conflict():
    assert find_conflicts([mkdoc()], [mkrun(state="SUCCESS")]) == []


def test_failed_run_without_error_text_is_not_a_conflict():
    assert find_conflicts([mkdoc()], [mkrun(error=None)]) == []


def test_no_shared_object_is_not_a_conflict():
    assert find_conflicts([mkdoc()], [mkrun(error="not found: main.sales.orders_clean")]) == []


def test_doc_without_claim_or_timestamp_is_ignored():
    assert find_conflicts([mkdoc(claim=None)], [mkrun()]) == []
    assert find_conflicts([mkdoc(ts=None)], [mkrun()]) == []


def test_doc_naming_no_object_is_ignored():
    assert find_conflicts([mkdoc(body="all fixed now")], [mkrun()]) == []


def test_timed_out_run_counts_as_failure():
    assert len(find_conflicts([mkdoc()], [mkrun(state="TIMED_OUT")])) == 1


def test_results_sorted_by_run_start():
    runs = [mkrun(start="2026-10-05T00:00:00+00:00", run_id="b"),
            mkrun(start="2026-10-04T00:00:00+00:00", run_id="a")]
    assert [c.run_id for c in find_conflicts([mkdoc()], runs)] == ["a", "b"]


# --- real mock data -----------------------------------------------------------

def test_demo_document_loads_clean():
    assert DEMO.issues == [] and len(DEMO.docs) == 1
    assert DEMO.docs[0].doc_id == "HIST-003" and DEMO.docs[0].claim == "resolved"


def test_exactly_one_conflict_on_mock_runs():
    found = find_conflicts(DEMO.docs, RUNS)
    assert [(c.doc_id, c.run_id) for c in found] == [("HIST-003", "r2003")]


def test_main_corpus_has_no_resolved_claims():
    assert find_conflicts(CORPUS, RUNS) == []


# --- end to end with the hashing embedder ------------------------------------------

def build(docs):
    e = HashingEmbedder(dim=4096)
    chunks = chunk_documents(docs)
    idx = VectorIndex(e.dim, e.model_name)
    idx.add(chunks, embed_chunks(chunks, e))
    return RetrievalPipeline(idx, e, known_run_ids=[r.run_id for r in RUNS])


def test_conflict_reported_when_document_is_retrieved():
    pipeline = build(CORPUS + DEMO.docs)
    ctx = retrieve_for_question(pipeline, "Why did run r2003 fail?", RUNS, docs=CORPUS + DEMO.docs)
    assert [(c.doc_id, c.run_id) for c in ctx.conflicts] == [("HIST-003", "r2003")]
    text = format_context(ctx)
    assert "POSSIBLE CONFLICTS" in text and "HIST-003" in text and "does not prove" in text


def test_no_conflict_reported_when_document_not_retrieved():
    pipeline = build(CORPUS)  # HIST-003 is not in the index
    ctx = retrieve_for_question(pipeline, "Why did run r2003 fail?", RUNS, docs=CORPUS + DEMO.docs)
    assert ctx.conflicts == ()
    assert "POSSIBLE CONFLICTS" not in format_context(ctx)


def test_docs_argument_is_optional():
    pipeline = build(CORPUS + DEMO.docs)
    assert retrieve_for_question(pipeline, "Why did run r2003 fail?", RUNS).conflicts == ()


def test_unrelated_run_gets_no_conflict():
    d = CORPUS + DEMO.docs
    ctx = retrieve_for_question(build(d), "Why did run r3004 fail?", RUNS, docs=d)
    assert ctx.conflicts == ()


def test_run_without_error_text_gets_no_conflict():
    d = CORPUS + DEMO.docs
    ctx = retrieve_for_question(build(d), "Why did run r2005 fail?", RUNS, docs=d)
    assert ctx.conflicts == ()


def test_related_run_keeps_conflict():
    d = CORPUS + DEMO.docs
    ctx = retrieve_for_question(build(d), "Why did run r2002 fail?", RUNS, docs=d)
    assert [(c.doc_id, c.run_id) for c in ctx.conflicts] == [("HIST-003", "r2003")]