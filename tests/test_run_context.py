from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.ingestion.chunker import chunk_documents
from src.ingestion.parser import Document, load_documents
from src.monitoring.collector import load_mock_runs
from src.monitoring.models import RunRecord
from src.retrieval.chunk_embeddings import embed_chunks
from src.retrieval.embeddings import HashingEmbedder
from src.retrieval.run_context import (
    facts_from_run,
    format_context,
    plan_related_query,
    relation_to_run,
    retrieve_for_question,
)
from src.retrieval.search import RetrievalPipeline
from src.retrieval.vector_index import VectorIndex

ROOT = Path(__file__).resolve().parents[1]
RUNS = load_mock_runs(ROOT / "data" / "mock_runs.json").runs
DOCS = load_documents(ROOT / "data" / "diagnostic_documents").docs
BY_ID = {r.run_id: r for r in RUNS}


def t(s):
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def mkrun(start=None, end=None):
    return facts_from_run(RunRecord.from_dict({
        "run_id": "x", "job_id": "j", "source": "mock", "result_state": "FAILED",
        "start_time": start, "end_time": end,
    }))


# --- facts ------------------------------------------------------------------

def test_facts_r2002():
    f = facts_from_run(BY_ID["r2002"])
    assert (f.result_state, f.duration_seconds, f.source) == ("FAILED", 210.0, "mock")
    assert f.failed_tasks == ("ingest_customers",)
    assert f.skipped_tasks == ("transform_customers", "publish")
    assert "main.crm.customers" in f.error_message


def test_facts_r2005_has_no_error():
    f = facts_from_run(BY_ID["r2005"])
    assert f.error_message is None and f.failed_tasks == ()


def test_facts_r1004_failed_task():
    assert facts_from_run(BY_ID["r1004"]).failed_tasks == ("load",)


# --- relation ---------------------------------------------------------------

def test_relation_cases():
    run = mkrun("2026-10-02T02:00:00+00:00", "2026-10-02T02:10:00+00:00")
    assert relation_to_run(None, run) == "undated"
    assert relation_to_run(t("2026-10-01T00:00:00"), run) == "before this run"
    assert relation_to_run(t("2026-10-02T02:05:00"), run) == "within this run's time window"
    assert relation_to_run(t("2026-10-02T02:10:00"), run) == "within this run's time window"
    assert relation_to_run(t("2026-10-03T00:00:00"), run) == "after this run"


def test_relation_with_unknown_times():
    assert relation_to_run(t("2026-10-01T00:00:00"), mkrun()) == "run start unknown"
    open_run = mkrun("2026-10-02T02:00:00+00:00", None)
    assert "run end unknown" in relation_to_run(t("2026-10-03T00:00:00"), open_run)


# --- query planning ---------------------------------------------------------

def test_plan_uses_error_message_for_single_run():
    q, note = plan_related_query([facts_from_run(BY_ID["r2002"])])
    assert q == BY_ID["r2002"].error_message
    assert "recorded error message" in note


def test_plan_without_error_falls_back_to_question():
    q, note = plan_related_query([facts_from_run(BY_ID["r2005"])])
    assert q is None and "no recorded error message" in note


def test_plan_several_runs_uses_question():
    q, note = plan_related_query([facts_from_run(BY_ID["r2002"]), facts_from_run(BY_ID["r2003"])])
    assert q is None and "several runs" in note


def test_plan_no_runs():
    assert plan_related_query([]) == (None, None)


# --- pipeline override ------------------------------------------------------

def build(docs, known):
    e = HashingEmbedder(dim=4096)
    chunks = chunk_documents(docs)
    idx = VectorIndex(e.dim, e.model_name)
    idx.add(chunks, embed_chunks(chunks, e))
    return RetrievalPipeline(idx, e, known_run_ids=known)


def mkd(doc_id, body, run_id=None):
    return Document(doc_id=doc_id, doc_type="error_log", source="mock", body=body,
                    run_id=run_id, origin=f"{doc_id}.txt")


def test_related_query_override_changes_related_results():
    p = build([mkd("A", "disk full on node", "r1"), mkd("D", "permission denied table", "r3")],
              ["r1", "r3"])
    plain = p.retrieve("why did run r1 fail")
    assert plain.related.status == "no_evidence"
    over = p.retrieve("why did run r1 fail", related_query="permission denied")
    assert [r.chunk.chunk_id for r in over.related.results] == ["D#0"]


def test_blank_override_falls_back_to_question():
    p = build([mkd("A", "disk full on node", "r1"), mkd("B", "disk full again", "r2")], ["r1", "r2"])
    a = p.retrieve("disk full", related_query="   ")
    b = p.retrieve("disk full")
    assert [r.chunk.chunk_id for r in a.related.results] == [r.chunk.chunk_id for r in b.related.results]


# --- end to end on the real corpus (hashing embedder, no model needed) ------

@pytest.fixture(scope="module")
def pipeline():
    return build(DOCS, [r.run_id for r in RUNS])


def test_r2002_related_now_contains_the_similar_error_of_another_run(pipeline):
    ctx = retrieve_for_question(pipeline, "Why did run r2002 fail?", RUNS)
    linked = [r.chunk.chunk_id for r in ctx.report.run_evidence["r2002"]]
    related = [r.chunk.chunk_id for r in ctx.report.related.results]
    assert linked == ["ERR-002#0"]
    assert "ERR-002#0" not in related
    assert "ERR-003#0" in related
    assert ctx.query_used == BY_ID["r2002"].error_message
    assert ctx.facts[0].failed_tasks == ("ingest_customers",)


def test_r2005_uses_question_text(pipeline):
    ctx = retrieve_for_question(pipeline, "Why did run r2005 fail?", RUNS)
    assert ctx.query_used == "Why did run r2005 fail?"
    assert "no recorded error message" in ctx.query_note
    assert ctx.facts[0].error_message is None


def test_unknown_run_has_no_facts(pipeline):
    ctx = retrieve_for_question(pipeline, "Why did run r9999 fail?", RUNS)
    assert ctx.facts == () and ctx.query_note is None
    assert ctx.report.unknown_run_ids == ("r9999",)


def test_two_runs_use_question_text(pipeline):
    q = "compare run r2002 and run r2003"
    ctx = retrieve_for_question(pipeline, q, RUNS)
    assert len(ctx.facts) == 2 and ctx.query_used == q


def test_format_shows_facts_and_relation(pipeline):
    text = format_context(retrieve_for_question(pipeline, "Why did run r2002 fail?", RUNS))
    assert "Verified run facts (monitoring data, source=mock):" in text
    assert "failed tasks: ingest_customers" in text
    assert "skipped tasks: transform_customers, publish" in text
    assert "recorded error message of run r2002" in text
    assert "relation to run r2002:" in text


def test_format_unknown_run_shows_not_found(pipeline):
    text = format_context(retrieve_for_question(pipeline, "Why did run r9999 fail?", RUNS))
    assert "not found" in text and "Verified run facts" not in text