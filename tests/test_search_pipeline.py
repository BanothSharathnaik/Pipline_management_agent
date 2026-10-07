import pytest

from src.ingestion.chunker import chunk_documents
from src.ingestion.parser import Document
from src.retrieval.chunk_embeddings import embed_chunks
from src.retrieval.embeddings import HashingEmbedder
from src.retrieval.search import RetrievalPipeline, extract_run_ids, format_report
from src.retrieval.vector_index import VectorIndex

KNOWN = ["r1", "r2", "r3", "r4"]


def mkd(doc_id, body, run_id=None, doc_type="error_log"):
    return Document(doc_id=doc_id, doc_type=doc_type, source="mock", body=body,
                    run_id=run_id, origin=f"{doc_id}.txt")


DOCS = [
    mkd("A", "disk full on node", run_id="r1"),
    mkd("B", "disk full on node again", run_id="r2"),
    mkd("C", "how to fix disk full problems", doc_type="troubleshooting"),
    mkd("D", "permission denied table", run_id="r3"),
]


def build(**kw):
    kw.setdefault("min_score", 0.15)
    e = HashingEmbedder(dim=4096)
    chunks = chunk_documents(DOCS)
    idx = VectorIndex(e.dim, e.model_name)
    idx.add(chunks, embed_chunks(chunks, e))
    return RetrievalPipeline(idx, e, known_run_ids=KNOWN, **kw)


def ids(results):
    return [r.chunk.chunk_id for r in results]


# --- extract_run_ids --------------------------------------------------------

@pytest.mark.parametrize("question,expected", [
    ("Why did run r2002 fail?", ["r2002"]),
    ("Why did run 123 fail?", ["123"]),
    ("why did my latest pipeline run fail?", []),
    ("compare r2002 and R2003", ["r2002", "r2003"]),
    ("run r2002 and again r2002", ["r2002"]),
    ("run #4567 failed", ["4567"]),
    ("how many runs failed in 2026?", []),
    ("run id 88 and run r7", ["88", "r7"]),
    ("", []),
])
def test_extract_run_ids(question, expected):
    assert extract_run_ids(question) == expected


def test_extract_run_ids_rejects_non_string():
    with pytest.raises(TypeError):
        extract_run_ids(None)


# --- pipeline ---------------------------------------------------------------

def test_model_mismatch_raises():
    e = HashingEmbedder(dim=8)
    with pytest.raises(ValueError, match="built with"):
        RetrievalPipeline(VectorIndex(8, "other"), e, known_run_ids=[])


def test_blank_question_raises():
    with pytest.raises(ValueError):
        build().retrieve("   ")


def test_stopword_only_question_raises():
    with pytest.raises(ValueError, match="searchable"):
        build().retrieve("why did the")


def test_invalid_top_k_raises():
    with pytest.raises(ValueError):
        build().retrieve("disk", top_k=0)


def test_linked_evidence_found_by_run_id():
    rep = build().retrieve("why did run r1 fail")
    assert list(rep.run_evidence) == ["r1"]
    assert ids(rep.run_evidence["r1"]) == ["A#0"]


def test_linked_evidence_ignores_min_score():
    rep = build().retrieve("why did run r3 fail kubernetes")  # no word overlap with D
    assert ids(rep.run_evidence["r3"]) == ["D#0"]


def test_related_excludes_linked_and_shows_other_runs():
    rep = build().retrieve("run r1 disk full")
    related = ids(rep.related.results)
    assert "A#0" not in related
    assert {"B#0", "C#0"} <= set(related)
    b = next(r for r in rep.related.results if r.chunk.chunk_id == "B#0")
    assert b.chunk.run_id == "r2"


def test_unknown_run_skips_semantic_search():
    rep = build().retrieve("why did run r999 fail")
    assert rep.unknown_run_ids == ("r999",)
    assert rep.run_evidence == {}
    assert rep.related.status == "no_evidence"
    assert any("not found" in n for n in rep.notes)


def test_known_run_without_documents_is_reported():
    rep = build().retrieve("why did run r4 fail disk full")
    assert rep.run_evidence == {"r4": ()}
    assert any("no diagnostic documents are linked to run r4" in n for n in rep.notes)
    assert rep.related.results


def test_mixed_known_and_unknown_runs():
    rep = build().retrieve("compare run r1 and run r999 disk full")
    assert rep.unknown_run_ids == ("r999",)
    assert "r1" in rep.run_evidence
    assert rep.related.status == "ok"


def test_question_without_run_id():
    rep = build().retrieve("disk full")
    assert rep.run_ids_mentioned == ()
    assert rep.run_evidence == {}
    assert {"A#0", "B#0", "C#0"} <= set(ids(rep.related.results))
    assert "D#0" not in ids(rep.related.results)


def test_high_min_score_gives_no_evidence_with_best_score():
    rep = build(min_score=0.99).retrieve("disk full")
    assert rep.related.status == "no_evidence"
    assert rep.related.best_score is not None


def test_run_id_is_case_insensitive():
    rep = build().retrieve("why did run R1 fail")
    assert rep.run_ids_mentioned == ("r1",)
    assert "r1" in rep.run_evidence


def test_retrieval_is_deterministic():
    p = build()
    a = p.retrieve("run r1 disk full")
    b = p.retrieve("run r1 disk full")
    assert ids(a.related.results) == ids(b.related.results)
    assert ids(a.run_evidence["r1"]) == ids(b.run_evidence["r1"])


def test_provenance_on_linked_evidence():
    c = build().retrieve("why did run r1 fail").run_evidence["r1"][0].chunk
    assert (c.doc_id, c.run_id, c.origin, c.source) == ("A", "r1", "A.txt", "mock")


def test_format_report_shows_labels_and_provenance():
    p = build()
    text = format_report(p.retrieve("why did run r1 fail disk"))
    assert "Evidence linked to run r1" in text
    assert "A#0" in text and "run=r1" in text and "source=mock" in text
    assert "Related evidence" in text
    unknown = format_report(p.retrieve("why did run r999 fail"))
    assert "not found" in unknown
    assert "no evidence above min score" in unknown