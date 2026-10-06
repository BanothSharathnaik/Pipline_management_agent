from pathlib import Path

import pytest

from src.ingestion.chunker import chunk_document, chunk_documents
from src.ingestion.parser import Document, load_documents
from src.retrieval.chunk_search import ChunkIndex

CORPUS = Path(__file__).resolve().parents[1] / "data" / "diagnostic_documents"


def mk(doc_id, body, doc_type="error_log", run_id=None, title=None):
    return Document(doc_id=doc_id, doc_type=doc_type, source="mock",
                    body=body, run_id=run_id, title=title, origin=f"{doc_id}.txt")


def test_result_carries_chunk_and_provenance():
    doc = mk("D", "disk full on node", run_id="r1")
    idx = ChunkIndex(chunk_documents([doc]))
    r = idx.search("disk")[0]
    assert r.chunk.chunk_id == "D#0"
    assert (r.chunk.doc_id, r.chunk.run_id, r.chunk.origin) == ("D", "r1", "D.txt")
    assert (r.chunk.start_char, r.chunk.end_char) == (0, len(doc.body))


def test_pinpoints_relevant_chunk():
    body = "disk full on node\npermission denied table\nrestart cluster"
    doc = mk("D", body)
    chunks = chunk_document(doc, max_chars=25, overlap_lines=0)
    assert [c.text for c in chunks] == [
        "disk full on node", "permission denied table", "restart cluster"]
    results = ChunkIndex(chunks).search("permission denied")
    assert len(results) == 1
    assert results[0].chunk.chunk_id == "D#1"
    assert results[0].chunk.text == "permission denied table"


def test_single_chunk_docs_match_document_level_scores():
    docs = [mk("A", "permission denied table"), mk("B", "table not found"), mk("C", "disk full")]
    results = ChunkIndex(chunk_documents(docs)).search("permission denied")
    assert [r.chunk.chunk_id for r in results] == ["A#0"]
    assert results[0].score == pytest.approx(0.8807, abs=1e-3)  # Phase 3 hand example


def test_filters():
    docs = [
        mk("A", "disk full", doc_type="error_log", run_id="r1"),
        mk("B", "disk full help", doc_type="troubleshooting"),
    ]
    idx = ChunkIndex(chunk_documents(docs))
    assert [r.chunk.chunk_id for r in idx.search("disk", doc_type="troubleshooting")] == ["B#0"]
    assert [r.chunk.chunk_id for r in idx.search("disk", run_id="r1")] == ["A#0"]
    assert idx.search("disk", run_id="r999") == []


def test_duplicate_chunk_id_raises():
    chunks = chunk_documents([mk("A", "disk full")])
    with pytest.raises(ValueError, match="duplicate chunk_id"):
        ChunkIndex(chunks + chunks)


def test_empty_index():
    idx = ChunkIndex([])
    assert idx.size == 0
    assert idx.search("anything") == []


def test_corpus_results_trace_back_to_source_text():
    docs = load_documents(CORPUS).docs
    by_id = {d.doc_id: d for d in docs}
    chunks = [c for d in docs for c in chunk_document(d, max_chars=120, overlap_lines=1)]
    idx = ChunkIndex(chunks)
    for q in ("permission denied customers", "show grants privileges",
              "unresolved column schema", "timeout seconds"):
        results = idx.search(q, top_k=5)
        assert results, q
        assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)
        for r in results:
            src = by_id[r.chunk.doc_id].body
            assert src[r.chunk.start_char:r.chunk.end_char] == r.chunk.text