import numpy as np
import pytest

from src.ingestion.chunker import Chunk
from src.retrieval.embeddings import HashingEmbedder
from src.retrieval.vector_index import VectorIndex


def mkc(chunk_id, doc_type="error_log", run_id=None):
    return Chunk(chunk_id=chunk_id, doc_id=chunk_id.split("#")[0], index=0, text=f"text {chunk_id}",
                 start_char=0, end_char=10, doc_type=doc_type, source="mock",
                 origin=f"{chunk_id}.txt", run_id=run_id)


def hand_index():
    idx = VectorIndex(dim=2, model_name="m")
    idx.add([mkc("d1"), mkc("d2"), mkc("d3")], np.array([[1, 0], [1, 1], [0, 1]], dtype=np.float32))
    return idx


def ids(results):
    return [r.chunk.chunk_id for r in results]


def test_hand_calculated_ranking():
    res = hand_index().search(np.array([2.0, 1.0]), top_k=3)
    assert ids(res) == ["d2", "d1", "d3"]
    assert [r.rank for r in res] == [1, 2, 3]
    assert res[0].score == pytest.approx(0.9487, abs=1e-3)
    assert res[1].score == pytest.approx(0.8944, abs=1e-3)
    assert res[2].score == pytest.approx(0.4472, abs=1e-3)


def test_stored_vectors_are_normalised():
    idx = VectorIndex(2, "m")
    idx.add([mkc("a")], np.array([[3.0, 4.0]]))
    assert idx.search(np.array([3.0, 4.0]))[0].score == pytest.approx(1.0, abs=1e-5)
    assert idx.search(np.array([30.0, 40.0]))[0].score == pytest.approx(1.0, abs=1e-5)


def test_top_k_limits_results():
    assert len(hand_index().search(np.array([2.0, 1.0]), top_k=2)) == 2


def test_top_k_larger_than_index():
    assert len(hand_index().search(np.array([2.0, 1.0]), top_k=50)) == 3


def test_ties_are_deterministic():
    idx = VectorIndex(2, "m")
    idx.add([mkc("b"), mkc("a")], np.array([[1.0, 0.0], [1.0, 0.0]]))
    assert ids(idx.search(np.array([1.0, 0.0]), top_k=2)) == ["a", "b"]
    assert ids(idx.search(np.array([1.0, 0.0]), top_k=1)) == ["a"]


def test_empty_index_returns_nothing():
    assert VectorIndex(2, "m").search(np.array([1.0, 0.0])) == []


def test_zero_query_returns_nothing():
    assert hand_index().search(np.array([0.0, 0.0])) == []


def test_query_dimension_mismatch_raises():
    with pytest.raises(ValueError, match="dim"):
        hand_index().search(np.array([1.0, 0.0, 0.0]))


def test_add_dimension_mismatch_raises():
    with pytest.raises(ValueError, match="dimension mismatch"):
        VectorIndex(2, "m").add([mkc("a")], np.array([[1.0, 0.0, 0.0]]))


def test_add_row_count_mismatch_raises():
    with pytest.raises(ValueError, match="chunks"):
        VectorIndex(2, "m").add([mkc("a"), mkc("b")], np.array([[1.0, 0.0]]))


def test_add_non_finite_raises():
    with pytest.raises(ValueError, match="NaN"):
        VectorIndex(2, "m").add([mkc("a")], np.array([[np.nan, 1.0]]))


def test_add_zero_vector_raises_and_names_chunk():
    with pytest.raises(ValueError, match="z"):
        VectorIndex(2, "m").add([mkc("ok"), mkc("z")], np.array([[1.0, 0.0], [0.0, 0.0]]))


def test_duplicate_chunk_id_raises():
    idx = VectorIndex(2, "m")
    with pytest.raises(ValueError, match="within the batch"):
        idx.add([mkc("a"), mkc("a")], np.array([[1.0, 0.0], [0.0, 1.0]]))
    idx.add([mkc("a")], np.array([[1.0, 0.0]]))
    with pytest.raises(ValueError, match="already in index"):
        idx.add([mkc("a")], np.array([[0.0, 1.0]]))


def test_failed_add_leaves_index_unchanged():
    idx = VectorIndex(2, "m")
    with pytest.raises(ValueError):
        idx.add([mkc("a"), mkc("z")], np.array([[1.0, 0.0], [0.0, 0.0]]))
    assert idx.size == 0
    idx.add([mkc("a"), mkc("z")], np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert idx.size == 2


def test_add_empty_is_noop():
    idx = VectorIndex(2, "m")
    idx.add([], np.zeros((0, 2), dtype=np.float32))
    assert idx.size == 0


def test_filters():
    idx = VectorIndex(2, "m")
    idx.add(
        [mkc("a", "error_log", "r1"), mkc("b", "troubleshooting"), mkc("c", "error_log", "r2")],
        np.array([[1.0, 0.0], [1.0, 0.1], [0.9, 0.2]]),
    )
    q = np.array([1.0, 0.0])
    assert ids(idx.search(q, top_k=5, doc_type="troubleshooting")) == ["b"]
    assert ids(idx.search(q, top_k=5, run_id="r2")) == ["c"]
    assert ids(idx.search(q, top_k=5, doc_type="error_log", run_id="r1")) == ["a"]
    assert idx.search(q, run_id="r999") == []


def test_min_score():
    idx = hand_index()
    q = np.array([2.0, 1.0])
    assert ids(idx.search(q, top_k=3, min_score=0.9)) == ["d2"]
    assert ids(idx.search(q, top_k=3, min_score=0.89)) == ["d2", "d1"]
    assert idx.search(q, top_k=3, min_score=0.99) == []


def test_invalid_top_k_raises():
    with pytest.raises(ValueError):
        hand_index().search(np.array([1.0, 0.0]), top_k=0)


def test_negative_similarity_range():
    idx = VectorIndex(2, "m")
    idx.add([mkc("pos"), mkc("neg")], np.array([[1.0, 0.0], [-1.0, 0.0]]))
    res = idx.search(np.array([1.0, 0.0]), top_k=2)
    assert ids(res) == ["pos", "neg"]
    assert res[0].score == pytest.approx(1.0, abs=1e-5)
    assert res[1].score == pytest.approx(-1.0, abs=1e-5)
    assert ids(idx.search(np.array([1.0, 0.0]), top_k=2, min_score=0.0)) == ["pos"]


def test_identical_vectors_both_returned():
    idx = VectorIndex(2, "m")
    idx.add([mkc("a"), mkc("b")], np.array([[1.0, 1.0], [1.0, 1.0]]))
    assert ids(idx.search(np.array([1.0, 1.0]), top_k=5)) == ["a", "b"]


def test_search_text_model_mismatch_raises():
    e = HashingEmbedder(dim=8)
    with pytest.raises(ValueError, match="built with"):
        VectorIndex(8, "other-model").search_text("permission denied", e)
    with pytest.raises(ValueError, match="built with"):
        VectorIndex(16, e.model_name).search_text("permission denied", e)


def test_search_text_end_to_end():
    e = HashingEmbedder(dim=4096)
    texts = ["permission denied table", "disk full node", "timeout seconds"]
    idx = VectorIndex(e.dim, e.model_name)
    idx.add([mkc("c0"), mkc("c1"), mkc("c2")], e.embed(texts))
    res = idx.search_text("permission denied", e, top_k=1)
    assert ids(res) == ["c0"]


def test_results_carry_provenance():
    idx = VectorIndex(2, "m")
    chunk = mkc("ERR-9#0", run_id="r1004")
    idx.add([chunk], np.array([[1.0, 0.0]]))
    r = idx.search(np.array([1.0, 0.0]))[0]
    assert r.chunk is chunk
    assert (r.chunk.doc_id, r.chunk.run_id, r.chunk.origin, r.chunk.source) == (
        "ERR-9", "r1004", "ERR-9#0.txt", "mock")


@pytest.mark.slow
def test_real_model_fixes_synonym_probe():
    from pathlib import Path
    from src.ingestion.chunker import chunk_document
    from src.ingestion.parser import load_documents
    from src.retrieval.chunk_embeddings import embed_chunks
    from src.retrieval.embeddings import SentenceTransformerEmbedder

    docs = load_documents(Path(__file__).resolve().parents[1] / "data" / "diagnostic_documents").docs
    chunks = [c for d in docs for c in chunk_document(d)]
    e = SentenceTransformerEmbedder()
    idx = VectorIndex(e.dim, e.model_name)
    idx.add(chunks, embed_chunks(chunks, e))
    top = idx.search_text("access denied for customer data", e, top_k=2)
    assert {r.chunk.doc_id for r in top} == {"ERR-002", "ERR-003"}