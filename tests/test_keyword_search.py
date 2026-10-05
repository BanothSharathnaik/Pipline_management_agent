import pytest

from src.ingestion.parser import Document, load_documents
from src.retrieval.keyword_search import TfidfIndex

from pathlib import Path

CORPUS = Path(__file__).resolve().parents[1] / "data" / "diagnostic_documents"


def mk(doc_id, body, doc_type="error_log", run_id=None, title=None):
    return Document(doc_id=doc_id, doc_type=doc_type, source="mock",
                    body=body, run_id=run_id, title=title, origin=f"{doc_id}.txt")


def hand_index():
    return TfidfIndex([
        mk("D1", "permission denied table"),
        mk("D2", "table not found"),
        mk("D3", "disk full"),
    ])


def test_hand_calculated_example():
    results = hand_index().search("permission denied")
    assert len(results) == 1
    assert results[0].doc.doc_id == "D1"
    assert results[0].rank == 1
    assert results[0].score == pytest.approx(0.8807, abs=1e-3)
    assert results[0].matched_terms == ("denied", "permission")


def test_idf_values():
    idx = hand_index()
    assert idx.idf("permission") == pytest.approx(1.6931, abs=1e-3)
    assert idx.idf("table") == pytest.approx(1.2877, abs=1e-3)
    assert idx.idf("nonexistent") is None


def test_empty_index_returns_nothing():
    assert TfidfIndex([]).search("anything") == []


def test_unknown_terms_return_nothing():
    assert hand_index().search("kubernetes") == []


def test_stopword_only_query_returns_nothing():
    assert hand_index().search("the of and") == []


def test_invalid_top_k_raises():
    with pytest.raises(ValueError):
        hand_index().search("table", top_k=0)


def test_top_k_limits_and_ranks_are_sequential():
    idx = TfidfIndex([mk("A", "disk full"), mk("B", "disk error"), mk("C", "disk space low")])
    results = idx.search("disk", top_k=2)
    assert len(results) == 2
    assert [r.rank for r in results] == [1, 2]
    assert results[0].score >= results[1].score


def test_ties_broken_by_doc_id():
    idx = TfidfIndex([mk("B", "disk full"), mk("A", "disk full")])
    assert [r.doc.doc_id for r in idx.search("disk full")] == ["A", "B"]


def test_filter_by_doc_type_and_run_id():
    idx = TfidfIndex([
        mk("A", "disk full", doc_type="error_log", run_id="r1"),
        mk("B", "disk full help", doc_type="troubleshooting"),
    ])
    assert [r.doc.doc_id for r in idx.search("disk", doc_type="troubleshooting")] == ["B"]
    assert [r.doc.doc_id for r in idx.search("disk", run_id="r1")] == ["A"]
    assert idx.search("disk", run_id="r999") == []


def test_duplicate_doc_id_raises():
    with pytest.raises(ValueError):
        TfidfIndex([mk("A", "x y"), mk("A", "z w")])


def test_results_carry_provenance():
    idx = TfidfIndex([mk("A", "disk full", run_id="r1")])
    r = idx.search("disk")[0]
    assert (r.doc.doc_id, r.doc.origin, r.doc.source, r.doc.run_id) == ("A", "A.txt", "mock", "r1")


# --- real corpus ------------------------------------------------------------

@pytest.fixture(scope="module")
def corpus_index():
    return TfidfIndex(load_documents(CORPUS).docs)


def test_corpus_permission_query_finds_both_variants(corpus_index):
    top2 = corpus_index.search("permission denied main.crm.customers", top_k=2)
    assert {r.doc.doc_id for r in top2} == {"ERR-002", "ERR-003"}


def test_corpus_out_of_memory_query(corpus_index):
    assert corpus_index.search("out of memory heap", top_k=1)[0].doc.doc_id == "ERR-006"


def test_corpus_timeout_query(corpus_index):
    assert corpus_index.search("timeout 1800 seconds", top_k=1)[0].doc.doc_id == "ERR-005"


def test_corpus_scores_valid_and_sorted(corpus_index):
    results = corpus_index.search("permission denied customers table", top_k=10)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 < s <= 1.0 + 1e-9 for s in scores)