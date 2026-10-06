import pytest

from src.ingestion.chunker import Chunk
from src.retrieval.evidence import select_evidence
from src.retrieval.vector_index import VectorResult


def res(chunk_id, score, rank=1):
    c = Chunk(chunk_id=chunk_id, doc_id=chunk_id.split("#")[0], index=0, text="t",
              start_char=0, end_char=1, doc_type="error_log", source="mock", origin="o")
    return VectorResult(rank=rank, score=score, chunk=c)


def test_threshold_is_inclusive():
    ev = select_evidence([res("A#0", 0.15), res("B#0", 0.1499)], min_score=0.15)
    assert [r.chunk.chunk_id for r in ev.results] == ["A#0"]
    assert ev.dropped_low_score == 1


def test_all_below_threshold_is_no_evidence_with_best_score():
    ev = select_evidence([res("A#0", 0.114), res("B#0", 0.079)], min_score=0.15)
    assert ev.status == "no_evidence"
    assert ev.results == ()
    assert ev.best_score == pytest.approx(0.114)
    assert ev.dropped_low_score == 2


def test_empty_input():
    ev = select_evidence([])
    assert ev.status == "no_evidence"
    assert ev.best_score is None
    assert ev.results == ()


def test_per_doc_cap_keeps_best_chunks():
    hits = [res("A#0", 0.9), res("A#1", 0.8), res("A#2", 0.7), res("B#0", 0.6)]
    ev = select_evidence(hits, min_score=0.0, max_per_doc=2, limit=10)
    assert [r.chunk.chunk_id for r in ev.results] == ["A#0", "A#1", "B#0"]
    assert ev.dropped_per_doc == 1


def test_ranks_are_renumbered_from_one():
    hits = [res("A#0", 0.9, rank=1), res("B#0", 0.05, rank=2), res("C#0", 0.5, rank=3)]
    ev = select_evidence(hits, min_score=0.15)
    assert [(r.rank, r.chunk.chunk_id) for r in ev.results] == [(1, "A#0"), (2, "C#0")]


def test_limit_applies_after_filters():
    hits = [res(f"D{i}#0", 0.9 - i * 0.01) for i in range(6)]
    ev = select_evidence(hits, min_score=0.0, limit=3)
    assert len(ev.results) == 3
    assert ev.status == "ok"


def test_unsorted_input_is_sorted_and_ties_are_deterministic():
    hits = [res("B#0", 0.5), res("A#0", 0.5), res("C#0", 0.9)]
    ev = select_evidence(hits, min_score=0.0)
    assert [r.chunk.chunk_id for r in ev.results] == ["C#0", "A#0", "B#0"]


def test_best_score_is_pre_filter_value():
    ev = select_evidence([res("A#0", 0.6), res("A#1", 0.5), res("A#2", 0.4)],
                         min_score=0.45, max_per_doc=1)
    assert ev.best_score == pytest.approx(0.6)
    assert len(ev.results) == 1


def test_invalid_parameters():
    with pytest.raises(ValueError):
        select_evidence([], max_per_doc=0)
    with pytest.raises(ValueError):
        select_evidence([], limit=0)