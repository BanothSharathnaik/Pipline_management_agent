from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.ingestion.chunker import chunk_document, chunk_documents, policy_for
from src.ingestion.parser import Document, load_documents

CORPUS = Path(__file__).resolve().parents[1] / "data" / "diagnostic_documents"


def mk(body, doc_id="D", doc_type="error_log", **kw):
    return Document(doc_id=doc_id, doc_type=doc_type, source="mock",
                    body=body, origin=f"{doc_id}.txt", **kw)


def texts(chunks):
    return [c.text for c in chunks]


def test_short_doc_is_single_chunk():
    chunks = chunk_document(mk("abc"))
    assert len(chunks) == 1
    c = chunks[0]
    assert (c.chunk_id, c.index, c.text, c.start_char, c.end_char) == ("D#0", 0, "abc", 0, 3)


def test_packing_without_overlap():
    chunks = chunk_document(mk("aaaa\nbbbb\ncccc\ndddd"), max_chars=10, overlap_lines=0)
    assert texts(chunks) == ["aaaa\nbbbb", "cccc\ndddd"]


def test_packing_with_overlap():
    chunks = chunk_document(mk("aaaa\nbbbb\ncccc\ndddd"), max_chars=10, overlap_lines=1)
    assert texts(chunks) == ["aaaa\nbbbb", "bbbb\ncccc", "cccc\ndddd"]


def test_long_line_is_hard_split():
    body = "x" * 25
    chunks = chunk_document(mk(body), max_chars=10, overlap_lines=0)
    assert [len(c.text) for c in chunks] == [10, 10, 5]
    assert "".join(texts(chunks)) == body


def test_empty_and_whitespace_bodies_give_no_chunks():
    assert chunk_document(mk("")) == []
    assert chunk_document(mk("  \n\n \t ")) == []


def test_blank_lines_between_lines_stay_inside_the_slice():
    body = "aaaa\n\n\nbbbb"
    chunks = chunk_document(mk(body), max_chars=20)
    assert len(chunks) == 1
    assert chunks[0].text == body


def test_metadata_copied_to_every_chunk():
    ts = datetime(2026, 10, 4, 1, 2, 15, tzinfo=timezone.utc)
    doc = mk("aaaa\nbbbb\ncccc\ndddd", doc_id="ERR-9", run_id="r1004",
             job_id="j_orders", timestamp=ts, title="T")
    chunks = chunk_document(doc, max_chars=10, overlap_lines=0)
    assert len(chunks) == 2
    for c in chunks:
        assert (c.doc_id, c.run_id, c.job_id, c.timestamp, c.title) == (
            "ERR-9", "r1004", "j_orders", ts, "T")
        assert (c.source, c.origin, c.doc_type) == ("mock", "ERR-9.txt", "error_log")


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        chunk_document(mk("abc"), max_chars=0)
    with pytest.raises(ValueError):
        chunk_document(mk("abc"), overlap_lines=-1)


def test_policy_defaults():
    assert policy_for("error_log") == (500, 0)
    assert policy_for("troubleshooting") == (300, 1)
    assert policy_for("something_new") == (500, 0)


# --- real corpus ------------------------------------------------------------

@pytest.fixture(scope="module")
def docs():
    return load_documents(CORPUS).docs


def test_corpus_chunk_ids_unique_and_sequential(docs):
    chunks = chunk_documents(docs)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert {c.doc_id for c in chunks} == {d.doc_id for d in docs}
    for d in docs:
        mine = [c for c in chunks if c.doc_id == d.doc_id]
        assert [c.index for c in mine] == list(range(len(mine)))


def test_corpus_invariants_with_small_chunks(docs):
    by_id = {d.doc_id: d for d in docs}
    for d in docs:
        chunks = chunk_document(d, max_chars=120, overlap_lines=1)
        assert chunks, d.doc_id
        covered = set()
        for c in chunks:
            assert c.text == by_id[c.doc_id].body[c.start_char:c.end_char]   # traceable
            assert 0 < len(c.text) <= 120                                    # size respected
            covered.update(range(c.start_char, c.end_char))
        for idx, ch in enumerate(d.body):                                    # nothing lost
            if not ch.isspace():
                assert idx in covered, f"{d.doc_id}: char {idx} not in any chunk"