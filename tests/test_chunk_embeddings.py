import dataclasses
import os

import numpy as np
import pytest

from src.ingestion.chunker import chunk_document, chunk_documents
from src.ingestion.parser import Document
from src.retrieval.chunk_embeddings import (
    EmbeddingCache,
    chunk_embedding_text,
    embed_chunks,
)
from src.retrieval.embeddings import HashingEmbedder


class Counting(HashingEmbedder):
    """HashingEmbedder that records every text it is asked to embed."""

    def __init__(self, dim=64):
        super().__init__(dim)
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return super().embed(texts)

    def total(self):
        return sum(len(c) for c in self.calls)


def mk(doc_id, body, title=None):
    return Document(doc_id=doc_id, doc_type="error_log", source="mock",
                    body=body, title=title, origin=f"{doc_id}.txt")


def three_chunks():
    doc = mk("D", "aaa bbb\nccc ddd\neee fff")
    chunks = chunk_document(doc, max_chars=10, overlap_lines=0)
    assert len(chunks) == 3
    return chunks


def test_text_includes_title_when_present():
    c = chunk_document(mk("D", "body line", title="My title"))[0]
    assert chunk_embedding_text(c) == "My title\nbody line"


def test_text_without_title():
    c = chunk_document(mk("D", "body line"))[0]
    assert chunk_embedding_text(c) == "body line"


def test_rows_align_with_chunks():
    e = Counting()
    chunks = three_chunks()
    out = embed_chunks(chunks, e)
    assert out.shape == (3, 64)
    for i, c in enumerate(chunks):
        assert np.array_equal(out[i], e.embed([chunk_embedding_text(c)])[0])


def test_empty_chunks_shape():
    out = embed_chunks([], Counting(dim=64))
    assert out.shape == (0, 64)


def test_cache_second_call_makes_no_model_calls():
    e = Counting()
    cache = EmbeddingCache(e.model_name, e.dim)
    chunks = three_chunks()
    first = embed_chunks(chunks, e, cache)
    calls_after_first = len(e.calls)
    second = embed_chunks(chunks, e, cache)
    assert len(e.calls) == calls_after_first
    assert np.array_equal(first, second)
    assert (cache.misses, cache.hits) == (3, 3)


def test_duplicate_texts_embedded_once():
    e = Counting()
    chunks = chunk_documents([mk("A", "same text here"), mk("B", "same text here")])
    out = embed_chunks(chunks, e)
    assert e.total() == 1
    assert np.array_equal(out[0], out[1])


def test_only_changed_chunk_is_reembedded():
    e = Counting()
    cache = EmbeddingCache(e.model_name, e.dim)
    chunks = three_chunks()
    embed_chunks(chunks, e, cache)
    edited = list(chunks)
    edited[1] = dataclasses.replace(chunks[1], text="zzz new text")
    embed_chunks(edited, e, cache)
    assert e.calls[-1] == ["zzz new text"]


def test_cache_model_mismatch_raises():
    e = Counting()
    with pytest.raises(ValueError, match="cache is for"):
        embed_chunks(three_chunks(), e, EmbeddingCache("other-model", e.dim))


def test_cache_save_load_roundtrip(tmp_path):
    p = tmp_path / "sub" / "c.npz"
    v = np.array([1, 0, 0, 0], dtype=np.float32)
    c = EmbeddingCache("m", 4)
    c.put("hello", v)
    c.save(p)
    assert sorted(os.listdir(p.parent)) == ["c.npz"]   # no leftover temp file
    c2 = EmbeddingCache.load(p, "m", 4)
    assert len(c2) == 1
    assert np.array_equal(c2.get("hello"), v)


def test_load_wrong_model_raises(tmp_path):
    p = tmp_path / "c.npz"
    EmbeddingCache("m", 4).save(p)
    with pytest.raises(ValueError, match="model"):
        EmbeddingCache.load(p, "other", 4)


def test_load_wrong_dim_raises(tmp_path):
    p = tmp_path / "c.npz"
    EmbeddingCache("m", 4).save(p)
    with pytest.raises(ValueError, match="dim"):
        EmbeddingCache.load(p, "m", 8)


def test_load_corrupt_file_raises(tmp_path):
    p = tmp_path / "c.npz"
    p.write_bytes(b"not an npz file")
    with pytest.raises(ValueError, match="unreadable"):
        EmbeddingCache.load(p, "m", 4)


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        EmbeddingCache.load(tmp_path / "nope.npz", "m", 4)


def test_put_wrong_shape_raises():
    with pytest.raises(ValueError):
        EmbeddingCache("m", 4).put("x", np.zeros(5, dtype=np.float32))