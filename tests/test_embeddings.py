import numpy as np
import pytest

from src.retrieval import embeddings as emb
from src.retrieval.embeddings import (
    EmbeddingError,
    HashingEmbedder,
    normalize_rows,
)


def cos(a, b):
    return float(np.dot(a, b))  # rows are unit length, so dot == cosine


# --- normalize_rows ---------------------------------------------------------

def test_normalize_rows_unit_length():
    out = normalize_rows(np.array([[3.0, 4.0], [0.0, 2.0]]))
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)
    assert np.allclose(out[0], [0.6, 0.8])


def test_normalize_rows_zero_row_stays_zero():
    out = normalize_rows(np.array([[0.0, 0.0], [1.0, 0.0]]))
    assert not np.isnan(out).any()
    assert np.allclose(out[0], 0.0)
    assert np.allclose(out[1], [1.0, 0.0])


def test_normalize_rows_rejects_1d():
    with pytest.raises(ValueError):
        normalize_rows(np.array([1.0, 2.0]))


# --- HashingEmbedder --------------------------------------------------------

def test_hashing_shape_dtype_norm():
    out = HashingEmbedder(dim=64).embed(["permission denied table", "disk full node"])
    assert out.shape == (2, 64)
    assert out.dtype == np.float32
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_hashing_deterministic_across_instances():
    a = HashingEmbedder(dim=128).embed(["permission denied table"])
    b = HashingEmbedder(dim=128).embed(["permission denied table"])
    assert np.array_equal(a, b)


def test_hashing_identical_text_cosine_one():
    v = HashingEmbedder(dim=256).embed(["permission denied table"] * 2)
    assert cos(v[0], v[1]) == pytest.approx(1.0, abs=1e-5)


def test_hashing_shared_tokens_more_similar():
    e = HashingEmbedder(dim=4096)
    v = e.embed(["permission denied table", "permission denied customers", "disk full node"])
    assert cos(v[0], v[1]) > 0.5
    assert cos(v[0], v[2]) < 0.5


def test_hashing_empty_list():
    out = HashingEmbedder(dim=32).embed([])
    assert out.shape == (0, 32)


def test_hashing_stopword_only_is_zero_vector():
    out = HashingEmbedder(dim=32).embed(["the of and"])
    assert np.allclose(out, 0.0)


def test_row_order_matches_input_order():
    e = HashingEmbedder(dim=128)
    texts = ["permission denied table", "disk full node", "timeout seconds"]
    batch = e.embed(texts)
    for i, t in enumerate(texts):
        assert np.array_equal(batch[i], e.embed([t])[0])


def test_single_string_rejected():
    with pytest.raises(TypeError):
        HashingEmbedder().embed("not a list")


def test_blank_text_rejected():
    with pytest.raises(ValueError):
        HashingEmbedder().embed(["ok text", "   "])


def test_non_string_element_rejected():
    with pytest.raises(TypeError):
        HashingEmbedder().embed(["ok", 5])


def test_hashing_bad_dim():
    with pytest.raises(ValueError):
        HashingEmbedder(dim=0)


# --- loader and fallback ----------------------------------------------------

class _Boom:
    def __init__(self, **kwargs):
        raise EmbeddingError("boom")


def test_load_embedder_failure_raises(monkeypatch):
    monkeypatch.setattr(emb, "SentenceTransformerEmbedder", _Boom)
    with pytest.raises(EmbeddingError):
        emb.load_embedder()


def test_load_embedder_fallback_warns(monkeypatch):
    monkeypatch.setattr(emb, "SentenceTransformerEmbedder", _Boom)
    with pytest.warns(UserWarning, match="NOT semantic"):
        e = emb.load_embedder(allow_fallback=True)
    assert isinstance(e, HashingEmbedder)
    assert e.model_name.startswith("hashing")


# --- real model (slow: run with `pytest -m slow`) ---------------------------

PAIR = [
    "Permission denied reading customers table",
    "Access to the customer data was refused",
    "Disk is full on the node",
]


@pytest.fixture(scope="module")
def real():
    return emb.SentenceTransformerEmbedder()


@pytest.mark.slow
def test_real_shape_and_norm(real):
    out = real.embed(PAIR)
    assert real.dim == 384
    assert out.shape == (3, 384)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-4)


@pytest.mark.slow
def test_real_paraphrase_beats_unrelated(real):
    v = real.embed(PAIR)
    assert cos(v[0], v[1]) > cos(v[0], v[2]) + 0.3
    assert cos(v[0], v[1]) > cos(v[1], v[2]) + 0.3


@pytest.mark.slow
def test_real_batch_matches_single(real):
    batch = real.embed(PAIR)
    for i, t in enumerate(PAIR):
        assert np.allclose(batch[i], real.embed([t])[0], atol=1e-3)


@pytest.mark.slow
def test_real_bad_model_name_raises_embedding_error():
    with pytest.raises(EmbeddingError):
        emb.SentenceTransformerEmbedder(model_name="no-such-org/no-such-model-xyz")