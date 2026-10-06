from __future__ import annotations

import hashlib
import warnings
from typing import Optional, Protocol, Sequence

import numpy as np

from src.ingestion.cleaner import tokenize

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingError(RuntimeError):
    """The embedding model could not be loaded or run."""


class Embedder(Protocol):
    model_name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an array of shape (len(texts), dim), float32, rows L2-normalised
        (or all-zero when the text has no usable content)."""
        ...


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to length 1. All-zero rows stay zero (no NaN)."""
    m = np.asarray(matrix, dtype=np.float32)
    if m.ndim != 2:
        raise ValueError(f"expected a 2-D array, got {m.ndim}-D")
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    safe = np.where(norms == 0, 1.0, norms)
    return (m / safe).astype(np.float32, copy=False)


def _check_texts(texts: Sequence[str]) -> list[str]:
    if isinstance(texts, (str, bytes)):
        raise TypeError("texts must be a sequence of strings, not a single string")
    out = list(texts)
    for i, t in enumerate(out):
        if not isinstance(t, str):
            raise TypeError(f"texts[{i}] must be a str, got {type(t).__name__}")
        if not t.strip():
            raise ValueError(f"texts[{i}] is blank")
    return out


class HashingEmbedder:
    """Offline fallback. Hashed bag-of-words: lexical overlap only, NOT semantic."""

    def __init__(self, dim: int = 256) -> None:
        if dim < 1:
            raise ValueError("dim must be at least 1")
        self.dim = dim
        self.model_name = f"hashing-v1-{dim}"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        items = _check_texts(texts)
        out = np.zeros((len(items), self.dim), dtype=np.float32)
        for row, text in enumerate(items):
            for token in tokenize(text):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % self.dim
                out[row, index] += 1.0 if digest[4] % 2 == 0 else -1.0
        return normalize_rows(out)


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 32,
        device: Optional[str] = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.model_name = model_name
        self.batch_size = batch_size
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name, device=device)
            getter = getattr(self._model, "get_embedding_dimension", None) \
                or self._model.get_sentence_embedding_dimension
            self.dim = int(getter())
        except Exception as exc:  # download, network, disk, bad name, missing package
            raise EmbeddingError(
                f"could not load embedding model {model_name!r}: {exc}. "
                "Check your internet connection (first run downloads the model), "
                "free disk space, and the model name."
            ) from exc

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        items = _check_texts(texts)
        if not items:
            return np.zeros((0, self.dim), dtype=np.float32)
        try:
            vectors = self._model.encode(
                items,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise EmbeddingError(f"embedding failed: {exc}") from exc
        vectors = np.asarray(vectors)
        if vectors.shape != (len(items), self.dim):
            raise EmbeddingError(
                f"unexpected embedding shape {vectors.shape}, expected {(len(items), self.dim)}"
            )
        return normalize_rows(vectors)


def load_embedder(allow_fallback: bool = False, **kwargs) -> Embedder:
    """Real model by default. With allow_fallback=True, a load failure gives a loud
    warning and a lexical HashingEmbedder instead of an exception."""
    try:
        return SentenceTransformerEmbedder(**kwargs)
    except EmbeddingError as exc:
        if not allow_fallback:
            raise
        warnings.warn(
            f"falling back to HashingEmbedder (lexical, NOT semantic): {exc}",
            UserWarning,
            stacklevel=2,
        )
        return HashingEmbedder()