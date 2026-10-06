from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from src.ingestion.chunker import Chunk
from src.retrieval.embeddings import Embedder


def chunk_embedding_text(chunk: Chunk) -> str:
    """The exact text that gets embedded. Single place to change (tested in Phase 11)."""
    return f"{chunk.title}\n{chunk.text}" if chunk.title else chunk.text


def _key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    """text -> vector for ONE model. Refuses to mix vectors from different models."""

    def __init__(self, model_name: str, dim: int) -> None:
        if dim < 1:
            raise ValueError("dim must be at least 1")
        self.model_name = model_name
        self.dim = dim
        self.hits = 0
        self.misses = 0
        self._vecs: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self._vecs)

    def get(self, text: str) -> Optional[np.ndarray]:
        v = self._vecs.get(_key(text))
        if v is None:
            self.misses += 1
            return None
        self.hits += 1
        return v.copy()

    def put(self, text: str, vector: np.ndarray) -> None:
        v = np.asarray(vector, dtype=np.float32)
        if v.shape != (self.dim,):
            raise ValueError(f"vector shape {v.shape} does not match cache dim {self.dim}")
        self._vecs[_key(text)] = v.copy()

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        keys = list(self._vecs)
        vectors = (
            np.stack([self._vecs[k] for k in keys])
            if keys else np.zeros((0, self.dim), dtype=np.float32)
        )
        tmp = p.with_name(p.name + ".tmp")
        with open(tmp, "wb") as f:  # file handle: np.savez would otherwise append ".npz"
            np.savez_compressed(
                f,
                keys=np.array(keys, dtype="<U64"),
                vectors=vectors,
                model_name=np.array(self.model_name),
                dim=np.array(self.dim),
            )
        os.replace(tmp, p)

    @classmethod
    def load(cls, path: str | Path, model_name: str, dim: int) -> "EmbeddingCache":
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"embedding cache not found: {p}")
        try:
            with np.load(p, allow_pickle=False) as data:
                saved_model = str(data["model_name"])
                saved_dim = int(data["dim"])
                keys = data["keys"]
                vectors = data["vectors"]
        except Exception as exc:
            raise ValueError(f"unreadable embedding cache {p}: {exc}") from exc

        if saved_model != model_name:
            raise ValueError(f"cache was built with model {saved_model!r}, not {model_name!r}")
        if saved_dim != dim:
            raise ValueError(f"cache dim is {saved_dim}, expected {dim}")
        if vectors.shape != (len(keys), dim):
            raise ValueError(f"cache is inconsistent: {len(keys)} keys, vectors {vectors.shape}")

        cache = cls(model_name, dim)
        for k, v in zip(keys, vectors):
            cache._vecs[str(k)] = v.astype(np.float32)
        return cache


def embed_chunks(
    chunks: Sequence[Chunk],
    embedder: Embedder,
    cache: Optional[EmbeddingCache] = None,
) -> np.ndarray:
    """Row i of the result is the vector for chunks[i]."""
    if cache is not None and (
        cache.model_name != embedder.model_name or cache.dim != embedder.dim
    ):
        raise ValueError(
            f"cache is for {cache.model_name!r} (dim {cache.dim}), "
            f"embedder is {embedder.model_name!r} (dim {embedder.dim})"
        )

    texts = [chunk_embedding_text(c) for c in chunks]
    if not texts:
        return np.zeros((0, embedder.dim), dtype=np.float32)

    rows: list[Optional[np.ndarray]] = [None] * len(texts)
    missing: dict[str, list[int]] = {}  # text -> positions that need it
    for i, t in enumerate(texts):
        cached = cache.get(t) if cache is not None else None
        if cached is not None:
            rows[i] = cached
        else:
            missing.setdefault(t, []).append(i)

    if missing:
        unique = list(missing)
        vectors = np.asarray(embedder.embed(unique))
        if vectors.shape != (len(unique), embedder.dim):
            raise ValueError(f"embedder returned shape {vectors.shape}")
        for t, v in zip(unique, vectors):
            if cache is not None:
                cache.put(t, v)
            for i in missing[t]:
                rows[i] = v

    return np.stack(rows).astype(np.float32)