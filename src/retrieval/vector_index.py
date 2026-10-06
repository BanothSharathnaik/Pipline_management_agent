from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from src.ingestion.chunker import Chunk
from src.retrieval.embeddings import Embedder, normalize_rows


@dataclass(frozen=True)
class VectorResult:
    rank: int       # 1 = best
    score: float    # cosine similarity in [-1, 1]; NOT a probability
    chunk: Chunk


class VectorIndex:
    def __init__(self, dim: int, model_name: str) -> None:
        if dim < 1:
            raise ValueError("dim must be at least 1")
        self.dim = dim
        self.model_name = model_name
        self._vectors = np.zeros((0, dim), dtype=np.float32)  # unit-length rows
        self._chunks: list[Chunk] = []
        self._ids: set[str] = set()

    @property
    def size(self) -> int:
        return len(self._chunks)

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        v = np.asarray(vectors)
        if len(chunks) == 0 and v.size == 0:
            return
        if v.ndim != 2:
            raise ValueError(f"vectors must be 2-D, got {v.ndim}-D")
        if v.shape[1] != self.dim:
            raise ValueError(f"dimension mismatch: index is {self.dim}, vectors are {v.shape[1]}")
        if v.shape[0] != len(chunks):
            raise ValueError(f"got {len(chunks)} chunks but {v.shape[0]} vectors")
        v = v.astype(np.float32)
        if not np.isfinite(v).all():
            raise ValueError("vectors contain NaN or infinity")

        ids = [c.chunk_id for c in chunks]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate chunk_id within the batch")
        clashes = [i for i in ids if i in self._ids]
        if clashes:
            raise ValueError(f"chunk_id already in index: {clashes[:3]}")

        zero_rows = np.nonzero(np.linalg.norm(v, axis=1) == 0.0)[0]
        if zero_rows.size:
            names = [ids[i] for i in zero_rows[:3]]
            raise ValueError(f"zero vector(s) cannot be indexed, e.g. chunk_id {names}")

        # every check passed: now mutate
        self._vectors = np.vstack([self._vectors, normalize_rows(v)])
        self._chunks.extend(chunks)
        self._ids.update(ids)

    def search(
        self,
        query: np.ndarray,
        top_k: int = 3,
        min_score: Optional[float] = None,
        doc_type: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> list[VectorResult]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        q = np.asarray(query, dtype=np.float32)
        if q.shape != (self.dim,):
            raise ValueError(f"query shape {q.shape} does not match index dim {self.dim}")
        if not np.isfinite(q).all():
            raise ValueError("query contains NaN or infinity")
        norm = float(np.linalg.norm(q))
        if norm == 0.0 or not self._chunks:
            return []
        q = q / norm

        candidates = [
            i for i, c in enumerate(self._chunks)
            if (doc_type is None or c.doc_type == doc_type)
            and (run_id is None or c.run_id == run_id)
        ]
        if not candidates:
            return []
        cand = np.asarray(candidates)

        scores = self._vectors[cand] @ q          # unit vectors: dot == cosine
        n = scores.size
        if top_k < n:
            kth = np.partition(scores, n - top_k)[n - top_k]   # k-th largest value
            keep = np.nonzero(scores >= kth)[0]                # includes exact ties
        else:
            keep = np.arange(n)
        if min_score is not None:
            keep = keep[scores[keep] >= min_score]

        order = sorted(
            keep.tolist(),
            key=lambda j: (-float(scores[j]), self._chunks[int(cand[j])].chunk_id),
        )[:top_k]
        return [
            VectorResult(rank=r, score=float(scores[j]), chunk=self._chunks[int(cand[j])])
            for r, j in enumerate(order, start=1)
        ]

    def search_text(self, text: str, embedder: Embedder, **kwargs) -> list[VectorResult]:
        if embedder.model_name != self.model_name or embedder.dim != self.dim:
            raise ValueError(
                f"index was built with {self.model_name!r} (dim {self.dim}), "
                f"embedder is {embedder.model_name!r} (dim {embedder.dim})"
            )
        return self.search(embedder.embed([text])[0], **kwargs)