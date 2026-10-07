from __future__ import annotations

import dataclasses
from collections import Counter
from dataclasses import dataclass
from typing import Optional, Sequence

from src.retrieval.vector_index import VectorResult

# Cosine similarity threshold for vector retrieval (min_score = 0.5)
DEFAULT_MIN_SCORE = 0.5


@dataclass(frozen=True)
class EvidenceSet:
    status: str                       # "ok" or "no_evidence"
    results: tuple[VectorResult, ...]
    min_score: float
    best_score: Optional[float]       # best score seen BEFORE filtering; None if no candidates
    dropped_low_score: int
    dropped_per_doc: int


def select_evidence(
    results: Sequence[VectorResult],
    min_score: float = DEFAULT_MIN_SCORE,
    max_per_doc: int = 2,
    limit: int = 5,
) -> EvidenceSet:
    """Turn raw search hits into the evidence we are willing to show.
    Order: drop below min_score, cap chunks per document, cut to limit, renumber ranks."""
    if max_per_doc < 1:
        raise ValueError("max_per_doc must be at least 1")
    if limit < 1:
        raise ValueError("limit must be at least 1")

    ordered = sorted(results, key=lambda r: (-r.score, r.chunk.chunk_id))
    best = ordered[0].score if ordered else None

    above = [r for r in ordered if r.score >= min_score]
    dropped_low = len(ordered) - len(above)

    per_doc: Counter[str] = Counter()
    kept: list[VectorResult] = []
    dropped_cap = 0
    for r in above:
        if per_doc[r.chunk.doc_id] >= max_per_doc:
            dropped_cap += 1
            continue
        per_doc[r.chunk.doc_id] += 1
        kept.append(r)

    kept = kept[:limit]
    final = tuple(dataclasses.replace(r, rank=i) for i, r in enumerate(kept, start=1))
    return EvidenceSet(
        status="ok" if final else "no_evidence",
        results=final,
        min_score=min_score,
        best_score=best,
        dropped_low_score=dropped_low,
        dropped_per_doc=dropped_cap,
    )