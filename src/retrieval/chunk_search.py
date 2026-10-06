from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from src.ingestion.chunker import Chunk
from src.ingestion.parser import Document
from src.retrieval.keyword_search import TfidfIndex


@dataclass(frozen=True)
class ChunkResult:
    rank: int
    score: float
    chunk: Chunk
    matched_terms: tuple[str, ...]


def _as_document(c: Chunk) -> Document:
    """Wrap a chunk so the existing TF-IDF index can score it. Never shown to users."""
    return Document(
        doc_id=c.chunk_id,
        doc_type=c.doc_type,
        source=c.source,
        body=c.text,
        title=c.title,
        run_id=c.run_id,
        job_id=c.job_id,
        timestamp=c.timestamp,
        origin=c.origin,
    )


class ChunkIndex:
    def __init__(self, chunks: Sequence[Chunk]) -> None:
        ids = [c.chunk_id for c in chunks]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate chunk_id in chunks")
        self._by_id = {c.chunk_id: c for c in chunks}
        self._index = TfidfIndex([_as_document(c) for c in chunks])

    @property
    def size(self) -> int:
        return len(self._by_id)

    def search(
        self,
        query: str,
        top_k: int = 3,
        doc_type: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> list[ChunkResult]:
        hits = self._index.search(query, top_k=top_k, doc_type=doc_type, run_id=run_id)
        return [
            ChunkResult(
                rank=h.rank,
                score=h.score,
                chunk=self._by_id[h.doc.doc_id],
                matched_terms=h.matched_terms,
            )
            for h in hits
        ]