from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Optional, Sequence

from src.ingestion.cleaner import tokenize
from src.ingestion.parser import Document


@dataclass(frozen=True)
class SearchResult:
    rank: int                       # 1 = best
    score: float                    # cosine similarity, 0 < score <= 1
    doc: Document                   # original, unmodified document (provenance intact)
    matched_terms: tuple[str, ...]  # query terms found in this document


def _doc_text(doc: Document) -> str:
    return f"{doc.title or ''}\n{doc.body}"


def _tf(count: int) -> float:
    return 1.0 + math.log(count)


class TfidfIndex:
    def __init__(self, docs: Sequence[Document]) -> None:
        ids = [d.doc_id for d in docs]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate doc_id in documents")

        self._docs = list(docs)
        token_lists = [tokenize(_doc_text(d)) for d in self._docs]

        n = len(self._docs)
        df: Counter[str] = Counter()
        for tokens in token_lists:
            df.update(set(tokens))  # set(): count each word once per document

        self._idf = {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}
        self._vectors = [self._weigh(tokens) for tokens in token_lists]

    @property
    def size(self) -> int:
        return len(self._docs)

    def idf(self, term: str) -> Optional[float]:
        return self._idf.get(term)

    def _weigh(self, tokens: list[str]) -> dict[str, float]:
        """Tokens -> L2-normalised {term: tf*idf}. Unknown terms are dropped."""
        counts = Counter(tokens)
        raw = {t: _tf(c) * self._idf[t] for t, c in counts.items() if t in self._idf}
        norm = math.sqrt(sum(w * w for w in raw.values()))
        if norm == 0.0:
            return {}
        return {t: w / norm for t, w in raw.items()}

    def search(
        self,
        query: str,
        top_k: int = 3,
        doc_type: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> list[SearchResult]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        qvec = self._weigh(tokenize(query))
        if not qvec:
            return []  # empty index, stopword-only query, or no known words

        scored: list[tuple[float, Document, tuple[str, ...]]] = []
        for doc, dvec in zip(self._docs, self._vectors):
            if doc_type is not None and doc.doc_type != doc_type:
                continue
            if run_id is not None and doc.run_id != run_id:
                continue
            score = sum(w * dvec.get(t, 0.0) for t, w in qvec.items())
            if score > 0.0:
                matched = tuple(sorted(t for t in qvec if t in dvec))
                scored.append((score, doc, matched))

        scored.sort(key=lambda s: (-s[0], s[1].doc_id))
        return [
            SearchResult(rank=i, score=s, doc=d, matched_terms=m)
            for i, (s, d, m) in enumerate(scored[:top_k], start=1)
        ]