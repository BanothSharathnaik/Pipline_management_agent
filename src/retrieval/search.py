from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from src.retrieval.embeddings import Embedder
from src.retrieval.evidence import DEFAULT_MIN_SCORE, EvidenceSet, select_evidence
from src.retrieval.vector_index import VectorIndex, VectorResult

_RUN_AFTER_WORD = re.compile(r"\brun(?:\s+id)?\s*#?\s*(r?\d[\w-]*)", re.IGNORECASE)
_BARE_RUN = re.compile(r"\br\d+\b", re.IGNORECASE)


def extract_run_ids(question: str) -> list[str]:
    """Run IDs mentioned in a question: lower-cased, de-duplicated, in order of appearance."""
    if not isinstance(question, str):
        raise TypeError("question must be a str")
    found: dict[str, int] = {}
    for pattern in (_RUN_AFTER_WORD, _BARE_RUN):
        for m in pattern.finditer(question):
            rid = (m.group(1) if m.groups() else m.group(0)).lower()
            pos = m.start(1) if m.groups() else m.start()
            found[rid] = min(pos, found.get(rid, pos))
    return [rid for rid, _ in sorted(found.items(), key=lambda kv: kv[1])]


@dataclass(frozen=True)
class RetrievalReport:
    question: str
    run_ids_mentioned: tuple[str, ...]
    unknown_run_ids: tuple[str, ...]
    run_evidence: dict[str, tuple[VectorResult, ...]]  # linked by run_id metadata
    related: EvidenceSet                               # linked by similarity only
    notes: tuple[str, ...]


class RetrievalPipeline:
    def __init__(
        self,
        index: VectorIndex,
        embedder: Embedder,
        known_run_ids: Iterable[str],
        min_score: float = DEFAULT_MIN_SCORE,
        max_per_doc: int = 2,
    ) -> None:
        if embedder.model_name != index.model_name or embedder.dim != index.dim:
            raise ValueError(
                f"index was built with {index.model_name!r} (dim {index.dim}), "
                f"embedder is {embedder.model_name!r} (dim {embedder.dim})"
            )
        self._index = index
        self._embedder = embedder
        self._known = {r.lower(): r for r in known_run_ids}  # lower-case -> canonical id
        self._min_score = min_score
        self._max_per_doc = max_per_doc

    def retrieve(
        self, question: str, top_k: int = 5, related_query: Optional[str] = None
    ) -> RetrievalReport:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        mentioned = extract_run_ids(question)
        known = [self._known[r] for r in mentioned if r in self._known]
        unknown = [r for r in mentioned if r not in self._known]
        notes = [
            f"run {r} not found in monitoring data; its status and errors cannot be established"
            for r in unknown
        ]

        if mentioned and not known:
            notes.append("semantic search skipped: no mentioned run exists in the monitoring data")
            return RetrievalReport(
                question=question,
                run_ids_mentioned=tuple(mentioned),
                unknown_run_ids=tuple(unknown),
                run_evidence={},
                related=EvidenceSet("no_evidence", (), self._min_score, None, 0, 0),
                notes=tuple(notes),
            )

        text = related_query if related_query and related_query.strip() else question
        qvec = self._embedder.embed([text])[0]
        if float(np.linalg.norm(qvec)) == 0.0:
            raise ValueError("question has no searchable content")

        run_evidence: dict[str, tuple[VectorResult, ...]] = {}
        linked_ids: set[str] = set()
        for rid in known:
            hits = self._index.search(qvec, top_k=top_k, run_id=rid)
            kept = select_evidence(hits, min_score=-1.0, max_per_doc=top_k, limit=top_k).results
            run_evidence[rid] = kept
            linked_ids.update(h.chunk.chunk_id for h in kept)
            if not kept:
                notes.append(f"no diagnostic documents are linked to run {rid}")

        hits = self._index.search(qvec, top_k=top_k + len(linked_ids))
        hits = [h for h in hits if h.chunk.chunk_id not in linked_ids]
        related = select_evidence(
            hits, min_score=self._min_score, max_per_doc=self._max_per_doc, limit=top_k
        )
        return RetrievalReport(
            question=question,
            run_ids_mentioned=tuple(mentioned),
            unknown_run_ids=tuple(unknown),
            run_evidence=run_evidence,
            related=related,
            notes=tuple(notes),
        )


def _line(r: VectorResult) -> str:
    c = r.chunk
    ts = c.timestamp.isoformat() if c.timestamp else "n/a"
    text = " ".join(c.text.split())
    if len(text) > 160:
        text = text[:157] + "..."
    return (
        f"  {r.rank}. {c.chunk_id}  score={r.score:.3f}  run={c.run_id or 'n/a'}  "
        f"source={c.source}  file={c.origin}  chars={c.start_char}-{c.end_char}  ts={ts}\n"
        f"       {text}"
    )


def format_report(rep: RetrievalReport) -> str:
    lines = [f"Question: {rep.question}"]
    if rep.run_ids_mentioned:
        lines.append("Run IDs mentioned: " + ", ".join(rep.run_ids_mentioned))
    lines.extend(f"NOTE: {n}" for n in rep.notes)

    for rid, results in rep.run_evidence.items():
        lines.append(f"\nEvidence linked to run {rid} (by run_id metadata, not similarity):")
        lines.extend(_line(r) for r in results) if results else lines.append("  (none)")

    lines.append("\nRelated evidence (similarity only; may belong to OTHER runs; not proof):")
    if rep.related.results:
        lines.extend(_line(r) for r in rep.related.results)
    else:
        best = rep.related.best_score
        detail = f" (best match scored {best:.3f})" if best is not None else ""
        lines.append(f"  no evidence above min score {rep.related.min_score:.2f}{detail}")
    return "\n".join(lines)