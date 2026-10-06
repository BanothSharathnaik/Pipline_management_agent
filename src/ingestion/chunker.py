from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Sequence

from src.ingestion.parser import Document

# (max_chars, overlap_lines). Starting values; tuned by measurement in Phase 11.
DEFAULT_POLICY = (500, 0)
POLICY = {
    "error_log": (500, 0),
    "historical_failure": (500, 0),
    "data_quality": (500, 0),
    "troubleshooting": (300, 1),
}

_LINE = re.compile(r"[^\n]+")


def policy_for(doc_type: str) -> tuple[int, int]:
    return POLICY.get(doc_type, DEFAULT_POLICY)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str              # "<doc_id>#<index>"
    doc_id: str
    index: int
    text: str                  # always equals document.body[start_char:end_char]
    start_char: int
    end_char: int
    doc_type: str
    source: str
    origin: str
    title: Optional[str] = None
    run_id: Optional[str] = None
    job_id: Optional[str] = None
    timestamp: Optional[datetime] = None


def _units(body: str, max_chars: int) -> list[tuple[int, int]]:
    """Non-blank lines as (start, end) offsets. Over-long lines are hard-split."""
    units: list[tuple[int, int]] = []
    for m in _LINE.finditer(body):
        raw = m.group()
        start = m.start() + (len(raw) - len(raw.lstrip()))
        end = m.end() - (len(raw) - len(raw.rstrip()))
        if end <= start:
            continue  # whitespace-only line
        for s in range(start, end, max_chars):
            units.append((s, min(s + max_chars, end)))
    return units


def chunk_document(
    doc: Document,
    max_chars: Optional[int] = None,
    overlap_lines: Optional[int] = None,
) -> list[Chunk]:
    default_max, default_overlap = policy_for(doc.doc_type)
    max_chars = default_max if max_chars is None else max_chars
    overlap_lines = default_overlap if overlap_lines is None else overlap_lines
    if max_chars < 1:
        raise ValueError("max_chars must be at least 1")
    if overlap_lines < 0:
        raise ValueError("overlap_lines must be >= 0")

    units = _units(doc.body, max_chars)
    if not units:
        return []

    spans: list[tuple[int, int]] = []
    n = len(units)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and units[j + 1][1] - units[i][0] <= max_chars:
            j += 1
        spans.append((units[i][0], units[j][1]))
        if j + 1 >= n:
            break
        nxt = max(j + 1 - overlap_lines, i + 1)
        # shrink overlap until the next unit fits; guarantees progress
        while nxt < j + 1 and units[j + 1][1] - units[nxt][0] > max_chars:
            nxt += 1
        i = nxt

    return [
        Chunk(
            chunk_id=f"{doc.doc_id}#{k}",
            doc_id=doc.doc_id,
            index=k,
            text=doc.body[s:e],
            start_char=s,
            end_char=e,
            doc_type=doc.doc_type,
            source=doc.source,
            origin=doc.origin,
            title=doc.title,
            run_id=doc.run_id,
            job_id=doc.job_id,
            timestamp=doc.timestamp,
        )
        for k, (s, e) in enumerate(spans)
    ]


def chunk_documents(docs: Sequence[Document]) -> list[Chunk]:
    ids = [d.doc_id for d in docs]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate doc_id in documents")
    chunks: list[Chunk] = []
    for d in docs:
        chunks.extend(chunk_document(d))
    return chunks