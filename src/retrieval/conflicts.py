from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from src.ingestion.parser import Document
from src.monitoring.models import RunRecord
from src.monitoring.queries import FAILURE_STATES
from src.retrieval.search import RetrievalReport

# catalog.schema.object, e.g. main.crm.customers
_OBJECT = re.compile(r"\b[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*\b", re.IGNORECASE)


def object_names(text: str) -> frozenset[str]:
    return frozenset(m.group().lower() for m in _OBJECT.finditer(text))


@dataclass(frozen=True)
class Conflict:
    doc_id: str
    origin: str
    claim_time: datetime
    run_id: str
    run_start: datetime
    shared_objects: tuple[str, ...]
    run_error: str


def find_conflicts(docs: Sequence[Document], runs: Sequence[RunRecord]) -> list[Conflict]:
    found: list[Conflict] = []
    for d in docs:
        if d.claim != "resolved" or d.timestamp is None:
            continue
        doc_objects = object_names(f"{d.title or ''}\n{d.body}")
        if not doc_objects:
            continue
        for r in runs:
            if r.start_time is None or r.start_time <= d.timestamp:
                continue
            if r.result_state not in FAILURE_STATES or not r.error_message:
                continue
            shared = doc_objects & object_names(r.error_message)
            if shared:
                found.append(Conflict(
                    doc_id=d.doc_id, origin=d.origin, claim_time=d.timestamp,
                    run_id=r.run_id, run_start=r.start_time,
                    shared_objects=tuple(sorted(shared)), run_error=r.error_message,
                ))
    return sorted(found, key=lambda c: (c.doc_id, c.run_start, c.run_id))


def relevant_conflicts(report: RetrievalReport, conflicts: Sequence[Conflict]) -> tuple[Conflict, ...]:
    """Only conflicts about documents that are actually shown in this report."""
    shown = {r.chunk.doc_id for rs in report.run_evidence.values() for r in rs}
    shown |= {r.chunk.doc_id for r in report.related.results}
    return tuple(c for c in conflicts if c.doc_id in shown)