from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.monitoring.models import _parse_time  # same timezone-strict parser as runs

VALID_DOC_TYPES = frozenset(
    {"error_log", "historical_failure", "troubleshooting", "data_quality"}
)
VALID_SOURCES = frozenset({"mock", "live"})
VALID_CLAIMS = frozenset({"resolved"})
REQUIRED_KEYS = ("doc_id", "doc_type", "source")
OPTIONAL_KEYS = ("title", "run_id", "job_id", "timestamp", "claim")
SEPARATOR = "---"


@dataclass(frozen=True)
class Document:
    doc_id: str
    doc_type: str
    source: str
    body: str
    title: Optional[str] = None
    run_id: Optional[str] = None
    job_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    claim: Optional[str] = None
    origin: str = "<memory>"  # file name, for traceability


@dataclass(frozen=True)
class DocIssue:
    filename: str
    message: str


@dataclass
class DocLoadResult:
    docs: list[Document] = field(default_factory=list)
    issues: list[DocIssue] = field(default_factory=list)


def parse_document(text: str, origin: str = "<memory>") -> Document:
    """Parse 'key: value' header lines, a '---' line, then the body.
    Raises ValueError with a specific message for any malformed input."""
    lines = text.replace("\r\n", "\n").split("\n")

    sep_index = next((i for i, ln in enumerate(lines) if ln.strip() == SEPARATOR), None)
    if sep_index is None:
        raise ValueError("missing separator line '---' between header and body")

    header: dict[str, str] = {}
    for ln in lines[:sep_index]:
        if not ln.strip():
            continue
        key, colon, value = ln.partition(":")
        key = key.strip().lower()
        if not colon or not key:
            raise ValueError(f"malformed header line: {ln!r}")
        if key not in REQUIRED_KEYS and key not in OPTIONAL_KEYS:
            raise ValueError(f"unknown header key: {key!r}")
        if key in header:
            raise ValueError(f"duplicate header key: {key!r}")
        header[key] = value.strip()

    for key in REQUIRED_KEYS:
        if not header.get(key):
            raise ValueError(f"missing required header: {key}")

    if header["doc_type"] not in VALID_DOC_TYPES:
        raise ValueError(f"doc_type must be one of {sorted(VALID_DOC_TYPES)}, got {header['doc_type']!r}")
    if header["source"] not in VALID_SOURCES:
        raise ValueError(f"source must be one of {sorted(VALID_SOURCES)}, got {header['source']!r}")

    claim = header.get("claim")
    if claim and claim not in VALID_CLAIMS:
        raise ValueError(f"claim must be one of {sorted(VALID_CLAIMS)}, got {claim!r}")

    body = "\n".join(lines[sep_index + 1:]).strip()
    if not body:
        raise ValueError("document body is empty")

    return Document(
        doc_id=header["doc_id"],
        doc_type=header["doc_type"],
        source=header["source"],
        body=body,
        title=header.get("title") or None,
        run_id=header.get("run_id") or None,
        job_id=header.get("job_id") or None,
        timestamp=_parse_time(header.get("timestamp")),
        claim=claim or None,
        origin=origin,
    )


def load_documents(directory: str | Path) -> DocLoadResult:
    """Load every *.txt file in a directory. Bad files are reported, never skipped silently."""
    d = Path(directory)
    if not d.is_dir():
        raise FileNotFoundError(f"document directory not found: {d}")

    result = DocLoadResult()
    seen: set[str] = set()

    for path in sorted(d.glob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            result.issues.append(DocIssue(path.name, "file is not valid UTF-8"))
            continue
        try:
            doc = parse_document(text, origin=path.name)
        except ValueError as exc:
            result.issues.append(DocIssue(path.name, str(exc)))
            continue
        if doc.doc_id in seen:
            result.issues.append(DocIssue(path.name, f"duplicate doc_id {doc.doc_id!r}; file skipped"))
            continue
        seen.add(doc.doc_id)
        result.docs.append(doc)

    return result