from collections import Counter
from pathlib import Path

import pytest

from src.ingestion.parser import load_documents, parse_document
from src.monitoring.collector import load_mock_runs

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "data" / "diagnostic_documents"
MOCK = ROOT / "data" / "mock_runs.json"

VALID = """doc_id: T-1
doc_type: error_log
source: mock
run_id: r1
timestamp: 2026-10-01T02:00:00+00:00
---
Something failed.
"""


def test_parse_valid():
    d = parse_document(VALID, origin="t.txt")
    assert d.doc_id == "T-1"
    assert d.run_id == "r1"
    assert d.timestamp.tzinfo is not None
    assert d.body == "Something failed."
    assert d.origin == "t.txt"


def test_optional_fields_absent():
    d = parse_document("doc_id: T-2\ndoc_type: troubleshooting\nsource: mock\n---\nbody")
    assert d.run_id is None and d.timestamp is None and d.title is None


def test_missing_separator_raises():
    with pytest.raises(ValueError, match="separator"):
        parse_document("doc_id: T\ndoc_type: error_log\nsource: mock\nbody")


def test_missing_required_raises():
    with pytest.raises(ValueError, match="doc_id"):
        parse_document("doc_type: error_log\nsource: mock\n---\nx")


def test_empty_body_raises():
    with pytest.raises(ValueError, match="empty"):
        parse_document("doc_id: T\ndoc_type: error_log\nsource: mock\n---\n   \n")


def test_bad_source_raises():
    with pytest.raises(ValueError, match="source"):
        parse_document("doc_id: T\ndoc_type: error_log\nsource: prod\n---\nx")


def test_bad_doc_type_raises():
    with pytest.raises(ValueError, match="doc_type"):
        parse_document("doc_id: T\ndoc_type: memo\nsource: mock\n---\nx")


def test_naive_timestamp_raises():
    with pytest.raises(ValueError):
        parse_document("doc_id: T\ndoc_type: error_log\nsource: mock\ntimestamp: 2026-10-01T02:00:00\n---\nx")


def test_unknown_key_raises():
    with pytest.raises(ValueError, match="unknown"):
        parse_document("doc_id: T\ndoc_type: error_log\nsource: mock\ncolor: red\n---\nx")


def test_duplicate_key_raises():
    with pytest.raises(ValueError, match="duplicate"):
        parse_document("doc_id: T\ndoc_id: U\ndoc_type: error_log\nsource: mock\n---\nx")


def test_crlf_handled():
    d = parse_document(VALID.replace("\n", "\r\n"))
    assert d.body == "Something failed."


def test_load_ignores_non_txt_and_reports_empty_file(tmp_path):
    (tmp_path / "good.txt").write_text(VALID)
    (tmp_path / "notes.md").write_text("ignored")
    (tmp_path / "empty.txt").write_text("")
    res = load_documents(tmp_path)
    assert [d.doc_id for d in res.docs] == ["T-1"]
    assert [i.filename for i in res.issues] == ["empty.txt"]


def test_duplicate_doc_id_reported(tmp_path):
    (tmp_path / "a.txt").write_text(VALID)
    (tmp_path / "b.txt").write_text(VALID)
    res = load_documents(tmp_path)
    assert len(res.docs) == 1
    assert len(res.issues) == 1 and "duplicate" in res.issues[0].message


def test_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "nope")


def test_real_corpus_loads_clean():
    res = load_documents(DOCS)
    assert res.issues == []
    assert len(res.docs) == 12
    assert {d.source for d in res.docs} == {"mock"}
    assert Counter(d.doc_type for d in res.docs) == {
        "error_log": 6, "historical_failure": 2, "data_quality": 1, "troubleshooting": 3,
    }


def test_corpus_run_links_match_mock_runs():
    runs = {r.run_id: r for r in load_mock_runs(MOCK).runs}
    linked = [d for d in load_documents(DOCS).docs if d.run_id]
    assert len(linked) == 5
    for d in linked:
        assert d.run_id in runs, f"{d.doc_id} points to unknown run {d.run_id}"
        assert runs[d.run_id].job_id == d.job_id