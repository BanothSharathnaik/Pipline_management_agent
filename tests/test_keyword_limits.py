from pathlib import Path

import pytest

from src.ingestion.parser import load_documents
from src.retrieval.keyword_search import TfidfIndex

CORPUS = Path(__file__).resolve().parents[1] / "data" / "diagnostic_documents"


@pytest.fixture(scope="module")
def index():
    return TfidfIndex(load_documents(CORPUS).docs)


def test_limit_no_stemming_fail_vs_failed(index):
    assert index.search("fail") == []          # corpus never contains the token "fail"
    assert len(index.search("failed")) > 0     # but it does contain "failed"


def test_limit_synonym_slow_finds_nothing(index):
    # Timeout documents exist, but share no words with this question.
    assert index.search("why is it so slow") == []


def test_out_of_scope_returns_nothing(index):
    assert index.search("how do I bake bread") == []