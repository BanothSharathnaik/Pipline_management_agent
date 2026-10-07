import pytest

from src.app import main

NOW = ["--now", "2026-10-05T12:00:00+00:00"]


def run(capsys, *args):
    code = main([*NOW, *args])
    return code, capsys.readouterr()


def test_ask_structured_question(capsys):
    code, out = run(capsys, "ask", "Which pipelines failed yesterday?")
    assert code == 0
    assert "Route: failed_list" in out.out
    assert out.out.index("r1004") < out.out.index("r3004") < out.out.index("r2005")


def test_ask_unsupported_question(capsys):
    code, out = run(capsys, "ask", "How has execution duration changed over time?")
    assert code == 0 and "not implemented yet" in out.out


def test_ask_blank_question(capsys):
    code, out = run(capsys, "ask", "   ")
    assert code == 2 and "non-empty" in out.err


def test_ask_options_parse(capsys):
    code, out = run(capsys, "ask", "--no-llm", "--top-k", "3", "What is the latest run?")
    assert code == 0 and "Run:        r1005" in out.out


@pytest.mark.slow
def test_ask_history_with_real_model(capsys):
    code, out = run(capsys, "ask", "--no-llm",
                    "Have we seen this permission denied error on main.crm.customers before?")
    assert code == 0
    assert "r2002" in out.out and "r2003" in out.out
    assert "Verified monitoring records" in out.out