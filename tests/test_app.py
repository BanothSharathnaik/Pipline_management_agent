from src.app import main

NOW = ["--now", "2026-10-05T12:00:00+00:00"]


def run(capsys, *args):
    code = main([*NOW, *args])
    return code, capsys.readouterr()


def test_banner_and_load_issues_shown(capsys):
    code, out = run(capsys, "summary")
    assert code == 0
    assert "[MOCK DATA] 16 runs loaded" in out.out
    assert "1 duplicate dropped" in out.out
    assert "3 records rejected" in out.out
    assert "missing required field: run_id" in out.out


def test_summary_counts(capsys):
    _, out = run(capsys, "summary")
    assert "SUCCESS    7" in out.out
    assert "FAILED     5" in out.out
    assert "TIMED_OUT  1" in out.out


def test_latest(capsys):
    code, out = run(capsys, "latest")
    assert code == 0
    assert "r1005" in out.out
    assert "Duration:   n/a" in out.out


def test_failed_last_24h(capsys):
    code, out = run(capsys, "failed", "--hours", "24")
    assert code == 0
    assert "r2005" in out.out
    assert "r1004" not in out.out


def test_failed_all_includes_timeout(capsys):
    _, out = run(capsys, "failed")
    assert "r3002" in out.out


def test_failed_no_results_message(capsys):
    code, out = run(capsys, "failed", "--hours", "1")
    assert code == 0
    assert "No failed runs in selection." in out.out


def test_failed_bad_hours(capsys):
    code, out = run(capsys, "failed", "--hours", "-5")
    assert code == 2
    assert "hours must be positive" in out.err


def test_slow_order(capsys):
    _, out = run(capsys, "slow", "--threshold", "1000")
    assert out.out.index("r3002") < out.out.index("r1003")


def test_repeats_none(capsys):
    _, out = run(capsys, "repeats")
    assert "No exactly repeated error messages." in out.out


def test_run_detail_with_tasks(capsys):
    code, out = run(capsys, "run", "r2002")
    assert code == 0
    assert "ingest_customers  FAILED" in out.out
    assert "depends_on=[ingest_customers]" in out.out


def test_run_missing_error_text_is_not_invented(capsys):
    _, out = run(capsys, "run", "r2005")
    assert "(no error message recorded)" in out.out


def test_run_not_found(capsys):
    code, out = run(capsys, "run", "nope")
    assert code == 1
    assert "No run found with id nope." in out.out


def test_bad_data_path(capsys):
    code = main(["--data", "nope.json", "summary"])
    assert code == 2
    assert "not found" in capsys.readouterr().err


def test_naive_now_rejected(capsys):
    code = main(["--now", "2026-10-05T12:00:00", "summary"])
    assert code == 2
    assert "timezone" in capsys.readouterr().err