from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.app import main
from src.monitoring.collector import load_mock_runs
from src.monitoring.models import ResultState, RunRecord, TaskRecord
from src.reporting.incident_report import IncidentReport, build_incident_report

ROOT = Path(__file__).resolve().parents[1]
MOCK_RUNS = load_mock_runs(ROOT / "data" / "mock_runs.json").runs
NOW = datetime(2026, 10, 7, 12, 0, tzinfo=timezone.utc)

LIVE_FAIL = RunRecord.from_dict({
    "run_id": "437531692562553",
    "job_id": "658020425585260",
    "job_name": "JOB-FAILURA",
    "source": "live",
    "result_state": "FAILED",
    "lifecycle_state": "TERMINATED",
    "start_time": "2026-10-07T03:00:25+00:00",
    "end_time": "2026-10-07T03:02:07+00:00",
    "queue_seconds": 15.0,
    "error_message": "Exception: Deliberate failure: Testing Databricks job error handling.",
    "tasks": [
        {
            "task_key": "task_failure",
            "result_state": "FAILED",
            "start_time": "2026-10-07T03:00:25+00:00",
            "end_time": "2026-10-07T03:02:07+00:00",
            "error_message": "Exception: Deliberate failure: Testing Databricks job error handling.",
            "attempts": 2,
        }
    ],
})

LIVE_OK = RunRecord.from_dict({
    "run_id": "799790931046816",
    "job_id": "568297559042585",
    "job_name": "JOB-SUCCESS",
    "source": "live",
    "result_state": "SUCCESS",
    "lifecycle_state": "TERMINATED",
    "start_time": "2026-10-07T03:00:43+00:00",
    "end_time": "2026-10-07T03:00:57+00:00",
    "error_message": None,
    "tasks": [
        {
            "task_key": "task_success",
            "result_state": "SUCCESS",
            "start_time": "2026-10-07T03:00:43+00:00",
            "end_time": "2026-10-07T03:00:57+00:00",
            "attempts": 1,
        }
    ],
})


def test_build_report_for_mock_failed_run():
    target = next(r for r in MOCK_RUNS if r.run_id == "r2002")
    report = build_incident_report(target_run=target, all_runs=MOCK_RUNS, now=NOW)

    assert isinstance(report, IncidentReport)
    assert report.run_id == "r2002"
    assert report.result_state == "FAILED"
    text = report.text

    # Required sections present
    assert "## 1. Incident Header" in text
    assert "- Incident / Run ID:   r2002" in text
    assert "- Data Source:          mock" in text
    assert "## 2. Executive Summary" in text
    assert "- Overall Result:       FAILED" in text
    assert "## 3. Impact Assessment" in text
    assert "## 4. Execution Timeline" in text
    assert "## 5. Verbatim Error Message" in text
    assert target.error_message in text
    assert "## 6. Diagnostic Evidence" in text
    assert "## 7. Earlier Occurrences & Recurrence" in text
    assert "## 8. Unconfirmed Claims & Unknowns" in text
    assert "## 9. Limitations & Scope Bounds" in text
    assert "No generative language model was used in this report" in text


def test_build_report_for_live_failed_run_with_retries():
    report = build_incident_report(
        target_run=LIVE_FAIL,
        all_runs=[LIVE_FAIL, LIVE_OK],
        now=NOW,
        data_note="live snapshot fetched 2 minutes ago",
    )

    assert report.run_id == "437531692562553"
    text = report.text

    # Check header & freshness note
    assert "- Data Source:          live" in text
    assert "- Data Freshness:       live snapshot fetched 2 minutes ago" in text

    # Check queue duration
    assert "- Queue Duration:       0m15s" in text

    # Check retry reporting in impact and timeline
    assert "task_failure (2 attempts)" in text
    assert "Tasks with Retries:   1 (task_failure [2x])" in text
    assert "2 attempts" in text

    # Verbatim error preserved
    assert "Exception: Deliberate failure: Testing Databricks job error handling." in text

    # Earlier occurrences
    assert report.has_prior_occurrences is False


def test_earlier_occurrences_identified():
    prior_fail = RunRecord.from_dict({
        "run_id": "prev-1",
        "job_id": LIVE_FAIL.job_id,
        "source": "live",
        "result_state": "FAILED",
        "start_time": "2026-10-06T03:00:00+00:00",
        "error_message": LIVE_FAIL.error_message,
    })

    report = build_incident_report(
        target_run=LIVE_FAIL,
        all_runs=[LIVE_FAIL, prior_fail, LIVE_OK],
        now=NOW,
    )

    assert report.has_prior_occurrences is True
    assert "Identical Error Repeated: 1 other run(s)" in report.text
    assert "Run prev-1" in report.text
    assert "Prior Failures for Job '658020425585260': 1 previous failure(s)" in report.text


def test_build_report_for_success_run():
    report = build_incident_report(target_run=LIVE_OK, all_runs=[LIVE_OK], now=NOW)
    assert report.result_state == "SUCCESS"
    assert "Overall Result:       SUCCESS" in report.text
    assert "Failed Tasks:         0" in report.text
    assert "No error message recorded in run or task details." in report.text


def test_cli_report_subcommand(capsys):
    exit_code = main(["report", "--no-llm", "r2002"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "# INCIDENT REPORT: Run r2002" in captured.out
    assert "## 1. Incident Header" in captured.out
    assert "## 5. Verbatim Error Message" in captured.out
    assert "## 8. Unconfirmed Claims & Unknowns" in captured.out


class FakeReportLLM:
    model = "mock-ollama-3"

    def generate(self, system: str, user: str):
        from src.generation.llm_client import LLMResponse
        return LLMResponse(
            text=(
                "Verified facts\n- [F1] fact\n\n"
                "Possible explanations (hypotheses)\n1. [H1] schema issue [E1]\n\n"
                "What to investigate next\n- inspect schema [E1]\n\n"
                "Limits\n- sample docs only [E1]"
            ),
            model="mock-ollama-3",
            prompt_tokens=100,
            completion_tokens=50,
            done_reason="stop",
            elapsed_s=0.1,
            num_ctx=4096,
        )


def test_incident_report_with_llm():
    from src.ingestion.chunker import Chunk
    from src.retrieval.embeddings import HashingEmbedder
    from src.retrieval.search import RetrievalPipeline
    from src.retrieval.vector_index import VectorIndex

    e = HashingEmbedder(dim=64)
    idx = VectorIndex(e.dim, e.model_name)
    target = next(r for r in MOCK_RUNS if r.run_id == "r2002")
    chunk = Chunk(
        chunk_id="TS-001#0",
        doc_id="TS-001",
        index=0,
        text=target.error_message,
        start_char=0,
        end_char=len(target.error_message),
        doc_type="troubleshooting",
        source="mock",
        origin="ts_001.txt",
    )
    vec = e.embed([chunk.text])
    idx.add([chunk], vec)
    pipe = RetrievalPipeline(idx, e, known_run_ids=["r2002"])

    report = build_incident_report(
        target_run=target,
        all_runs=MOCK_RUNS,
        pipeline=pipe,
        llm=FakeReportLLM(),
        now=NOW,
    )

    assert "## 10. Potential Root Causes & Hypotheses (Language Model — UNVERIFIED)" in report.text
    assert "UNVERIFIED HYPOTHESES" in report.text
    assert "mock-ollama-3" in report.text
    assert "Section 10 below was synthesized by a language model as hypotheses and is UNVERIFIED." in report.text
    assert report.llm_used is True
