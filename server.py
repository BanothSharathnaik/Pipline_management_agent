"""
server.py - REST API bridge connecting Frontend (React/Lovable) to Databricks Monitoring & Custom RAG.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure repo root is on sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.app import CACHE_PATH, DEFAULT_DATA, DOC_DIR, _load_dotenv
from src.generation.llm_client import OllamaClient
from src.ingestion.chunker import chunk_document
from src.ingestion.parser import Document, load_documents
from src.monitoring.collector import load_mock_runs
from src.monitoring.databricks_client import DatabricksClient
from src.monitoring.formatting import fmt_duration, fmt_time
from src.monitoring.live_collector import (
    SNAPSHOT_PATH,
    collect_live,
    load_snapshot,
    save_snapshot,
)
from src.monitoring.models import ResultState, RunRecord
from src.monitoring.queries import failed_runs, get_failed_runs, repeated_errors
from src.reporting.incident_report import build_incident_report
from src.retrieval.chunk_embeddings import EmbeddingCache, embed_chunks
from src.retrieval.embeddings import load_embedder
from src.retrieval.run_context import retrieve_for_question
from src.retrieval.search import RetrievalPipeline
from src.retrieval.vector_index import VectorIndex
from src.routing.executor import DisabledLLM, answer_question

# Load environment variables (.env)
_load_dotenv()

app = FastAPI(
    title="PipelinePulse AI API",
    description="Databricks Pipeline Monitoring & Diagnostic Assistant API",
    version="1.0.0",
)

# Allow requests from local frontend (Vite port 5173, 3000, 8080, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
STATIC_DIR = ROOT / "static"
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Global in-memory system instances
runs_data: list[RunRecord] = []
docs: list[Document] = []
search_pipeline: Optional[RetrievalPipeline] = None
llm_client = None


def _format_run(r: RunRecord) -> dict[str, Any]:
    return {
        "run_id": r.run_id,
        "job_id": r.job_id,
        "pipeline_name": r.job_name or "unnamed",
        "job_name": r.job_name or "unnamed",
        "status": r.result_state.value,
        "lifecycle_state": r.lifecycle_state or "n/a",
        "start_time": r.start_time.isoformat() if r.start_time else None,
        "end_time": r.end_time.isoformat() if r.end_time else None,
        "duration_seconds": r.duration_seconds or 0,
        "duration_formatted": fmt_duration(r.duration_seconds),
        "error_message": r.error_message or "",
        "tasks_count": len(r.tasks),
        "source": r.source,
    }


def _build_search_pipeline(documents: list[Document], runs: list[RunRecord]) -> Optional[RetrievalPipeline]:
    if not documents or not runs:
        return None
    try:
        embedder = load_embedder()
        chunks = [c for d in documents for c in chunk_document(d)]
        try:
            cache = EmbeddingCache.load(CACHE_PATH, embedder.model_name, embedder.dim)
        except Exception:
            cache = EmbeddingCache(embedder.model_name, embedder.dim)
        vectors = embed_chunks(chunks, embedder, cache)
        cache.save(CACHE_PATH)
        index = VectorIndex(embedder.dim, embedder.model_name)
        index.add(chunks, vectors)
        return RetrievalPipeline(index, embedder, known_run_ids=[r.run_id for r in runs], min_score=0.5)
    except Exception as exc:
        print(f"Warning: Could not build search pipeline: {exc}", file=sys.stderr)
        return None


def init_system(source: str = "live") -> None:
    global runs_data, docs, search_pipeline, llm_client

    # 1. Ingest live snapshot (fallback to mock if live is absent)
    runs_data = []
    if source == "live":
        try:
            collection = load_snapshot(SNAPSHOT_PATH)
            runs_data = list(collection.runs)
        except Exception:
            pass

    if not runs_data:
        try:
            loaded = load_mock_runs(DEFAULT_DATA)
            runs_data = list(loaded.runs)
        except Exception as exc:
            print(f"Warning: Could not load mock runs: {exc}", file=sys.stderr)

    # 2. Ingest diagnostic documents
    docs = []
    if DOC_DIR.is_dir():
        try:
            docs = load_documents(DOC_DIR).docs
        except Exception as exc:
            print(f"Warning: Could not load documents: {exc}", file=sys.stderr)

    # 3. Build Vector Retrieval Pipeline
    search_pipeline = _build_search_pipeline(docs, runs_data)

    # 4. Initialize LLM client
    try:
        llm_client = OllamaClient.from_env()
    except Exception:
        llm_client = None


@app.on_event("startup")
def startup_event():
    init_system("live")
    print("Backend initialized successfully!")


# --- ENDPOINTS ---

@app.get("/")
@app.get("/dashboard")
def dashboard():
    index_file = STATIC_DIR / "index.html"
    if index_file.is_file():
        return FileResponse(index_file)
    return {
        "status": "healthy",
        "service": "PipelinePulse AI Backend",
        "runs_count": len(runs_data),
        "docs_count": len(docs),
    }


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "PipelinePulse AI Backend",
        "runs_count": len(runs_data),
        "docs_count": len(docs),
    }


# 1. Pipeline Runs Telemetry Endpoint
@app.get("/api/runs")
def get_runs(source: Optional[str] = None):
    global runs_data
    if source in ("live", "mock"):
        init_system(source)
    return {
        "count": len(runs_data),
        "runs": [_format_run(r) for r in runs_data],
    }


# 2. Refresh Snapshot from Databricks API
@app.post("/api/runs/refresh")
def refresh_runs():
    global runs_data, search_pipeline
    try:
        client = DatabricksClient.from_env()
        collection = collect_live(client, max_runs=100, max_output_fetches=20)
        save_snapshot(collection, SNAPSHOT_PATH)
        runs_data = list(collection.runs)
        search_pipeline = _build_search_pipeline(docs, runs_data)
        return {"status": "refreshed", "count": len(runs_data)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to refresh live runs: {exc}")


# 3. Incident Report Generator Endpoint
@app.get("/api/incident-report/{run_id}")
def generate_incident_report(run_id: str, similarity_threshold: float = 0.5):
    found = [r for r in runs_data if str(r.run_id) == str(run_id)]
    if not found:
        raise HTTPException(status_code=404, detail=f"Run ID '{run_id}' not found in active telemetry store.")

    target_run = found[0]
    try:
        report = build_incident_report(
            target_run=target_run,
            all_runs=runs_data,
            pipeline=search_pipeline,
            docs=docs,
            llm=llm_client,
            min_score=similarity_threshold,
        )
        return {
            "status": "success",
            "run_id": target_run.run_id,
            "job_id": target_run.job_id,
            "job_name": target_run.job_name,
            "result_state": target_run.result_state.value,
            "report_markdown": report.text,
            "has_prior_occurrences": report.has_prior_occurrences,
            "evidence_count": report.evidence_count,
            "llm_used": report.llm_used,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {exc}")


# 4. Interactive RAG & Question Answering Endpoint
class ChatRequest(BaseModel):
    query: str
    similarity_threshold: Optional[float] = 0.5


@app.post("/api/chat")
def handle_chat(req: ChatRequest):
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    now = datetime.now(timezone.utc)
    threshold = req.similarity_threshold or 0.5

    # Use unified query router & executor
    try:
        pipeline = search_pipeline or _build_search_pipeline(docs, runs_data)
        answer = answer_question(
            question=query,
            now=now,
            runs=runs_data,
            get_pipeline=lambda: pipeline,
            docs=docs,
            llm=llm_client or DisabledLLM(),
            top_k=3,
        )
        return {
            "route": answer.intent,
            "reply": answer.text,
            "llm_used": answer.llm_used,
            "resolved_run_id": answer.resolved_run_id,
            "warnings": list(answer.warnings),
        }
    except Exception as exc:
        # Fallback to direct semantic vector search if routing raises
        if search_pipeline is not None:
            try:
                rep = search_pipeline.retrieve(query, top_k=3, min_score=threshold)
                hits = rep.related.results
                if hits:
                    formatted = "\n\n".join([f"- **[{h.chunk.chunk_id}]**: {h.chunk.text}" for h in hits])
                    return {
                        "route": "semantic_retrieval",
                        "reply": f"Found relevant reference documentation:\n\n{formatted}",
                        "evidence": [
                            {"chunk_id": h.chunk.chunk_id, "text": h.chunk.text, "score": h.score}
                            for h in hits
                        ],
                    }
            except Exception:
                pass

        return {
            "route": "error",
            "reply": f"Unable to process question: {exc}",
            "evidence": [],
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
