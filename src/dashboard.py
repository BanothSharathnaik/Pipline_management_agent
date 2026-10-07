"""
Streamlit Web Dashboard for Databricks Pipeline Monitoring Assistant.
Provides a visual operational summary alongside an interactive RAG Q&A interface.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
import streamlit as st

# Ensure root directory is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.app import DOC_DIR, CACHE_PATH, DEFAULT_DATA, _load_dotenv
from src.generation.llm_client import OllamaClient
from src.ingestion.parser import load_documents, Document
from src.monitoring.collector import load_mock_runs
from src.monitoring.databricks_client import DatabricksClient
from src.monitoring.formatting import fmt_duration, fmt_time
from src.monitoring.live_collector import SNAPSHOT_PATH, collect_live, load_snapshot, save_snapshot
from src.monitoring.models import ResultState, RunRecord
from src.reporting.incident_report import build_incident_report
from src.retrieval.run_context import retrieve_for_question
from src.retrieval.search import RetrievalPipeline
from src.routing.executor import DisabledLLM, answer_question

# Load environment variables (.env)
_load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Databricks Pipeline Assistant",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Databricks Pipeline Monitoring & Diagnostic Assistant")

# --- SIDEBAR: Data Source & Settings ---
st.sidebar.header("Configuration")
data_source = st.sidebar.radio("Data Source", ["live", "mock"], index=0)
similarity_threshold = st.sidebar.slider(
    "RAG Similarity Threshold", min_value=0.2, max_value=0.9, value=0.5, step=0.05
)

# Button to refresh live data
if st.sidebar.button("🔄 Refresh Telemetry Snapshot"):
    with st.spinner("Fetching snapshot from Databricks API..."):
        try:
            client = DatabricksClient.from_env()
            collection = collect_live(client, max_runs=100, max_output_fetches=20)
            save_snapshot(collection, SNAPSHOT_PATH)
            st.sidebar.success(f"Snapshot updated! ({len(collection.runs)} runs)")
            st.cache_resource.clear()
        except Exception as exc:
            st.sidebar.error(f"Failed to fetch live runs: {exc}")


# --- CACHED PIPELINE INITIALIZATION ---
@st.cache_resource
def load_data_and_pipeline(source: str):
    """Loads run records, diagnostic documents, and vector retrieval pipeline."""
    from src.ingestion.chunker import chunk_document
    from src.retrieval.chunk_embeddings import EmbeddingCache, embed_chunks
    from src.retrieval.embeddings import load_embedder
    from src.retrieval.vector_index import VectorIndex

    # 1. Load runs
    if source == "live":
        try:
            collection = load_snapshot(SNAPSHOT_PATH)
            runs = list(collection.runs)
        except Exception:
            runs = []
    else:
        loaded = load_mock_runs(DEFAULT_DATA)
        runs = list(loaded.runs)

    # 2. Load documents
    docs: list[Document] = []
    if DOC_DIR.is_dir():
        docs = load_documents(DOC_DIR).docs

    # 3. Build Vector Pipeline
    pipeline: Optional[RetrievalPipeline] = None
    if docs and runs:
        try:
            embedder = load_embedder()
            chunks = [c for d in docs for c in chunk_document(d)]
            try:
                cache = EmbeddingCache.load(CACHE_PATH, embedder.model_name, embedder.dim)
            except Exception:
                cache = EmbeddingCache(embedder.model_name, embedder.dim)
            vectors = embed_chunks(chunks, embedder, cache)
            cache.save(CACHE_PATH)
            index = VectorIndex(embedder.dim, embedder.model_name)
            index.add(chunks, vectors)
            pipeline = RetrievalPipeline(
                index, embedder, known_run_ids=[r.run_id for r in runs], min_score=0.5
            )
        except Exception as exc:
            st.warning(f"Could not initialize vector search pipeline: {exc}")

    # 4. LLM client
    try:
        llm = OllamaClient.from_env()
    except Exception:
        llm = DisabledLLM()

    return runs, docs, pipeline, llm


try:
    runs_data, docs, search_pipeline, llm_client = load_data_and_pipeline(data_source)
except Exception as e:
    st.error(f"Error loading monitoring backend: {e}")
    st.stop()


# Helper to convert RunRecord list to DataFrame
def runs_to_dataframe(runs: Sequence[RunRecord]) -> pd.DataFrame:
    rows = []
    for r in runs:
        rows.append({
            "run_id": r.run_id,
            "job_id": r.job_id,
            "pipeline_name": r.job_name or "unnamed",
            "status": r.result_state.value,
            "lifecycle_state": r.lifecycle_state or "n/a",
            "start_time": fmt_time(r.start_time),
            "duration": fmt_duration(r.duration_seconds),
            "duration_seconds": r.duration_seconds or 0,
            "tasks_count": len(r.tasks),
            "error_message": r.error_message or "",
        })
    return pd.DataFrame(rows)


# --- TAB LAYOUT ---
tab_overview, tab_incident, tab_chat = st.tabs(
    ["📊 Health & Runs Overview", "📄 Incident Report Generator", "💬 Interactive Q&A"]
)

# ==========================================
# TAB 1: OVERVIEW & RUN METRICS
# ==========================================
with tab_overview:
    st.subheader(f"Recent Pipeline Runs ({data_source.upper()} Data)")

    if runs_data:
        df = runs_to_dataframe(runs_data)

        # High-level KPIs
        col1, col2, col3, col4 = st.columns(4)
        total_runs = len(df)
        failed_count = len(df[df["status"].isin(["FAILED", "TIMED_OUT"])])
        success_count = len(df[df["status"] == "SUCCESS"])
        other_count = total_runs - failed_count - success_count

        col1.metric("Total Tracked Runs", total_runs)
        col2.metric("Successful Runs", success_count)
        col3.metric(
            "Failed Runs",
            failed_count,
            delta=f"-{failed_count}" if failed_count else "0",
            delta_color="inverse",
        )
        col4.metric("Other States", other_count)

        # Filters
        status_filter = st.multiselect(
            "Filter by Status",
            options=sorted(df["status"].unique()),
            default=sorted(df["status"].unique()),
        )
        filtered_df = df[df["status"].isin(status_filter)]

        # Display tabular view
        display_cols = [
            "run_id",
            "pipeline_name",
            "status",
            "start_time",
            "duration",
            "error_message",
        ]
        st.dataframe(filtered_df[display_cols], use_container_width=True, hide_index=True)
    else:
        st.info("No runs found in snapshot. Click 'Refresh Telemetry Snapshot' in sidebar.")

# ==========================================
# TAB 2: INCIDENT REPORT GENERATOR
# ==========================================
with tab_incident:
    st.subheader("Generate Automated Incident Report")

    if runs_data:
        run_options = {
            f"{r.run_id} ({r.job_name or r.job_id} - {r.result_state.value})": r.run_id
            for r in runs_data
        }
        selected_label = st.selectbox("Select Run to diagnose:", options=list(run_options.keys()))
        selected_run_id = run_options[selected_label]
        target_run = next(r for r in runs_data if r.run_id == selected_run_id)

        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            gen_btn = st.button("Generate Incident Report", type="primary")

        if gen_btn:
            with st.spinner(f"Analyzing run {selected_run_id} and synthesizing report..."):
                report = build_incident_report(
                    target_run=target_run,
                    all_runs=runs_data,
                    pipeline=search_pipeline,
                    docs=docs,
                    llm=llm_client,
                    min_score=similarity_threshold,
                )
                st.markdown(report.text)
    else:
        st.info("No runs available to generate report.")

# ==========================================
# TAB 3: INTERACTIVE CHAT (Q&A)
# ==========================================
with tab_chat:
    st.subheader("Ask Natural-Language Questions")
    st.caption("Ask questions about failure causes, run durations, or recurring errors.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # React to user input
    if prompt := st.chat_input("E.g., Why did run 437531692562553 fail?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Process query through backend router & executor
        with st.chat_message("assistant"):
            with st.spinner("Processing telemetry and evidence..."):
                now = datetime.now(timezone.utc)
                answer = answer_question(
                    question=prompt,
                    now=now,
                    runs=runs_data,
                    pipeline_factory=lambda: search_pipeline or load_data_and_pipeline(data_source)[2],
                    docs=docs,
                    llm=llm_client,
                    top_k=3,
                    data_note=f"{data_source} data",
                )
                response_text = answer.text
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
