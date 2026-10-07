from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from src.monitoring import queries as q
from src.monitoring.collector import load_mock_runs
from src.monitoring.formatting import fmt_run, run_detail_lines
from src.monitoring.models import RunRecord

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "mock_runs.json"
DOC_DIR = ROOT / "data" / "diagnostic_documents"
DEMO_DIR = ROOT / "data" / "conflict_demo"
CACHE_PATH = ROOT / "data" / "cache" / "embeddings.npz"


def print_runs(runs: Sequence[RunRecord], empty_message: str) -> None:
    if not runs:
        print(empty_message)
        return
    for r in runs:
        print(fmt_run(r))


def print_run_detail(r: RunRecord) -> None:
    print("\n".join(run_detail_lines(r)))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python3 -m src.app")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA, help="mock runs JSON file")
    p.add_argument("--now", help="ISO time with timezone; default is the current time")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("summary", help="counts per result state")

    s = sub.add_parser("latest", help="most recent run")
    s.add_argument("--job", help="job_id filter")

    s = sub.add_parser("failed", help="failed or timed-out runs")
    s.add_argument("--hours", type=float, help="only the last N hours before --now")
    s.add_argument("--job", help="job_id filter")

    s = sub.add_parser("slow", help="runs longer than a threshold")
    s.add_argument("--threshold", type=float, required=True, help="seconds")

    sub.add_parser("repeats", help="exactly repeated error messages")

    s = sub.add_parser("run", help="details of one run")
    s.add_argument("run_id")

    s = sub.add_parser("ask", help="ask a natural-language question")
    s.add_argument("question")
    s.add_argument("--no-llm", action="store_true", help="never call the language model")
    s.add_argument("--with-conflict-demo", action="store_true",
                   help="also index the planted conflict document HIST-003")
    s.add_argument("--top-k", type=int, default=5)
    return p


def _build_pipeline(docs, runs):
    """Slow part (loads the embedding model). Only called when a question needs retrieval."""
    from src.ingestion.chunker import chunk_document
    from src.retrieval.chunk_embeddings import EmbeddingCache, embed_chunks
    from src.retrieval.embeddings import load_embedder
    from src.retrieval.search import RetrievalPipeline
    from src.retrieval.vector_index import VectorIndex

    embedder = load_embedder()
    chunks = [c for d in docs for c in chunk_document(d)]
    try:
        cache = EmbeddingCache.load(CACHE_PATH, embedder.model_name, embedder.dim)
    except (FileNotFoundError, ValueError):
        cache = EmbeddingCache(embedder.model_name, embedder.dim)
    vectors = embed_chunks(chunks, embedder, cache)
    cache.save(CACHE_PATH)
    index = VectorIndex(embedder.dim, embedder.model_name)
    index.add(chunks, vectors)
    return RetrievalPipeline(index, embedder, known_run_ids=[r.run_id for r in runs])


def _run_ask(args, now: datetime, runs: Sequence[RunRecord]) -> int:
    from src.generation.llm_client import OllamaClient
    from src.ingestion.parser import load_documents
    from src.retrieval.embeddings import EmbeddingError
    from src.routing.executor import DisabledLLM, answer_question

    try:
        docs = []
        for directory in [DOC_DIR] + ([DEMO_DIR] if args.with_conflict_demo else []):
            loaded_docs = load_documents(directory)
            docs.extend(loaded_docs.docs)
            for issue in loaded_docs.issues:
                print(f"  document issue in {issue.filename}: {issue.message}")
        llm = DisabledLLM() if args.no_llm else OllamaClient.from_env()
        answer = answer_question(args.question, now, runs, lambda: _build_pipeline(docs, runs),
                                 docs, llm, top_k=args.top_k)
    except (ValueError, FileNotFoundError, EmbeddingError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(answer.text)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        now = datetime.fromisoformat(args.now) if args.now else datetime.now(timezone.utc)
        if now.tzinfo is None:
            raise ValueError("--now must include a timezone, e.g. 2026-10-05T12:00:00+00:00")
        loaded = load_mock_runs(args.data)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    runs = loaded.runs
    print(
        f"[MOCK DATA] {len(runs)} runs loaded | "
        f"{loaded.duplicates_dropped} duplicate dropped | "
        f"{len(loaded.issues)} records rejected"
    )
    for issue in loaded.issues:
        print(f"  load issue at index {issue.index} ({issue.run_id}): {issue.message}")
    print()

    cmd = args.command
    if cmd == "summary":
        for state, n in sorted(q.status_counts(runs).items(), key=lambda kv: kv[0].value):
            print(f"  {state.value:<10} {n}")

    elif cmd == "latest":
        r = q.latest_run(runs, args.job)
        if r is None:
            print("No run found.")
            return 1
        print_run_detail(r)

    elif cmd == "failed":
        if args.hours is not None:
            try:
                hits = q.failed_in_last(runs, now, args.hours, args.job)
            except ValueError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
        else:
            hits = q.failed_runs(runs, job_id=args.job)
        print_runs(hits, "No failed runs in selection.")

    elif cmd == "slow":
        try:
            hits = q.runs_exceeding(runs, args.threshold)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print_runs(hits, f"No runs exceed {args.threshold:g} seconds.")

    elif cmd == "repeats":
        groups = q.repeated_errors(runs)
        if not groups:
            print("No exactly repeated error messages.")
        for g in groups:
            print(f"{g.count}x  runs={list(g.run_ids)}  {g.message}")

    elif cmd == "run":
        found = q.find_runs(runs, args.run_id)
        if not found:
            print(f"No run found with id {args.run_id}.")
            return 1
        for r in found:
            print_run_detail(r)

    elif cmd == "ask":
        return _run_ask(args, now, runs)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())