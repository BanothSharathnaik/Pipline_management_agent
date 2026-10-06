from pathlib import Path

from src.generation.prompt_builder import build_prompt
from src.ingestion.chunker import chunk_document
from src.ingestion.parser import load_documents
from src.monitoring.collector import load_mock_runs
from src.retrieval.chunk_embeddings import EmbeddingCache, embed_chunks
from src.retrieval.embeddings import load_embedder
from src.retrieval.run_context import retrieve_for_question
from src.retrieval.search import RetrievalPipeline
from src.retrieval.vector_index import VectorIndex

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ["Why did run r2003 fail?", "Why did run r3004 fail?", "Why did run r9999 fail?"]


def main() -> None:
    e = load_embedder()
    docs = (load_documents(ROOT / "data" / "diagnostic_documents").docs
            + load_documents(ROOT / "data" / "conflict_demo").docs)
    chunks = [c for d in docs for c in chunk_document(d)]
    cache = EmbeddingCache.load(ROOT / "data" / "cache" / "embeddings.npz", e.model_name, e.dim)
    idx = VectorIndex(e.dim, e.model_name)
    idx.add(chunks, embed_chunks(chunks, e, cache))
    runs = load_mock_runs(ROOT / "data" / "mock_runs.json").runs
    pipeline = RetrievalPipeline(idx, e, known_run_ids=[r.run_id for r in runs])

    print("[MOCK DATA] prompts are built, not sent anywhere\n")
    for q in QUESTIONS:
        p = build_prompt(retrieve_for_question(pipeline, q, runs, docs=docs))
        total = len(p.system) + len(p.user)
        print(f"##### {q}")
        print(f"system {len(p.system)} chars + user {len(p.user)} chars = {total} "
              f"(roughly {total // 4} tokens, a rough estimate)")
        print(f"fact ids: {p.fact_ids}")
        print(f"evidence ids: {p.evidence_ids}\n")
        print(p.user)
        print("\n" + "=" * 78 + "\n")


if __name__ == "__main__":
    main()