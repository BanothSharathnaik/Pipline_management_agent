from pathlib import Path

from src.ingestion.chunker import chunk_document
from src.ingestion.parser import load_documents
from src.monitoring.collector import load_mock_runs
from src.retrieval.chunk_embeddings import EmbeddingCache, embed_chunks
from src.retrieval.embeddings import load_embedder
from src.retrieval.run_context import format_context, retrieve_for_question
from src.retrieval.search import RetrievalPipeline
from src.retrieval.vector_index import VectorIndex

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache" / "embeddings.npz"
QUESTIONS = [
    "Why did run r2003 fail?",
    "Why did run r2002 fail?",
    "Have we seen this permission error before?",
    "Why did run r3004 fail?",
]


def main() -> None:
    e = load_embedder()
    docs = (load_documents(ROOT / "data" / "diagnostic_documents").docs
            + load_documents(ROOT / "data" / "conflict_demo").docs)
    chunks = [c for d in docs for c in chunk_document(d)]
    cache = EmbeddingCache.load(CACHE, e.model_name, e.dim)
    idx = VectorIndex(e.dim, e.model_name)
    idx.add(chunks, embed_chunks(chunks, e, cache))
    cache.save(CACHE)
    runs = load_mock_runs(ROOT / "data" / "mock_runs.json").runs
    pipeline = RetrievalPipeline(idx, e, known_run_ids=[r.run_id for r in runs])

    print(f"[MOCK DATA] {len(docs)} documents (including planted conflict document HIST-003), "
          f"{idx.size} chunks, {len(runs)} runs\n")
    for q in QUESTIONS:
        print(format_context(retrieve_for_question(pipeline, q, runs, docs=docs)))
        print("\n" + "=" * 78 + "\n")


if __name__ == "__main__":
    main()