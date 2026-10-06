from pathlib import Path

from src.ingestion.chunker import chunk_document
from src.ingestion.parser import load_documents
from src.monitoring.collector import load_mock_runs
from src.retrieval.chunk_embeddings import EmbeddingCache, embed_chunks
from src.retrieval.embeddings import load_embedder
from src.retrieval.search import RetrievalPipeline, format_report
from src.retrieval.vector_index import VectorIndex

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = [
    "Why did run r2002 fail?",
    "Why did run r2003 fail?",
    "Why did run r1004 fail?",
    "Why did run r2005 fail?",
    "Why did run r9999 fail?",
    "Why is it so slow?",
    "Have we seen this permission error before?",
    "how do I bake bread",
]


def main() -> None:
    e = load_embedder()
    docs = load_documents(ROOT / "data" / "diagnostic_documents").docs
    chunks = [c for d in docs for c in chunk_document(d)]
    cache = EmbeddingCache.load(ROOT / "data" / "cache" / "embeddings.npz", e.model_name, e.dim)
    idx = VectorIndex(e.dim, e.model_name)
    idx.add(chunks, embed_chunks(chunks, e, cache))
    runs = load_mock_runs(ROOT / "data" / "mock_runs.json").runs
    pipeline = RetrievalPipeline(idx, e, known_run_ids=[r.run_id for r in runs])

    print(f"[MOCK DATA] {idx.size} chunks, {len(runs)} runs\n")
    for q in QUESTIONS:
        print(format_report(pipeline.retrieve(q)))
        print("\n" + "=" * 78 + "\n")


if __name__ == "__main__":
    main()