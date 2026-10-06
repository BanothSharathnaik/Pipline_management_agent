from pathlib import Path

from src.ingestion.chunker import chunk_document
from src.ingestion.parser import load_documents
from src.retrieval.chunk_embeddings import EmbeddingCache, embed_chunks
from src.retrieval.embeddings import load_embedder
from src.retrieval.evidence import select_evidence
from src.retrieval.vector_index import VectorIndex

ROOT = Path(__file__).resolve().parents[1]
QUERIES = [
    "access denied for customer data", "why is it so slow", "fail",
    "SELECT denied on main.sales.customers", "user has SELECT permission on customers",
    "how do I bake bread",
]


def main() -> None:
    e = load_embedder()
    docs = load_documents(ROOT / "data" / "diagnostic_documents").docs
    chunks = [c for d in docs for c in chunk_document(d)]
    cache = EmbeddingCache.load(ROOT / "data" / "cache" / "embeddings.npz", e.model_name, e.dim)
    idx = VectorIndex(e.dim, e.model_name)
    idx.add(chunks, embed_chunks(chunks, e, cache))

    print(f"[MOCK DATA] {idx.size} chunks | min_score 0.15 is provisional (fitted to these 6 queries)\n")
    for q in QUERIES:
        ev = select_evidence(idx.search_text(q, e, top_k=5))
        best = f"{ev.best_score:.3f}" if ev.best_score is not None else "n/a"
        print(f"{q!r}: {ev.status} | best score {best} | kept {len(ev.results)}, "
              f"dropped low {ev.dropped_low_score}, dropped per-doc {ev.dropped_per_doc}")
        for r in ev.results:
            print(f"   {r.rank}. {r.chunk.chunk_id:<10} {r.score:.3f}")


if __name__ == "__main__":
    main()