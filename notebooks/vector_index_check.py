from pathlib import Path

import numpy as np

from src.ingestion.chunker import chunk_document
from src.ingestion.parser import load_documents
from src.retrieval.chunk_embeddings import EmbeddingCache, embed_chunks
from src.retrieval.embeddings import load_embedder
from src.retrieval.vector_index import VectorIndex

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache" / "embeddings.npz"
QUERIES = [
    "access denied for customer data", "why is it so slow", "fail",
    "SELECT denied on main.sales.customers", "user has SELECT permission on customers",
    "how do I bake bread",
]


def main() -> None:
    e = load_embedder()
    docs = load_documents(ROOT / "data" / "diagnostic_documents").docs
    chunks = [c for d in docs for c in chunk_document(d)]
    cache = EmbeddingCache.load(CACHE, e.model_name, e.dim)
    vectors = embed_chunks(chunks, e, cache)

    idx = VectorIndex(e.dim, e.model_name)
    idx.add(chunks, vectors)
    print(f"[MOCK DATA] index holds {idx.size} chunks, dim {idx.dim}\n")

    qvecs = e.embed(QUERIES)
    brute = qvecs @ vectors.T            # vectors are already unit length
    all_match = True
    for q, qv, row in zip(QUERIES, qvecs, brute):
        res = idx.search(qv, top_k=3)
        expected = [chunks[i].chunk_id for i in np.argsort(-row)[:3]]
        same = [r.chunk.chunk_id for r in res] == expected
        all_match &= same
        print(f"{q!r}: {'matches brute force' if same else 'DIFFERS'}")
        for r in res:
            print(f"   {r.rank}. {r.chunk.chunk_id:<10} {r.score:.3f}")

    print("\nfilter demo: 'why is it so slow', troubleshooting only")
    for r in idx.search_text("why is it so slow", e, top_k=3, doc_type="troubleshooting"):
        print(f"   {r.rank}. {r.chunk.chunk_id:<10} {r.score:.3f}  {r.chunk.title}")
    print("\nALL MATCH" if all_match else "\nMISMATCH FOUND")


if __name__ == "__main__":
    main()