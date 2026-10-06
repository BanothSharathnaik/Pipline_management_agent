import time
from pathlib import Path

import numpy as np

from src.ingestion.chunker import chunk_document
from src.ingestion.parser import load_documents
from src.retrieval.chunk_embeddings import EmbeddingCache, embed_chunks
from src.retrieval.embeddings import load_embedder

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "data" / "diagnostic_documents"
CACHE_PATH = ROOT / "data" / "cache" / "embeddings.npz"

QUERIES = [
    "access denied for customer data",     # probe B (keyword search failed)
    "why is it so slow",                   # probe C
    "fail",                                # probe D
    "SELECT denied on main.sales.customers",  # probe E
    "user has SELECT permission on customers",  # probe F
    "how do I bake bread",                 # probe G
]


def main() -> None:
    t0 = time.perf_counter()
    embedder = load_embedder()
    print(f"model {embedder.model_name} (dim {embedder.dim}) ready in {time.perf_counter() - t0:.1f}s")

    docs = load_documents(DOCS).docs
    chunks = [c for d in docs for c in chunk_document(d)]
    print(f"[MOCK DATA] {len(docs)} documents -> {len(chunks)} chunks")

    try:
        cache = EmbeddingCache.load(CACHE_PATH, embedder.model_name, embedder.dim)
        print(f"loaded cache with {len(cache)} entries")
    except (FileNotFoundError, ValueError) as exc:
        cache = EmbeddingCache(embedder.model_name, embedder.dim)
        print(f"starting a new cache ({exc})")

    before = len(cache)
    t0 = time.perf_counter()
    vectors = embed_chunks(chunks, embedder, cache)
    print(f"embedded chunks in {time.perf_counter() - t0:.2f}s | "
          f"new entries: {len(cache) - before} | hits {cache.hits}, misses {cache.misses}")
    cache.save(CACHE_PATH)
    print(f"vectors shape {vectors.shape}\n")

    # Phase 6 preview: unit vectors, so dot product == cosine similarity
    qvecs = embedder.embed(QUERIES)
    scores = qvecs @ vectors.T
    for q, row in zip(QUERIES, scores):
        print(f"=== {q!r}")
        for rank, i in enumerate(np.argsort(-row)[:3], start=1):
            c = chunks[i]
            print(f"  {rank}. {c.chunk_id:<10} {row[i]:.3f}  {c.title}")
        print()


if __name__ == "__main__":
    main()