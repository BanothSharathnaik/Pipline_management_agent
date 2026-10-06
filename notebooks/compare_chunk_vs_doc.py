from pathlib import Path

from src.ingestion.chunker import chunk_document
from src.ingestion.parser import load_documents
from src.retrieval.chunk_search import ChunkIndex
from src.retrieval.keyword_search import TfidfIndex

DOCS = Path(__file__).resolve().parents[1] / "data" / "diagnostic_documents"
QUERIES = [
    "show grants privileges",
    "permission denied main.crm.customers",
    "unresolved column schema",
    "timeout seconds",
]


def main() -> None:
    docs = load_documents(DOCS).docs
    doc_index = TfidfIndex(docs)

    default_chunks = [c for d in docs for c in chunk_document(d)]
    small_chunks = [c for d in docs for c in chunk_document(d, max_chars=120, overlap_lines=1)]
    print(f"[MOCK DATA] {len(docs)} documents -> {len(default_chunks)} chunks (default policy), "
          f"{len(small_chunks)} chunks (max_chars=120, overlap 1)\n")
    chunk_index = ChunkIndex(small_chunks)

    for q in QUERIES:
        print(f"=== {q!r}")
        print("document level:")
        for r in doc_index.search(q, top_k=3):
            print(f"  {r.rank}. {r.doc.doc_id:<9} {r.score:.3f}  ({len(r.doc.body)} chars)")
        print("chunk level (small chunks):")
        for r in chunk_index.search(q, top_k=3):
            c = r.chunk
            print(f"  {r.rank}. {c.chunk_id:<11} {r.score:.3f}  chars {c.start_char}-{c.end_char}  "
                  f"matched={list(r.matched_terms)}")
            print(f"       {c.text[:70]!r}")
        print()


if __name__ == "__main__":
    main()