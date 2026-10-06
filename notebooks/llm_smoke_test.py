from pathlib import Path

from src.generation.llm_client import PROMPT_BUDGET_CHARS, LLMError, OllamaClient
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
    llm = OllamaClient.from_env()
    print(f"model {llm.model} at {llm.base_url} | num_ctx {llm.num_ctx} | "
          f"num_predict {llm.num_predict} | timeout {llm.timeout_s:g}s\n")

    try:
        r = llm.generate("You are a test.", "Reply with exactly the word OK.")
    except LLMError as exc:
        print(f"LLM unavailable: {exc}")
        return
    print(f"tiny call: {r.text!r} | {r.elapsed_s:.1f}s | tokens in/out {r.prompt_tokens}/{r.completion_tokens} "
          f"| done_reason {r.done_reason}\n")

    e = load_embedder()
    docs = (load_documents(ROOT / "data" / "diagnostic_documents").docs
            + load_documents(ROOT / "data" / "conflict_demo").docs)
    chunks = [c for d in docs for c in chunk_document(d)]
    cache = EmbeddingCache.load(ROOT / "data" / "cache" / "embeddings.npz", e.model_name, e.dim)
    idx = VectorIndex(e.dim, e.model_name)
    idx.add(chunks, embed_chunks(chunks, e, cache))
    runs = load_mock_runs(ROOT / "data" / "mock_runs.json").runs
    pipeline = RetrievalPipeline(idx, e, known_run_ids=[x.run_id for x in runs])

    print("[MOCK DATA] raw model answers, not yet checked for citations or invented facts\n")
    for q in QUESTIONS:
        ctx = retrieve_for_question(pipeline, q, runs, docs=docs)
        bundle = build_prompt(ctx, max_chars=PROMPT_BUDGET_CHARS)
        chars = len(bundle.system) + len(bundle.user)
        print("#" * 5, q)
        try:
            r = llm.generate(bundle.system, bundle.user)
        except LLMError as exc:
            print(f"LLM unavailable: {exc}\n")
            continue
        ratio = f"{chars / r.prompt_tokens:.2f}" if r.prompt_tokens else "n/a"
        print(f"prompt {chars} chars = {r.prompt_tokens} tokens ({ratio} chars/token) | "
              f"answer {r.completion_tokens} tokens | {r.elapsed_s:.1f}s | "
              f"done_reason {r.done_reason} | cut_off {r.cut_off} | context_nearly_full {r.context_nearly_full}")
        print(f"supplied evidence ids: {bundle.evidence_ids} | fact ids: {bundle.fact_ids}\n")
        print(r.text)
        print("\n" + "=" * 78 + "\n")


if __name__ == "__main__":
    main()