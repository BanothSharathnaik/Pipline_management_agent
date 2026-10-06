import statistics
import time

import numpy as np

from src.ingestion.chunker import Chunk
from src.retrieval.vector_index import VectorIndex

DIM = 384
REPEATS = 20


def fake_chunks(n: int) -> list[Chunk]:
    # Synthetic benchmark data only. Nothing here is a monitoring record.
    return [
        Chunk(chunk_id=f"b{i}#0", doc_id=f"b{i}", index=0, text="x", start_char=0, end_char=1,
              doc_type="error_log" if i % 2 else "troubleshooting", source="mock", origin="bench")
        for i in range(n)
    ]


def median_ms(fn) -> float:
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def main() -> None:
    rng = np.random.default_rng(0)
    q = rng.normal(size=DIM).astype(np.float32)
    print("synthetic random vectors, dim 384, top_k=5, median of 20 runs\n")
    print(f"{'chunks':>8} {'matrix MB':>10} {'no filter ms':>13} {'with filter ms':>15}")
    for n in (1_000, 10_000, 100_000):
        idx = VectorIndex(DIM, "bench")
        idx.add(fake_chunks(n), rng.normal(size=(n, DIM)).astype(np.float32))
        no_filter = median_ms(lambda: idx.search(q, top_k=5))
        with_filter = median_ms(lambda: idx.search(q, top_k=5, doc_type="error_log"))
        print(f"{n:>8} {n * DIM * 4 / 1e6:>10.1f} {no_filter:>13.2f} {with_filter:>15.2f}")


if __name__ == "__main__":
    main()