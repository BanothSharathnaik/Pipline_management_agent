import time

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

SENTENCES = [
    "Permission denied reading customers table",
    "Access to the customer data was refused",
    "Disk is full on the node",
]


def main() -> None:
    from sentence_transformers import SentenceTransformer

    t0 = time.perf_counter()
    model = SentenceTransformer(MODEL_NAME)
    load_s = time.perf_counter() - t0
    print(f"loaded {MODEL_NAME} in {load_s:.1f}s on device: {model.device}")
    print(f"embedding dimension: {model.get_sentence_embedding_dimension()}")
    print(f"max sequence length: {model.max_seq_length}")

    t0 = time.perf_counter()
    vecs = np.asarray(model.encode(SENTENCES))
    print(f"encoded {len(SENTENCES)} texts in {time.perf_counter() - t0:.3f}s")
    print(f"shape: {vecs.shape}, dtype: {vecs.dtype}")
    print("vector norms:", [round(float(np.linalg.norm(v)), 4) for v in vecs])

    unit = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    sims = unit @ unit.T
    print("\ncosine similarities:")
    for i in range(len(SENTENCES)):
        for j in range(i + 1, len(SENTENCES)):
            print(f"  {i} vs {j}: {sims[i, j]:.3f}   {SENTENCES[i]!r} / {SENTENCES[j]!r}")


if __name__ == "__main__":
    main()