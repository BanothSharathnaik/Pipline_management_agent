from pathlib import Path

from src.ingestion.parser import load_documents
from src.retrieval.keyword_search import TfidfIndex

DOCS = Path(__file__).resolve().parents[1] / "data" / "diagnostic_documents"

PROBES = [
    ("A. control: exact wording", "permission denied main.crm.customers",
     "ERR-002 and ERR-003 on top"),
    ("B. synonym: access/permission, customer/customers", "access denied for customer data",
     "is the right evidence on top, or an unrelated doc?"),
    ("C. no shared words: slow vs timeout", "why is it so slow",
     "TS-003 / ERR-005 are relevant. Are they found?"),
    ("D. no stemming: fail", "fail",
     "corpus only says 'failed'"),
    ("D2. no stemming: failed", "failed",
     "compare with D"),
    ("E. identifier split: different table", "SELECT denied on main.sales.customers",
     "ERR-002/003 are about main.crm.customers, a different table"),
    ("F. opposite meaning", "user has SELECT permission on customers",
     "query says the user HAS permission. What ranks first?"),
    ("G. out of scope", "how do I bake bread",
     "should return nothing"),
]


def main() -> None:
    loaded = load_documents(DOCS)
    index = TfidfIndex(loaded.docs)
    print(f"[MOCK DATA] {index.size} documents indexed, {len(loaded.issues)} load issues\n")

    for label, query, look_for in PROBES:
        print(f"=== {label}")
        print(f"query: {query!r}")
        print(f"look for: {look_for}")
        results = index.search(query, top_k=3)
        if not results:
            print("  (no results)")
        for r in results:
            print(
                f"  {r.rank}. {r.doc.doc_id:<9} score={r.score:.3f}  "
                f"[{r.doc.doc_type}]  matched={list(r.matched_terms)}"
            )
            print(f"     {r.doc.title}")
        print()


if __name__ == "__main__":
    main()