from datetime import datetime, timezone

from src.routing.router import route_question
from tests.test_router import CASES

NOW = datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc)
EXTRA = [
    "Why did the pipeline fail yesterday?",
    "Which runs failed in the last 0 hours?",
    "Which runs exceeded 5 minutes today?",
]


def main() -> None:
    print(f"now = {NOW.isoformat()}  (explicit, never read from the clock)\n")
    for question in [q for q, _ in CASES] + EXTRA:
        r = route_question(question, NOW)
        print(f"{question}")
        print(f"   -> {r.intent} | structured={r.needs_structured} retrieval={r.needs_retrieval} | {r.matched_rule}")
        if r.run_ids:
            print(f"      run ids: {list(r.run_ids)}")
        if r.threshold_seconds is not None:
            print(f"      threshold: {r.threshold_seconds:g}s")
        for n in r.notes:
            print(f"      note: {n}")


if __name__ == "__main__":
    main()