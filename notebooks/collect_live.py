"""Collect live runs (read-only), save a local snapshot, and print what was mapped.
Needs DATABRICKS_HOST/DATABRICKS_TOKEN in the environment."""
from collections import Counter

from src.monitoring.databricks_client import DatabricksClient, DatabricksError
from src.monitoring.formatting import fmt_run, run_detail_lines
from src.monitoring.live_collector import SNAPSHOT_PATH, collect_live, save_snapshot


def main() -> None:
    try:
        col = collect_live(DatabricksClient.from_env())
    except (ValueError, DatabricksError) as exc:
        print(f"error: {exc}")
        return
    save_snapshot(col, SNAPSHOT_PATH)
    print(f"[LIVE DATA] {len(col.runs)} runs fetched at {col.fetched_at.strftime('%Y-%m-%d %H:%M UTC')} "
          f"| truncated {col.truncated} | snapshot {SNAPSHOT_PATH.name}")

    raw_states = Counter((r.get("state") or {}).get("result_state") for r in col.raw_runs)
    print(f"raw result_state values seen: {dict(raw_states)}\n")

    for r in col.runs:
        print(fmt_run(r))
    print()
    for r in col.runs:
        if r.error_message or r.queue_seconds:
            print("\n".join(run_detail_lines(r)) + "\n")

    print(f"{len(col.warnings)} warning(s)")
    for w in col.warnings:
        print(f"  WARNING: {w}")


if __name__ == "__main__":
    main()