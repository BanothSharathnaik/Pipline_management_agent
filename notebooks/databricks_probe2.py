"""Second read-only probe. Needs DATABRICKS_HOST/DATABRICKS_TOKEN in the environment."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.monitoring.databricks_client import DatabricksClient, DatabricksError

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw_samples"


def key_paths(value, prefix: str = "", out: set | None = None) -> set:
    out = set() if out is None else out
    if isinstance(value, dict):
        for k, v in value.items():
            key_paths(v, f"{prefix}.{k}" if prefix else k, out)
    elif isinstance(value, list):
        for item in value:
            key_paths(item, f"{prefix}[]", out)
    else:
        out.add(f"{prefix}: {type(value).__name__}")
    return out


def ms(value) -> str:
    if not isinstance(value, (int, float)) or value <= 0:
        return repr(value)
    return datetime.fromtimestamp(value / 1000, timezone.utc).strftime("%H:%M:%S")


def short(value, n: int = 300) -> str:
    text = json.dumps(value, default=str)
    return text if len(text) <= n else text[: n - 3] + "..."


def main() -> None:
    client = DatabricksClient.from_env()
    fetch = client.list_runs(max_runs=50)
    print(f"runs {len(fetch.runs)} | pages {fetch.pages} | truncated {fetch.truncated} "
          f"| duplicates dropped {fetch.duplicates_dropped}\n")

    paths: set = set()
    observed: dict[str, set] = {}
    for run in fetch.runs:
        key_paths({k: v for k, v in run.items() if k != "tasks"}, "run", paths)
        for t in run.get("tasks") or []:
            key_paths(t, "task", paths)

    print("all key paths seen across runs and tasks (names and types, no values):")
    for p in sorted(paths):
        print(f"  {p}")

    def note(label: str, value) -> None:
        if value is not None:
            observed.setdefault(label, set()).add(str(value))

    print("\nper run:")
    for run in fetch.runs:
        st, status = run.get("state") or {}, run.get("status") or {}
        note("run life_cycle_state", st.get("life_cycle_state"))
        note("run result_state", st.get("result_state"))
        note("run status.state", status.get("state"))
        note("run termination code", (status.get("termination_details") or {}).get("code"))
        start, end = run.get("start_time"), run.get("end_time")
        computed = (end - start) / 1000 if isinstance(start, int) and isinstance(end, int) and end > 0 else None
        print(f"run {run['run_id']} | job {run.get('job_id')} | {run.get('run_name')} | "
              f"#{run.get('number_in_job')} | start {ms(start)} end {ms(end)} | "
              f"end-start {computed}s vs run_duration {run.get('run_duration')}ms")
        print(f"    state:  {short(st)}")
        print(f"    status: {short(status)}")
        for t in run.get("tasks") or []:
            ts, tstatus = t.get("state") or {}, t.get("status") or {}
            note("task life_cycle_state", ts.get("life_cycle_state"))
            note("task result_state", ts.get("result_state"))
            note("task status.state", tstatus.get("state"))
            note("task termination code", (tstatus.get("termination_details") or {}).get("code"))
            print(f"    task {t.get('task_key')} (task run {t.get('run_id')}, attempt {t.get('attempt_number')}) "
                  f"| state {short(ts, 200)} | status {short(tstatus, 200)}")
            finished_badly = ts.get("life_cycle_state") == "TERMINATED" and ts.get("result_state") != "SUCCESS"
            if finished_badly:
                try:
                    out = client.get_run_output(t["run_id"])
                except DatabricksError as exc:
                    print(f"        get-output failed: {exc}")
                    continue
                RAW.mkdir(parents=True, exist_ok=True)
                (RAW / f"get_output_{t['run_id']}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
                trace = out.get("error_trace") or ""
                print(f"        get-output keys: {sorted(out)}")
                print(f"        error: {short(out.get('error'), 300)} | error_trace length: {len(trace)}")

    print("\nobserved values (these decide the state mapper):")
    for label in sorted(observed):
        print(f"  {label}: {sorted(observed[label])}")


if __name__ == "__main__":
    main()