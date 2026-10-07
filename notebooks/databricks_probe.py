"""Read-only probe: one GET to the Jobs API to see the real response shape.
Prints the host (never the token). Saves the raw response under data/raw_samples/ (git-ignored)."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode, urlparse

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw_samples" / "runs_list_sample.json"


def read_config() -> tuple[str, str]:
    host = (os.environ.get("DATABRICKS_HOST") or "").strip().rstrip("/")
    token = (os.environ.get("DATABRICKS_TOKEN") or "").strip()
    if not host or not token:
        raise SystemExit("DATABRICKS_HOST and DATABRICKS_TOKEN must be set "
                         "(fill .env, then: set -a; source .env; set +a)")
    p = urlparse(host)
    if p.scheme != "https" or not p.netloc or p.path not in ("", "/"):
        raise SystemExit("DATABRICKS_HOST must look like https://dbc-xxxx.cloud.databricks.com (no path)")
    return f"{p.scheme}://{p.netloc}", token


def get(host: str, token: str, path: str, params: dict) -> tuple[int, object]:
    url = f"{host}{path}?{urlencode(params)}"
    request = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {token}", "Accept": "application/json"}, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(2000).decode("utf-8", errors="replace")[:300]
    except urllib.error.URLError as exc:
        raise SystemExit(f"cannot reach {host}: {exc.reason}")


def shape(value, depth: int = 0):
    """Key names and value types only (first list element), so no values are shown."""
    if isinstance(value, dict):
        return "{...}" if depth >= 4 else {k: shape(v, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [shape(value[0], depth + 1)] if value else []
    return type(value).__name__


def ms(value) -> str:
    if not isinstance(value, (int, float)) or value <= 0:
        return f"{value!r}"
    return datetime.fromtimestamp(value / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def main() -> None:
    host, token = read_config()
    print(f"host: {host}")
    status, data = get(host, token, "/api/2.2/jobs/runs/list", {"limit": 10, "expand_tasks": "true"})
    print(f"HTTP {status}")
    if status != 200:
        hints = {401: "token rejected: check it was copied fully, has not expired, and matches this host",
                 403: "token valid but not allowed to list runs",
                 404: "endpoint not found: check the host URL",
                 429: "rate limited: wait a minute and retry"}
        print(hints.get(status, "unexpected status"))
        print(f"server said: {data}")
        return

    assert isinstance(data, dict)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=2), encoding="utf-8")
    runs = data.get("runs", [])
    print(f"runs returned: {len(runs)} | next_page_token present: {'next_page_token' in data} "
          f"| top-level keys: {sorted(data)}")
    print(f"raw response saved to {OUT.relative_to(ROOT)} (git-ignored)\n")
    if not runs:
        print("no runs: run the demo jobs first (Step A)")
        return

    print("shape of the first run (names and types only):")
    print(json.dumps(shape(runs[0]), indent=2))

    print("\nper-run summary:")
    for r in runs:
        st = r.get("state") or {}
        print(f"run {r.get('run_id')} | job {r.get('job_id')} | life_cycle {st.get('life_cycle_state')} | "
              f"result {st.get('result_state')} | start {ms(r.get('start_time'))} | end {ms(r.get('end_time'))}")
        message = (st.get("state_message") or "")[:200]
        if message:
            print(f"    state_message: {message}")
        for t in r.get("tasks") or []:
            ts = t.get("state") or {}
            deps = [d.get("task_key") for d in (t.get("depends_on") or []) if isinstance(d, dict)]
            print(f"    task {t.get('task_key')}: life_cycle {ts.get('life_cycle_state')} | "
                  f"result {ts.get('result_state')} | depends_on {deps}")


if __name__ == "__main__":
    main()