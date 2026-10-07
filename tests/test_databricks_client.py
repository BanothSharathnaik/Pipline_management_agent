import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

from src.monitoring.databricks_client import DatabricksClient, DatabricksError

TOKEN = "SECRET-TOKEN-123"


@pytest.fixture
def serve():
    servers = []

    def start(script):
        state = {"requests": [], "i": 0}
        lock = threading.Lock()

        class Handler(BaseHTTPRequestHandler):
            def handle_any(self):
                with lock:
                    entry = script[min(state["i"], len(script) - 1)]
                    state["i"] += 1
                    parsed = urlparse(self.path)
                    state["requests"].append({
                        "method": self.command,
                        "path": parsed.path,
                        "query": parse_qs(parsed.query),
                        "auth": self.headers.get("Authorization"),
                    })
                if entry.get("delay"):
                    time.sleep(entry["delay"])
                body = entry.get("json", {})
                if entry.get("echo_auth"):
                    body = {"message": f"got {self.headers.get('Authorization')}"}
                payload = entry["raw"] if "raw" in entry else json.dumps(body).encode()
                try:
                    self.send_response(entry.get("status", 200))
                    for k, v in entry.get("headers", {}).items():
                        self.send_header(k, v)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = handle_any

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{server.server_address[1]}", state

    yield start
    for s in servers:
        s.shutdown()
        s.server_close()


def client(url, **kw):
    sleeps = []
    return DatabricksClient(url, TOKEN, sleep=sleeps.append, **kw), sleeps


def ids(result):
    return [r["run_id"] for r in result.runs]


# --- requests ---------------------------------------------------------------------------

def test_list_runs_request_shape(serve):
    url, state = serve([{"json": {"runs": [{"run_id": 1}]}}])
    c, _ = client(url)
    res = c.list_runs()
    assert ids(res) == [1] and res.pages == 1 and res.truncated is False
    req = state["requests"][0]
    assert req["method"] == "GET" and req["path"] == "/api/2.2/jobs/runs/list"
    assert req["query"] == {"limit": ["25"], "expand_tasks": ["true"]}
    assert req["auth"] == f"Bearer {TOKEN}"


def test_filters_become_query_params(serve):
    url, state = serve([{"json": {}}])
    c, _ = client(url)
    c.list_runs(job_id=7, completed_only=True, start_time_from_ms=1000,
                start_time_to_ms=2000, page_size=10, expand_tasks=False)
    assert state["requests"][0]["query"] == {
        "limit": ["10"], "expand_tasks": ["false"], "job_id": ["7"],
        "completed_only": ["true"], "start_time_from": ["1000"], "start_time_to": ["2000"],
    }


def test_empty_response_means_no_runs(serve):
    url, _ = serve([{"json": {}}])
    res = client(url)[0].list_runs()
    assert res.runs == [] and res.pages == 1 and res.truncated is False


def test_get_run_and_output_requests(serve):
    url, state = serve([{"json": {"run_id": 42}}])
    c, _ = client(url)
    assert c.get_run(42) == {"run_id": 42}
    c.get_run_output(43)
    first, second = state["requests"]
    assert (first["path"], first["query"]) == ("/api/2.2/jobs/runs/get", {"run_id": ["42"]})
    assert (second["path"], second["query"]) == ("/api/2.2/jobs/runs/get-output", {"run_id": ["43"]})


# --- pagination -------------------------------------------------------------------------

def test_pagination_follows_tokens(serve):
    url, state = serve([
        {"json": {"runs": [{"run_id": 1}, {"run_id": 2}], "next_page_token": "T2"}},
        {"json": {"runs": [{"run_id": 3}]}},
    ])
    res = client(url)[0].list_runs()
    assert ids(res) == [1, 2, 3] and res.pages == 2 and res.truncated is False
    assert "page_token" not in state["requests"][0]["query"]
    assert state["requests"][1]["query"]["page_token"] == ["T2"]


def test_duplicates_across_pages_are_dropped_and_counted(serve):
    url, _ = serve([
        {"json": {"runs": [{"run_id": 1}, {"run_id": 2}], "next_page_token": "T2"}},
        {"json": {"runs": [{"run_id": 2}, {"run_id": 3}]}},
    ])
    res = client(url)[0].list_runs()
    assert ids(res) == [1, 2, 3] and res.duplicates_dropped == 1


def test_max_pages_truncates_and_says_so(serve):
    url, state = serve([
        {"json": {"runs": [{"run_id": 1}], "next_page_token": "a"}},
        {"json": {"runs": [{"run_id": 2}], "next_page_token": "b"}},
        {"json": {"runs": [{"run_id": 3}]}},
    ])
    res = client(url)[0].list_runs(max_pages=2)
    assert ids(res) == [1, 2] and res.pages == 2 and res.truncated is True
    assert len(state["requests"]) == 2


def test_repeated_page_token_raises(serve):
    url, _ = serve([{"json": {"runs": [{"run_id": 1}], "next_page_token": "T"}},
                    {"json": {"runs": [{"run_id": 2}], "next_page_token": "T"}}])
    with pytest.raises(DatabricksError, match="did not advance"):
        client(url)[0].list_runs()


def test_max_runs_stops_early_and_reports_truncation(serve):
    url, state = serve([{"json": {"runs": [{"run_id": 1}, {"run_id": 2}, {"run_id": 3}],
                                  "next_page_token": "T"}}])
    res = client(url)[0].list_runs(max_runs=2)
    assert ids(res) == [1, 2] and res.truncated is True and len(state["requests"]) == 1


def test_max_runs_exactly_all_runs_is_not_truncated(serve):
    url, _ = serve([{"json": {"runs": [{"run_id": 1}, {"run_id": 2}, {"run_id": 3}]}}])
    res = client(url)[0].list_runs(max_runs=3)
    assert ids(res) == [1, 2, 3] and res.truncated is False


# --- errors that must not be retried --------------------------------------------------------

@pytest.mark.parametrize("status,word", [(401, "token"), (403, "allowed"), (404, "not found")])
def test_client_errors_are_not_retried(serve, status, word):
    url, state = serve([{"status": status, "json": {"error_code": "X", "message": "nope"}}])
    c, sleeps = client(url)
    with pytest.raises(DatabricksError) as exc:
        c.list_runs()
    assert exc.value.status == status and word in str(exc.value) and "nope" in str(exc.value)
    assert len(state["requests"]) == 1 and sleeps == []


def test_invalid_json_not_retried(serve):
    url, state = serve([{"raw": b"not json"}])
    with pytest.raises(DatabricksError, match="invalid JSON"):
        client(url)[0].list_runs()
    assert len(state["requests"]) == 1


def test_non_object_json_rejected(serve):
    url, _ = serve([{"raw": b"[]"}])
    with pytest.raises(DatabricksError, match="unexpected response shape"):
        client(url)[0].list_runs()


def test_run_without_integer_id_rejected(serve):
    url, _ = serve([{"json": {"runs": [{"nope": 1}]}}])
    with pytest.raises(DatabricksError, match="run_id"):
        client(url)[0].list_runs()


# --- retries ----------------------------------------------------------------------------------

def test_429_honors_retry_after(serve):
    url, state = serve([{"status": 429, "headers": {"Retry-After": "2"}, "json": {"message": "slow"}},
                        {"json": {"runs": []}}])
    c, sleeps = client(url)
    assert c.list_runs().runs == [] and sleeps == [2.0] and len(state["requests"]) == 2


def test_retry_after_is_capped(serve):
    url, _ = serve([{"status": 429, "headers": {"Retry-After": "9999"}}, {"json": {}}])
    c, sleeps = client(url)
    c.list_runs()
    assert sleeps == [60.0]


def test_retry_after_http_date_falls_back_to_backoff(serve):
    url, _ = serve([{"status": 503, "headers": {"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}},
                    {"json": {}}])
    c, sleeps = client(url, backoff_s=1.0)
    c.list_runs()
    assert sleeps == [1.0]


def test_server_errors_back_off_exponentially_then_succeed(serve):
    url, state = serve([{"status": 500}, {"status": 502}, {"json": {"runs": [{"run_id": 5}]}}])
    c, sleeps = client(url, backoff_s=1.0)
    assert ids(c.list_runs()) == [5] and sleeps == [1.0, 2.0] and len(state["requests"]) == 3


def test_persistent_503_gives_up_after_max_retries(serve):
    url, state = serve([{"status": 503, "json": {"message": "down"}}])
    c, sleeps = client(url, max_retries=2, backoff_s=1.0)
    with pytest.raises(DatabricksError) as exc:
        c.list_runs()
    assert exc.value.status == 503 and len(state["requests"]) == 3 and sleeps == [1.0, 2.0]


def test_connection_refused_is_retried_then_reported():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    c, sleeps = client(f"http://127.0.0.1:{port}", max_retries=2)
    with pytest.raises(DatabricksError, match="cannot reach"):
        c.list_runs()
    assert sleeps == [1.0, 2.0]


def test_timeout_is_retried_then_reported(serve):
    url, state = serve([{"json": {}, "delay": 1.0}])
    c, sleeps = client(url, timeout_s=0.2, max_retries=1)
    with pytest.raises(DatabricksError):
        c.list_runs()
    assert len(state["requests"]) == 2 and sleeps == [1.0]


# --- secrets and read-only -------------------------------------------------------------------

def test_token_never_appears_in_errors_or_repr(serve):
    url, _ = serve([{"status": 401, "echo_auth": True}])
    c, _ = client(url)
    with pytest.raises(DatabricksError) as exc:
        c.list_runs()
    assert TOKEN not in str(exc.value) and "***" in str(exc.value)
    assert TOKEN not in repr(c) and "***" in repr(c)


def test_only_get_requests_are_ever_sent(serve):
    url, state = serve([{"json": {"runs": [{"run_id": 1}]}}])
    c, _ = client(url)
    c.list_runs()
    c.get_run(1)
    c.get_run_output(2)
    assert {r["method"] for r in state["requests"]} == {"GET"}


def test_public_surface_is_exactly_the_read_operations():
    public = {n for n in dir(DatabricksClient)
              if not n.startswith("_") and callable(getattr(DatabricksClient, n))}
    assert public == {"from_env", "list_runs", "get_run", "get_run_output"}


# --- validation -----------------------------------------------------------------------------

@pytest.mark.parametrize("host", [
    "", "ftp://x.com", "http://example.com", "https://dbc.cloud.databricks.com/path",
    "https://dbc.cloud.databricks.com/?o=1", "not a url",
])
def test_bad_hosts_rejected(host):
    with pytest.raises(ValueError):
        DatabricksClient(host, TOKEN)


def test_good_hosts_accepted():
    DatabricksClient("https://dbc-x.cloud.databricks.com/", TOKEN)
    DatabricksClient("http://127.0.0.1:8080", TOKEN)
    DatabricksClient("http://localhost:8080", TOKEN)


def test_bad_constructor_values():
    for kwargs in ({"timeout_s": 0}, {"max_retries": -1}, {"backoff_s": -1}):
        with pytest.raises(ValueError):
            DatabricksClient("https://dbc-x.cloud.databricks.com", TOKEN, **kwargs)
    with pytest.raises(ValueError):
        DatabricksClient("https://dbc-x.cloud.databricks.com", "  ")


@pytest.mark.parametrize("bad", [0, -1, "5", True, None, 1.5])
def test_bad_ids_rejected(bad):
    c = DatabricksClient("https://dbc-x.cloud.databricks.com", TOKEN)
    with pytest.raises(ValueError):
        c.get_run(bad)
    with pytest.raises(ValueError):
        c.get_run_output(bad)


@pytest.mark.parametrize("kwargs", [
    {"page_size": 0}, {"page_size": 26}, {"max_pages": 0}, {"max_runs": 0},
    {"job_id": 0}, {"active_only": True, "completed_only": True},
    {"start_time_from_ms": 5, "start_time_to_ms": 1}, {"start_time_from_ms": -1},
])
def test_bad_list_arguments_rejected(kwargs):
    c = DatabricksClient("https://dbc-x.cloud.databricks.com", TOKEN)
    with pytest.raises(ValueError):
        c.list_runs(**kwargs)


def test_from_env(monkeypatch):
    monkeypatch.delenv("DATABRICKS_HOST", raising=False)
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)
    with pytest.raises(ValueError, match="must be set"):
        DatabricksClient.from_env()
    monkeypatch.setenv("DATABRICKS_HOST", "https://dbc-x.cloud.databricks.com")
    monkeypatch.setenv("DATABRICKS_TOKEN", TOKEN)
    assert "dbc-x" in repr(DatabricksClient.from_env()) and TOKEN not in repr(DatabricksClient.from_env())