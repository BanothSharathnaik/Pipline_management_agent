import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from src.generation.llm_client import LLMError, LLMResponse, OllamaClient

OK_BODY = {
    "model": "m",
    "message": {"role": "assistant", "content": "  Hello  "},
    "done": True,
    "done_reason": "stop",
    "prompt_eval_count": 120,
    "eval_count": 30,
}


@pytest.fixture
def serve():
    servers = []

    def start(status=200, body=None, raw=None, delay=0.0):
        state = {"requests": []}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                state["requests"].append(
                    {"path": self.path, "body": json.loads(self.rfile.read(length))}
                )
                if delay:
                    time.sleep(delay)
                payload = raw if raw is not None else json.dumps(body).encode()
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass

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


def test_success_parses_and_sends_expected_request(serve):
    url, state = serve(body=OK_BODY)
    r = OllamaClient(model="m", base_url=url).generate("SYS", "USER")
    assert r.text == "Hello"
    assert (r.prompt_tokens, r.completion_tokens, r.done_reason) == (120, 30, "stop")
    assert r.num_ctx == 4096 and r.elapsed_s >= 0

    sent = state["requests"][0]
    assert sent["path"] == "/api/chat"
    body = sent["body"]
    assert body["model"] == "m" and body["stream"] is False
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert body["messages"][0]["content"] == "SYS" and body["messages"][1]["content"] == "USER"
    assert body["options"] == {"num_ctx": 4096, "num_predict": 700, "temperature": 0.0, "seed": 0}


def test_trailing_slash_in_url_is_fine(serve):
    url, state = serve(body=OK_BODY)
    OllamaClient(model="m", base_url=url + "/").generate("s", "u")
    assert state["requests"][0]["path"] == "/api/chat"


def test_http_404_mentions_pull(serve):
    url, _ = serve(status=404, body={"error": "model 'nope' not found"})
    with pytest.raises(LLMError) as exc:
        OllamaClient(model="nope", base_url=url).generate("s", "u")
    assert "404" in str(exc.value) and "ollama pull nope" in str(exc.value)
    assert "not found" in str(exc.value)


def test_http_500_plain_text(serve):
    url, _ = serve(status=500, raw=b"internal kaboom")
    with pytest.raises(LLMError, match="500"):
        OllamaClient(model="m", base_url=url).generate("s", "u")


def test_invalid_json(serve):
    url, _ = serve(raw=b"this is not json")
    with pytest.raises(LLMError, match="invalid JSON"):
        OllamaClient(model="m", base_url=url).generate("s", "u")


def test_error_field_in_200_response(serve):
    url, _ = serve(body={"error": "boom"})
    with pytest.raises(LLMError, match="boom"):
        OllamaClient(model="m", base_url=url).generate("s", "u")


def test_empty_content(serve):
    url, _ = serve(body={"message": {"role": "assistant", "content": "   "}})
    with pytest.raises(LLMError, match="empty"):
        OllamaClient(model="m", base_url=url).generate("s", "u")


def test_missing_message(serve):
    url, _ = serve(body={"done": True})
    with pytest.raises(LLMError, match="empty"):
        OllamaClient(model="m", base_url=url).generate("s", "u")


def test_missing_token_counts_become_none(serve):
    url, _ = serve(body={"message": {"role": "assistant", "content": "hi"}})
    r = OllamaClient(model="m", base_url=url).generate("s", "u")
    assert r.prompt_tokens is None and r.completion_tokens is None and r.done_reason is None
    assert r.context_nearly_full is False and r.cut_off is False


def test_timeout(serve):
    url, _ = serve(body=OK_BODY, delay=1.0)
    with pytest.raises(LLMError, match="timed out"):
        OllamaClient(model="m", base_url=url, timeout_s=0.2).generate("s", "u")


def test_connection_refused():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with pytest.raises(LLMError, match="cannot reach Ollama"):
        OllamaClient(model="m", base_url=f"http://127.0.0.1:{port}", timeout_s=2).generate("s", "u")


def test_blank_prompts_rejected():
    c = OllamaClient()
    with pytest.raises(ValueError):
        c.generate("  ", "user")
    with pytest.raises(ValueError):
        c.generate("system", "")


def test_constructor_validation():
    for kwargs in (
        {"model": "  "},
        {"base_url": "ftp://x"},
        {"base_url": "not a url"},
        {"timeout_s": 0},
        {"num_ctx": 10},
        {"num_predict": 0},
        {"temperature": -1},
    ):
        with pytest.raises(ValueError):
            OllamaClient(**kwargs)


def test_from_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "other:1b")
    monkeypatch.setenv("OLLAMA_URL", "http://example.invalid:9999")
    c = OllamaClient.from_env()
    assert (c.model, c.base_url) == ("other:1b", "http://example.invalid:9999")
    monkeypatch.delenv("OLLAMA_MODEL")
    monkeypatch.delenv("OLLAMA_URL")
    c = OllamaClient.from_env(timeout_s=5)
    assert (c.model, c.base_url, c.timeout_s) == ("llama3.2:3b", "http://localhost:11434", 5)


def test_response_flags():
    base = dict(text="t", model="m", elapsed_s=1.0, num_ctx=1000)
    assert LLMResponse(**base, prompt_tokens=100, completion_tokens=50, done_reason="length").cut_off
    assert not LLMResponse(**base, prompt_tokens=100, completion_tokens=50, done_reason="stop").cut_off
    assert LLMResponse(**base, prompt_tokens=900, completion_tokens=100, done_reason="stop").context_nearly_full
    assert not LLMResponse(**base, prompt_tokens=900, completion_tokens=99, done_reason="stop").context_nearly_full