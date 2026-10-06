from __future__ import annotations

import http.client
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional, Protocol
from urllib.parse import urlparse

DEFAULT_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:3b"
DEFAULT_NUM_CTX = 4096
DEFAULT_NUM_PREDICT = 700
DEFAULT_TIMEOUT_S = 120.0
PROMPT_BUDGET_CHARS = 8000  # pass to build_prompt(max_chars=...); calibrate with the smoke test


class LLMError(RuntimeError):
    """The language model could not be reached or returned something unusable."""


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    done_reason: Optional[str]
    elapsed_s: float
    num_ctx: int

    @property
    def cut_off(self) -> bool:
        """The answer hit the num_predict limit and was cut short."""
        return self.done_reason == "length"

    @property
    def context_nearly_full(self) -> bool:
        """Prompt plus answer filled the context window, so the prompt may have been truncated."""
        if self.prompt_tokens is None or self.completion_tokens is None:
            return False
        return self.prompt_tokens + self.completion_tokens >= self.num_ctx


class LLM(Protocol):
    model: str

    def generate(self, system: str, user: str) -> LLMResponse: ...


def _as_int(value: object) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read(2000).decode("utf-8", errors="replace")
    except Exception:
        return str(exc.reason or "no details")
    try:
        parsed = json.loads(body)
        if isinstance(parsed, dict) and parsed.get("error"):
            return str(parsed["error"])
    except json.JSONDecodeError:
        pass
    return body.strip()[:300] or str(exc.reason or "no details")


class OllamaClient:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_URL,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        num_ctx: int = DEFAULT_NUM_CTX,
        num_predict: int = DEFAULT_NUM_PREDICT,
        temperature: float = 0.0,
        seed: int = 0,
    ) -> None:
        if not model or not model.strip():
            raise ValueError("model must not be blank")
        parsed = urlparse(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(f"base_url must be an http(s) URL, got {base_url!r}")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if num_ctx < 256:
            raise ValueError("num_ctx must be at least 256")
        if num_predict < 1:
            raise ValueError("num_predict must be at least 1")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")

        self.model = model.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.temperature = temperature
        self.seed = seed
        # An opener with no proxies: prompts contain log text and go to the server we named only.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    @classmethod
    def from_env(cls, **kwargs) -> "OllamaClient":
        return cls(
            model=os.environ.get("OLLAMA_MODEL") or DEFAULT_MODEL,
            base_url=os.environ.get("OLLAMA_URL") or DEFAULT_URL,
            **kwargs,
        )

    def generate(self, system: str, user: str) -> LLMResponse:
        if not system.strip() or not user.strip():
            raise ValueError("system and user prompts must not be blank")

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
                "temperature": self.temperature,
                "seed": self.seed,
            },
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        started = time.perf_counter()
        try:
            with self._opener.open(request, timeout=self.timeout_s) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            hint = f" Pull the model with: ollama pull {self.model}" if exc.code == 404 else ""
            raise LLMError(f"Ollama returned HTTP {exc.code}: {_error_detail(exc)}.{hint}") from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise LLMError(self._timeout_message()) from exc
            raise LLMError(
                f"cannot reach Ollama at {self.base_url}: {exc.reason}. "
                "Is the server running? (brew services start ollama, or open the Ollama app)"
            ) from exc
        except TimeoutError as exc:
            raise LLMError(self._timeout_message()) from exc
        except (http.client.HTTPException, OSError) as exc:
            raise LLMError(f"connection to Ollama failed: {exc}") from exc
        elapsed = time.perf_counter() - started

        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise LLMError("Ollama returned invalid JSON") from exc
        if not isinstance(data, dict):
            raise LLMError("Ollama returned an unexpected response shape")
        if data.get("error"):
            raise LLMError(f"Ollama error: {data['error']}")

        message = data.get("message")
        text = message.get("content") if isinstance(message, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise LLMError("Ollama returned an empty answer")

        return LLMResponse(
            text=text.strip(),
            model=str(data.get("model") or self.model),
            prompt_tokens=_as_int(data.get("prompt_eval_count")),
            completion_tokens=_as_int(data.get("eval_count")),
            done_reason=data.get("done_reason") if isinstance(data.get("done_reason"), str) else None,
            elapsed_s=elapsed,
            num_ctx=self.num_ctx,
        )

    def _timeout_message(self) -> str:
        return (
            f"Ollama did not answer within {self.timeout_s:g}s (timed out). The first request "
            "after idle also loads the model; raise timeout_s or use a smaller model."
        )