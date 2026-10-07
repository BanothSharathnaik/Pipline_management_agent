from __future__ import annotations

import http.client
import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional
from urllib.parse import urlencode, urlparse

log = logging.getLogger(__name__)

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
MAX_PAGE_SIZE = 25          # conservative; not verified against the current API limit
MAX_BACKOFF_S = 30.0
MAX_RETRY_AFTER_S = 60.0
_HINTS = {
    400: "the request was rejected: check the parameters",
    401: "the token was rejected: check it is complete, unexpired and belongs to this workspace",
    403: "the token is valid but not allowed to do this",
    404: "not found: check the host URL and the run id",
}


class DatabricksError(RuntimeError):
    """A Databricks request failed. Messages never contain the token."""

    def __init__(self, message: str, status: Optional[int] = None) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class RunsFetch:
    runs: list[dict]
    pages: int
    truncated: bool            # True when more runs exist than were returned
    duplicates_dropped: int


def _check_id(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _check_ms(name: str, value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer (milliseconds since epoch)")
    return value


class DatabricksClient:
    def __init__(
        self,
        host: str,
        token: str,
        timeout_s: float = 30.0,
        max_retries: int = 3,
        backoff_s: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        host = (host or "").strip().rstrip("/")
        token = (token or "").strip()
        if not host:
            raise ValueError("host must not be blank")
        if not token:
            raise ValueError("token must not be blank")
        parsed = urlparse(host)
        if (parsed.scheme not in ("https", "http") or not parsed.netloc
                or parsed.path not in ("", "/") or parsed.query or parsed.fragment):
            raise ValueError("host must look like https://dbc-xxxx.cloud.databricks.com (no path)")
        if parsed.scheme == "http" and parsed.hostname not in LOOPBACK_HOSTS:
            raise ValueError("the token is only sent over https (plain http is allowed for localhost only)")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if backoff_s < 0:
            raise ValueError("backoff_s must be >= 0")

        self._base = f"{parsed.scheme}://{parsed.netloc}"
        self._token = token
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._backoff_s = backoff_s
        self._sleep = sleep

    def __repr__(self) -> str:
        return f"DatabricksClient(host={self._base!r}, token=***)"

    @classmethod
    def from_env(cls, **kwargs) -> "DatabricksClient":
        host, token = os.environ.get("DATABRICKS_HOST"), os.environ.get("DATABRICKS_TOKEN")
        if not host or not token:
            raise ValueError("DATABRICKS_HOST and DATABRICKS_TOKEN must be set "
                             "(fill .env, then: set -a; source .env; set +a)")
        parsed = urlparse(host.strip())
        if parsed.scheme in ("https", "http") and parsed.netloc:
            host = f"{parsed.scheme}://{parsed.netloc}"
        return cls(host, token, **kwargs)

    # --- public, read-only operations -----------------------------------------------------

    def list_runs(
        self,
        job_id: Optional[int] = None,
        active_only: bool = False,
        completed_only: bool = False,
        start_time_from_ms: Optional[int] = None,
        start_time_to_ms: Optional[int] = None,
        expand_tasks: bool = True,
        page_size: int = MAX_PAGE_SIZE,
        max_pages: int = 40,
        max_runs: Optional[int] = None,
    ) -> RunsFetch:
        if job_id is not None:
            _check_id("job_id", job_id)
        if active_only and completed_only:
            raise ValueError("active_only and completed_only cannot both be set")
        t_from = _check_ms("start_time_from_ms", start_time_from_ms)
        t_to = _check_ms("start_time_to_ms", start_time_to_ms)
        if t_from is not None and t_to is not None and t_from > t_to:
            raise ValueError("start_time_from_ms must not be after start_time_to_ms")
        if not 1 <= page_size <= MAX_PAGE_SIZE:
            raise ValueError(f"page_size must be between 1 and {MAX_PAGE_SIZE}")
        if max_pages < 1:
            raise ValueError("max_pages must be at least 1")
        if max_runs is not None and max_runs < 1:
            raise ValueError("max_runs must be at least 1")

        runs: list[dict] = []
        seen_ids: set[int] = set()
        seen_tokens: set[str] = set()
        duplicates = 0
        pages = 0
        page_token: Optional[str] = None

        while True:
            params: dict[str, object] = {
                "limit": page_size,
                "expand_tasks": "true" if expand_tasks else "false",
            }
            if job_id is not None:
                params["job_id"] = job_id
            if active_only:
                params["active_only"] = "true"
            if completed_only:
                params["completed_only"] = "true"
            if t_from is not None:
                params["start_time_from"] = t_from
            if t_to is not None:
                params["start_time_to"] = t_to
            if page_token:
                params["page_token"] = page_token

            data = self._get("/api/2.2/jobs/runs/list", params)
            pages += 1
            page_runs = data.get("runs") or []
            if not isinstance(page_runs, list):
                raise DatabricksError("unexpected response: 'runs' is not a list")

            for i, run in enumerate(page_runs):
                run_id = run.get("run_id") if isinstance(run, dict) else None
                if isinstance(run_id, bool) or not isinstance(run_id, int):
                    raise DatabricksError("unexpected response: a run has no integer run_id")
                if run_id in seen_ids:
                    duplicates += 1
                    continue
                seen_ids.add(run_id)
                runs.append(run)
                if max_runs is not None and len(runs) >= max_runs:
                    more = bool(page_runs[i + 1:]) or bool(data.get("next_page_token"))
                    return RunsFetch(runs, pages, more, duplicates)

            next_token = data.get("next_page_token")
            if not next_token:
                return RunsFetch(runs, pages, False, duplicates)
            if not isinstance(next_token, str) or next_token in seen_tokens:
                raise DatabricksError("pagination did not advance (repeated or invalid page token)")
            seen_tokens.add(next_token)
            if pages >= max_pages:
                return RunsFetch(runs, pages, True, duplicates)
            page_token = next_token

    def get_run(self, run_id: int) -> dict:
        return self._get("/api/2.2/jobs/runs/get", {"run_id": _check_id("run_id", run_id)})

    def get_run_output(self, task_run_id: int) -> dict:
        """Output and error text of one TASK run. Parent run ids are rejected by Databricks."""
        return self._get("/api/2.2/jobs/runs/get-output",
                         {"run_id": _check_id("task_run_id", task_run_id)})

    # --- internals -------------------------------------------------------------------------

    def _redact(self, text: str) -> str:
        return text.replace(self._token, "***")

    def _wait(self, attempt: int, retry_after: Optional[str], why: str, path: str) -> None:
        delay = min(self._backoff_s * (2 ** attempt), MAX_BACKOFF_S)
        if retry_after:
            try:
                delay = min(max(float(retry_after), 0.0), MAX_RETRY_AFTER_S)
            except ValueError:
                pass  # an HTTP-date value: keep the exponential delay
        log.warning("retrying %s after %s (retry %d of %d) in %.1fs",
                    path, why, attempt + 1, self._max_retries, delay)
        self._sleep(delay)

    def _error_detail(self, exc: urllib.error.HTTPError) -> str:
        try:
            body = exc.read(2000).decode("utf-8", errors="replace")
        except Exception:
            return str(exc.reason or "no details")
        finally:
            exc.close()
        try:
            parsed = json.loads(body)
            if isinstance(parsed, dict):
                message = parsed.get("message") or parsed.get("error")
                if message:
                    code = parsed.get("error_code")
                    return self._redact(f"{code}: {message}" if code else str(message))[:300]
        except json.JSONDecodeError:
            pass
        return self._redact(body.strip())[:300] or "no details"

    def _get(self, path: str, params: dict) -> dict:
        url = f"{self._base}{path}" + (f"?{urlencode(params)}" if params else "")
        attempt = 0
        while True:
            request = urllib.request.Request(
                url,
                headers={"Authorization": f"Bearer {self._token}", "Accept": "application/json"},
                method="GET",
            )
            try:
                with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
                    raw = response.read()
            except urllib.error.HTTPError as exc:
                detail = self._error_detail(exc)
                if exc.code in RETRY_STATUSES and attempt < self._max_retries:
                    retry_after = exc.headers.get("Retry-After") if exc.headers else None
                    self._wait(attempt, retry_after, f"HTTP {exc.code}", path)
                    attempt += 1
                    continue
                hint = _HINTS.get(exc.code)
                raise DatabricksError(
                    f"Databricks returned HTTP {exc.code}: {detail}" + (f". {hint}" if hint else ""),
                    exc.code,
                ) from None
            except (urllib.error.URLError, TimeoutError, http.client.HTTPException, OSError) as exc:
                if attempt < self._max_retries:
                    self._wait(attempt, None, "a connection problem", path)
                    attempt += 1
                    continue
                reason = getattr(exc, "reason", exc)
                raise DatabricksError(f"cannot reach {self._base}: {self._redact(str(reason))}") from None

            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise DatabricksError("Databricks returned invalid JSON") from None
            if not isinstance(data, dict):
                raise DatabricksError("Databricks returned an unexpected response shape")
            return data