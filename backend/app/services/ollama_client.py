"""Thin, synchronous client for the Ollama HTTP API.

Every request goes through the offline network guard. Errors from httpx and
from Ollama are translated into readable :mod:`app.core.errors` exceptions.

Ollama API reference: https://docs.ollama.com/api
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any

import httpx

from app.core.errors import (
    AppError,
    InferenceTimeout,
    InsufficientResources,
    ModelNotAvailable,
    OfflineModeViolation,
    RuntimeUnavailable,
)
from app.core.network_guard import OfflineNetworkBlocked, guarded_http_client

log = logging.getLogger("app.ollama")


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        connect_timeout_s: float = 5.0,
        label: str = "local",
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.label = label
        self.connect_timeout_s = connect_timeout_s
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        self._http = guarded_http_client(
            self.base_url,
            timeout=httpx.Timeout(30.0, connect=connect_timeout_s),
            headers=headers,
            transport=transport,
        )

    def close(self) -> None:
        self._http.close()

    # ------------------------------------------------------------- helpers
    def _hint(self) -> str:
        if self.label == "local":
            return (
                f"Start Ollama (run `ollama serve` or open the Ollama app) and check that it listens on "
                f"{self.base_url}. Set OLLAMA_BASE_URL in .env if it runs elsewhere."
            )
        return "Check your internet connection, OLLAMA_API_KEY and that Offline-Only mode is disabled."

    def _translate(self, exc: Exception) -> AppError:
        if isinstance(exc, AppError):
            return exc
        if isinstance(exc, OfflineNetworkBlocked):
            return OfflineModeViolation(str(exc))
        if isinstance(exc, httpx.TimeoutException) and not isinstance(exc, httpx.ConnectTimeout):
            return InferenceTimeout("The model did not respond within the configured timeout.",
                                    hint="Increase the timeout in Settings or use a smaller model.")
        if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.RemoteProtocolError)):
            return RuntimeUnavailable(f"Cannot reach the {self.label} Ollama runtime at {self.base_url}.", hint=self._hint())
        return RuntimeUnavailable(f"Ollama request failed: {type(exc).__name__}", hint=self._hint())

    def _error_from_body(self, status: int, body: str, model: str | None = None) -> AppError:
        try:
            message = json.loads(body).get("error", body)
        except (ValueError, AttributeError):
            message = body
        message = str(message)[:400] or f"HTTP {status}"
        low = message.lower()
        if status == 404 or "not found" in low:
            name = f" '{model}'" if model else ""
            return ModelNotAvailable(
                f"Model{name} is not available: {message}",
                hint=f"Download it first: ollama pull {model}" if model and self.label == "local" else None,
            )
        if "memory" in low and ("require" in low or "insufficient" in low or "not enough" in low):
            return InsufficientResources(message, hint="Choose a smaller / more quantized model or reduce the context length.")
        if status in (401, 403):
            return AppError(f"Ollama rejected the request: {message}", code="runtime_auth_failed", status_code=502,
                            hint="Check OLLAMA_API_KEY in the backend .env file.")
        return AppError(f"Ollama error: {message}", code="runtime_error", status_code=502)

    def _get(self, path: str) -> Any:
        try:
            r = self._http.get(path)
        except Exception as exc:  # noqa: BLE001
            raise self._translate(exc) from exc
        if r.status_code >= 400:
            raise self._error_from_body(r.status_code, r.text)
        return r.json()

    def _post(self, path: str, payload: dict, *, timeout: float | None = None, model: str | None = None) -> Any:
        try:
            r = self._http.post(path, json=payload, timeout=httpx.Timeout(timeout or 60.0, connect=self.connect_timeout_s))
        except Exception as exc:  # noqa: BLE001
            raise self._translate(exc) from exc
        if r.status_code >= 400:
            raise self._error_from_body(r.status_code, r.text, model)
        return r.json()

    # ------------------------------------------------------------- endpoints
    def version(self) -> str:
        return str(self._get("/api/version").get("version", "unknown"))

    def tags(self) -> list[dict]:
        return list(self._get("/api/tags").get("models", []))

    def ps(self) -> list[dict]:
        return list(self._get("/api/ps").get("models", []))

    def show(self, model: str) -> dict:
        return self._post("/api/show", {"model": model}, timeout=30, model=model)

    def embed(self, model: str, inputs: list[str], *, timeout: float = 120.0) -> list[list[float]]:
        data = self._post("/api/embed", {"model": model, "input": inputs, "truncate": True}, timeout=timeout, model=model)
        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(inputs):
            raise AppError("Embedding model returned an unexpected response.", code="runtime_error", status_code=502)
        return embeddings

    def chat(self, payload: dict, *, timeout: float) -> dict:
        """Non-streaming POST /api/chat (used for tool calling)."""
        return self._post("/api/chat", {**payload, "stream": False}, timeout=timeout, model=payload.get("model"))

    def chat_stream(self, payload: dict, *, timeout: float) -> Iterator[dict]:
        """Yield the raw NDJSON objects streamed by POST /api/chat."""
        payload = {**payload, "stream": True}
        model = payload.get("model")
        try:
            with self._http.stream(
                "POST",
                "/api/chat",
                json=payload,
                timeout=httpx.Timeout(timeout, connect=self.connect_timeout_s),
            ) as r:
                if r.status_code >= 400:
                    r.read()
                    raise self._error_from_body(r.status_code, r.text, model)
                for line in r.iter_lines():
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                    except ValueError:
                        continue
                    if "error" in obj:
                        raise self._error_from_body(500, json.dumps(obj), model)
                    yield obj
        except AppError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise self._translate(exc) from exc
