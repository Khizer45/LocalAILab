"""Model discovery and access control.

Model identifiers used across the app:
* ``llama3.2:3b``           – a model served by the local Ollama runtime
* ``cloud::gemma4:31b``     – a model on Ollama Cloud (ollama.com API)

Local Ollama can also *proxy* cloud models (tags such as ``gemma4:cloud`` or
``gpt-oss:120b-cloud``). Those run remotely even though they are listed by the
local runtime, so they are classified as cloud models too.

Rules enforced by :meth:`ModelRegistry.resolve`:
* the model must exist and must not be an embedding-only model (for chat);
* if an allow-list is configured, the model must be on it;
* in Offline-Only mode every cloud model is refused.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from app.core.config import Settings
from app.core.errors import AppError, ModelNotAvailable, OfflineModeViolation, ValidationFailed
from app.services.ollama_client import OllamaClient
from app.services.settings_service import SettingsService

log = logging.getLogger("app.models")

CLOUD_PREFIX = "cloud::"
EMBED_NAME_HINTS = ("embed", "minilm", "bge-", "bge:", "e5-", "gte-", "arctic-embed", "granite-embedding")


def is_cloud_tag(name: str, entry: dict | None = None) -> bool:
    """True if a model listed by *local* Ollama actually runs in the cloud."""
    if entry and (entry.get("remote_host") or entry.get("remote_model")):
        return True
    tag = name.split(":", 1)[1] if ":" in name else ""
    return tag == "cloud" or tag.endswith("-cloud")


@dataclass
class ResolvedModel:
    ref: str
    name: str  # name sent to the runtime
    provider: str  # local | cloud
    runs_in_cloud: bool
    client: OllamaClient
    digest: str | None = None
    parameter_size: str | None = None
    quantization: str | None = None


class ModelRegistry:
    CACHE_TTL_S = 10.0

    def __init__(self, settings: Settings, settings_service: SettingsService, local: OllamaClient,
                 cloud: OllamaClient | None = None):
        self.settings = settings
        self.settings_service = settings_service
        self.local = local
        self.cloud = cloud
        self._lock = threading.Lock()
        self._cache: tuple[float, list[dict], dict] | None = None
        self._caps: dict[str, list[str]] = {}

    # ------------------------------------------------------------ discovery
    def _capabilities(self, name: str, digest: str | None) -> list[str]:
        key = f"{name}@{digest}"
        if key in self._caps:
            return self._caps[key]
        caps: list[str] = []
        try:
            caps = list(self.local.show(name).get("capabilities") or [])
        except AppError:
            caps = []
        if not caps:
            caps = ["embedding"] if any(h in name.lower() for h in EMBED_NAME_HINTS) else ["completion"]
        self._caps[key] = caps
        return caps

    def invalidate(self) -> None:
        with self._lock:
            self._cache = None

    def list_models(self, *, force: bool = False) -> tuple[list[dict], dict]:
        """Return (models, status). Never raises: runtime problems go into status."""
        with self._lock:
            if not force and self._cache and time.monotonic() - self._cache[0] < self.CACHE_TTL_S:
                return self._decorate(self._cache[1]), self._cache[2]

        rs = self.settings_service.get()
        models: list[dict] = []
        status: dict = {"local": {"reachable": False, "base_url": self.local.base_url}, "cloud": {"enabled": False}}
        try:
            status["local"]["version"] = self.local.version()
            for m in self.local.tags():
                name = m.get("name") or m.get("model")
                if not name:
                    continue
                details = m.get("details") or {}
                cloud = is_cloud_tag(name, m)
                caps = ["completion"] if cloud else self._capabilities(name, m.get("digest"))
                models.append({
                    "id": name,
                    "name": name,
                    "provider": "cloud" if cloud else "local",
                    "location": "cloud (via local Ollama)" if cloud else "this computer",
                    "size_bytes": m.get("size"),
                    "digest": m.get("digest"),
                    "modified_at": m.get("modified_at"),
                    "family": details.get("family"),
                    "parameter_size": details.get("parameter_size"),
                    "quantization": details.get("quantization_level"),
                    "capabilities": caps,
                    "is_embedding": "embedding" in caps and "completion" not in caps,
                })
            status["local"]["reachable"] = True
        except AppError as exc:
            status["local"]["error"] = exc.message
            status["local"]["hint"] = exc.hint

        cloud_status: dict = {"enabled": bool(self.cloud), "configured": bool(self.cloud)}
        if self.cloud and not rs.offline_only:
            try:
                for m in self.cloud.tags():
                    name = m.get("name") or m.get("model")
                    if not name:
                        continue
                    details = m.get("details") or {}
                    models.append({
                        "id": CLOUD_PREFIX + name,
                        "name": name,
                        "provider": "cloud",
                        "location": "Ollama Cloud",
                        "size_bytes": m.get("size"),
                        "digest": m.get("digest"),
                        "modified_at": m.get("modified_at"),
                        "family": details.get("family"),
                        "parameter_size": details.get("parameter_size"),
                        "quantization": details.get("quantization_level"),
                        "capabilities": ["completion"],
                        "is_embedding": False,
                    })
                cloud_status["reachable"] = True
            except AppError as exc:
                cloud_status["reachable"] = False
                cloud_status["error"] = exc.message
        elif self.cloud:
            cloud_status["note"] = "Hidden while Offline-Only mode is on."
        status["cloud"] = cloud_status

        with self._lock:
            self._cache = (time.monotonic(), models, status)
        return self._decorate(models), status

    def _decorate(self, models: list[dict]) -> list[dict]:
        rs = self.settings_service.get()
        allow = set(rs.allowed_models)
        out = []
        for m in models:
            m = dict(m)
            reason = None
            if m["provider"] == "cloud" and rs.offline_only:
                reason = "Cloud model – blocked in Offline-Only mode"
            elif allow and not m["is_embedding"] and m["id"] not in allow:
                reason = "Not in the allowed-models list"
            m["allowed"] = reason is None
            m["blocked_reason"] = reason
            out.append(m)
        return out

    # ------------------------------------------------------------ resolution
    def resolve(self, ref: str, *, purpose: str = "chat") -> ResolvedModel:
        ref = (ref or "").strip()
        if not ref:
            rs = self.settings_service.get()
            ref = rs.default_model
            if not ref:
                raise ValidationFailed("No model selected.", hint="Pick a model or set a default model in Settings.")
        rs = self.settings_service.get()
        models, status = self.list_models()
        match = next((m for m in models if m["id"] == ref), None)
        if match is None:
            if ref.startswith(CLOUD_PREFIX):
                if rs.offline_only:
                    raise OfflineModeViolation(f"'{ref}' is a cloud model and Offline-Only mode is on.",
                                               hint="Disable Offline-Only mode in Settings to use cloud models.")
                raise ModelNotAvailable(f"Cloud model '{ref}' is not available.",
                                        hint="Set OLLAMA_CLOUD_ENABLED=true and OLLAMA_API_KEY in the backend .env file.")
            if not status["local"].get("reachable"):
                from app.core.errors import RuntimeUnavailable

                raise RuntimeUnavailable(status["local"].get("error", "Ollama is not reachable."),
                                         hint=status["local"].get("hint"))
            raise ModelNotAvailable(f"Model '{ref}' is not installed.", hint=f"Download it first: ollama pull {ref}")
        if match["provider"] == "cloud" and rs.offline_only:
            raise OfflineModeViolation(f"'{ref}' runs in the cloud and Offline-Only mode is on.",
                                       hint="Disable Offline-Only mode in Settings to use cloud models.")
        if purpose == "chat" and match["is_embedding"]:
            raise ValidationFailed(f"'{ref}' is an embedding model and cannot chat.")
        if purpose == "chat" and not match["allowed"]:
            raise OfflineModeViolation(match["blocked_reason"] or "Model not allowed.", code="model_not_allowed")
        is_cloud_api = ref.startswith(CLOUD_PREFIX)
        return ResolvedModel(
            ref=ref,
            name=match["name"],
            provider=match["provider"],
            runs_in_cloud=match["provider"] == "cloud",
            client=self.cloud if is_cloud_api and self.cloud else self.local,
            digest=match.get("digest"),
            parameter_size=match.get("parameter_size"),
            quantization=match.get("quantization"),
        )

    def loaded_models(self) -> list[dict]:
        try:
            return self.local.ps()
        except AppError:
            return []
