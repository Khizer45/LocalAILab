"""Runtime-editable settings stored in the database (Settings screen)."""

from __future__ import annotations

import threading

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.core.network_guard import guard
from app.models.core import AppSetting

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, concise assistant running fully on the user's own computer. "
    "If you are not sure about something, say so."
)


class RuntimeSettings(BaseModel):
    offline_only: bool = True
    allowed_models: list[str] = Field(default_factory=list, description="Empty list = every installed model is allowed")
    default_model: str = ""
    embedding_model: str = "nomic-embed-text"
    default_system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_upload_mb: int = Field(20, ge=1, le=500)
    max_context_tokens: int = Field(8192, ge=512, le=262144)
    max_output_tokens: int = Field(1024, ge=16, le=32768)
    request_timeout_s: int = Field(180, ge=5, le=3600)
    benchmark_timeout_s: int = Field(120, ge=5, le=3600)
    retrieval_top_k: int = Field(4, ge=1, le=20)
    rag_min_score: float | None = Field(None, ge=0.0, le=1.0, description="None = per-embedder default")
    allow_documents_with_cloud: bool = False


class RuntimeSettingsUpdate(BaseModel):
    offline_only: bool | None = None
    allowed_models: list[str] | None = None
    default_model: str | None = None
    embedding_model: str | None = None
    default_system_prompt: str | None = Field(None, max_length=8000)
    max_upload_mb: int | None = Field(None, ge=1, le=500)
    max_context_tokens: int | None = Field(None, ge=512, le=262144)
    max_output_tokens: int | None = Field(None, ge=16, le=32768)
    request_timeout_s: int | None = Field(None, ge=5, le=3600)
    benchmark_timeout_s: int | None = Field(None, ge=5, le=3600)
    retrieval_top_k: int | None = Field(None, ge=1, le=20)
    rag_min_score: float | None = Field(None, ge=0.0, le=1.0)
    clear_rag_min_score: bool = False
    allow_documents_with_cloud: bool | None = None


class SettingsService:
    def __init__(self, base: Settings, SessionLocal: sessionmaker[Session]):
        self.base = base
        self.SessionLocal = SessionLocal
        self._lock = threading.Lock()
        self._cache: RuntimeSettings | None = None

    def _defaults(self) -> dict:
        b = self.base
        return {
            "offline_only": b.offline_only_default,
            "default_model": b.default_model,
            "embedding_model": b.embedding_model,
            "max_upload_mb": b.max_upload_mb,
            "max_context_tokens": b.max_context_tokens,
            "max_output_tokens": b.max_output_tokens,
            "request_timeout_s": b.request_timeout_s,
            "benchmark_timeout_s": b.benchmark_timeout_s,
            "retrieval_top_k": b.retrieval_top_k,
        }

    def get(self) -> RuntimeSettings:
        with self._lock:
            if self._cache is None:
                values = self._defaults()
                with self.SessionLocal() as db:
                    for row in db.query(AppSetting).all():
                        if row.key in RuntimeSettings.model_fields:
                            values[row.key] = row.value
                self._cache = RuntimeSettings(**values)
                self._apply(self._cache)
            return self._cache

    def update(self, patch: RuntimeSettingsUpdate) -> RuntimeSettings:
        data = patch.model_dump(exclude_unset=True, exclude={"clear_rag_min_score"})
        data = {k: v for k, v in data.items() if v is not None}
        if patch.clear_rag_min_score:
            data["rag_min_score"] = None
        if "allowed_models" in data:
            data["allowed_models"] = sorted({m.strip() for m in data["allowed_models"] if m.strip()})
        current = self.get().model_dump()
        current.update(data)
        new = RuntimeSettings(**current)  # validates
        with self.SessionLocal() as db:
            for key, value in data.items():
                row = db.get(AppSetting, key)
                if row is None:
                    db.add(AppSetting(key=key, value=value))
                else:
                    row.value = value
            db.commit()
        with self._lock:
            self._cache = new
            self._apply(new)
        return new

    def _apply(self, rs: RuntimeSettings) -> None:
        guard.configure(offline_only=rs.offline_only, allowed_hosts=self.base.offline_allowed_hosts)
