"""Shared FastAPI dependencies: service container, DB session, current user."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime

from fastapi import Depends, Request
from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.agents.runner import AgentManager
from app.agents.suites import AgentSuiteRepository
from app.agents.work import WorkAgentService
from app.arena.runner import ArenaManager
from app.arena.service import ArenaStore
from app.benchmark.datasets import SuiteRepository
from app.benchmark.runner import BenchmarkManager
from app.core.config import Settings
from app.core.errors import AuthError, ForbiddenError
from app.core.jobs import JobLock
from app.core.security import hash_token
from app.db.base import utcnow
from app.ml.intent_baseline import IntentBaseline
from app.ml.tracking import Tracker
from app.models.core import ApiToken, User
from app.services.chat_service import ChatService
from app.services.document_service import DocumentService
from app.services.model_registry import ModelRegistry
from app.services.rag_service import RagService
from app.services.scheduler import ScheduleManager
from app.services.settings_service import SettingsService
from app.services.setup_service import SetupService


@dataclass
class AppContext:
    settings: Settings
    engine: Engine
    SessionLocal: sessionmaker[Session]
    settings_service: SettingsService
    registry: ModelRegistry
    documents: DocumentService
    rag: RagService
    chat: ChatService
    suites: SuiteRepository
    tracker: Tracker
    benchmarks: BenchmarkManager
    baseline: IntentBaseline
    job_lock: JobLock
    arena_store: ArenaStore
    arena: ArenaManager
    agent_suites: AgentSuiteRepository
    agents: AgentManager
    started_at: datetime
    migration: dict = field(default_factory=dict)
    setup: SetupService | None = None
    scheduler: ScheduleManager | None = None
    work_agent: WorkAgentService | None = None


def get_ctx(request: Request) -> AppContext:
    return request.app.state.ctx


def get_db(ctx: AppContext = Depends(get_ctx)) -> Iterator[Session]:
    db = ctx.SessionLocal()
    try:
        yield db
    finally:
        db.close()


SCOPES = {
    "read": "Read results, models, suites and settings (GET requests)",
    "chat": "Chat, document Q&A and the OpenAI-compatible API (/v1)",
    "documents": "Upload, re-index and delete documents",
    "evaluate": "Start and cancel benchmark, Model Arena and Agent Arena runs; schedules",
    "agent": "Run the Work Agent (tools that need approval stay gated)",
    "admin": "Everything – only for administrator accounts",
}


def required_scope(method: str, path: str) -> str:
    if path.startswith("/v1/"):
        return "chat"
    if path.startswith("/api/system") or path.startswith("/api/setup/models/") or path.startswith("/api/auth/users"):
        return "admin" if method not in ("GET", "HEAD") or "/audit" in path or "/backups" in path else "read"
    if method in ("GET", "HEAD", "OPTIONS"):
        return "read"
    if path.startswith(("/api/chat", "/api/documents/ask", "/api/documents/retrieval-eval")):
        return "chat"
    if path.startswith("/api/documents"):
        return "documents"
    if path.startswith(("/api/benchmarks", "/api/arena", "/api/agents", "/api/ml", "/api/schedules", "/api/bundles")):
        return "evaluate"
    if path.startswith("/api/work-agent"):
        return "agent"
    return "admin"


def current_user(request: Request, ctx: AppContext = Depends(get_ctx), db: Session = Depends(get_db)) -> User:
    if not ctx.settings.auth_enabled:
        user = db.scalar(select(User).where(User.username == "local"))
        assert user is not None
        request.state.token = None
        return user
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("bearer "):
        raise AuthError("Login required.")
    token = header.split(" ", 1)[1].strip()
    row = db.scalar(select(ApiToken).where(ApiToken.token_hash == hash_token(token)))
    now = utcnow()
    if row is None or row.expires_at < now or row.revoked_at is not None:
        raise AuthError("Session expired, revoked or invalid. Please log in again.")
    user = db.get(User, row.user_id)
    if user is None:
        raise AuthError("User no longer exists.")
    if row.kind == "api":
        scopes = set(row.scopes or [])
        need = required_scope(request.method, request.url.path)
        if "admin" not in scopes and need not in scopes:
            raise ForbiddenError(f"This API token does not have the '{need}' scope.",
                                 hint="Create a token with that scope under Settings → API tokens.")
    if row.last_used_at is None or (now - row.last_used_at).total_seconds() > 60:
        row.last_used_at = now
        db.commit()
    request.state.token = {"id": row.id, "kind": row.kind, "scopes": row.scopes}
    return user


def require_admin(request: Request, user: User = Depends(current_user)) -> User:
    if not user.is_admin:
        raise ForbiddenError("Administrator rights are required for this action.")
    tok = getattr(request.state, "token", None)
    if tok and tok.get("kind") == "api" and "admin" not in set(tok.get("scopes") or []):
        raise ForbiddenError("This API token does not have the 'admin' scope.")
    return user
