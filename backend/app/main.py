"""FastAPI application factory.

Run locally:  uvicorn app.main:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import ipaddress
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

# Python imports the parent package before this module, so app/__init__.py has already applied the
# third-party telemetry opt-outs before FastAPI, ONNX Runtime, MLflow or any ML library is loaded.
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402

from app import __version__  # noqa: E402
from app.agents.runner import AgentManager  # noqa: E402
from app.agents.suites import AgentSuiteRepository  # noqa: E402
from app.agents.work import WorkAgentService  # noqa: E402
from app.api.deps import AppContext  # noqa: E402
from app.api.routes import (  # noqa: E402
    agents,
    arena,
    auth,
    automation,
    backups,
    benchmarks,
    bundles,
    chat,
    documents,
    health,
    ml,
    models,
    openai_compat,
    setup,
    system,
    work_agent,
)
from app.api.routes import settings as settings_routes  # noqa: E402
from app.arena.runner import ArenaManager  # noqa: E402
from app.arena.service import ArenaStore  # noqa: E402
from app.benchmark.datasets import SuiteRepository  # noqa: E402
from app.benchmark.runner import BenchmarkManager  # noqa: E402
from app.core.backup import apply_pending_restore  # noqa: E402
from app.core.config import Settings, get_settings  # noqa: E402
from app.core.errors import AppError  # noqa: E402
from app.core.jobs import JobLock  # noqa: E402
from app.core.logging import configure_logging  # noqa: E402
from app.core.network_guard import guard  # noqa: E402
from app.db.session import init_db, make_engine, make_sessionmaker  # noqa: E402
from app.ml.intent_baseline import ARTIFACT_NAME, IntentBaseline  # noqa: E402
from app.ml.tracking import Tracker  # noqa: E402
from app.services.chat_service import ChatService  # noqa: E402
from app.services.document_service import DocumentService  # noqa: E402
from app.services.model_registry import ModelRegistry  # noqa: E402
from app.services.ollama_client import OllamaClient  # noqa: E402
from app.services.rag_service import RagService  # noqa: E402
from app.services.scheduler import ScheduleManager  # noqa: E402
from app.services.settings_service import SettingsService  # noqa: E402
from app.services.setup_service import SetupService  # noqa: E402

log = logging.getLogger("app")

CSP = ("default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; font-src 'self'; "
       "connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'")


def build_context(settings: Settings) -> AppContext:
    settings.ensure_dirs()
    restored = apply_pending_restore(settings.data_dir)  # a restore staged from the UI is applied before the DB opens
    if restored:
        log.warning("staged backup restore applied" if restored.get("ok") else "staged backup restore FAILED",
                    extra={k: v for k, v in restored.items() if k in ("restored_from", "previous_data_moved_to", "error")})
    engine = make_engine(settings.sqlalchemy_url)
    SessionLocal = make_sessionmaker(engine)
    migration = init_db(engine, SessionLocal, settings.backups_dir)

    guard.configure(allowed_hosts=settings.offline_allowed_hosts)
    settings_service = SettingsService(settings, SessionLocal)
    settings_service.get()  # loads persisted settings and applies offline mode to the guard
    if settings.network_guard_enabled:
        guard.install_audit_hook()

    local = OllamaClient(settings.ollama_base_url, connect_timeout_s=settings.ollama_connect_timeout_s, label="local")
    cloud = None
    if settings.ollama_cloud_enabled and settings.ollama_api_key:
        cloud = OllamaClient(settings.ollama_cloud_base_url, api_key=settings.ollama_api_key,
                             connect_timeout_s=settings.ollama_connect_timeout_s, label="cloud")
    registry = ModelRegistry(settings, settings_service, local, cloud)
    documents_svc = DocumentService(settings, SessionLocal, settings_service, registry)
    rag = RagService(documents_svc, settings_service)
    chat_svc = ChatService(SessionLocal, settings_service, registry, rag)
    suites = SuiteRepository(settings.datasets_dir, SessionLocal)
    tracker = Tracker(settings.mlflow_enabled, settings.mlflow_uri, settings.data_dir / "mlartifacts")
    job_lock = JobLock()  # benchmark, arena and agent runs never measure at the same time
    bench = BenchmarkManager(settings, SessionLocal, settings_service, registry, suites, tracker, job_lock)
    arena_store = ArenaStore(settings, SessionLocal)
    arena_mgr = ArenaManager(settings, SessionLocal, arena_store, registry, settings_service, tracker, job_lock)
    agent_suites = AgentSuiteRepository(settings.datasets_dir, SessionLocal)
    agent_mgr = AgentManager(settings, SessionLocal, agent_suites, registry, settings_service, tracker, job_lock,
                             rag=rag)
    return AppContext(settings=settings, engine=engine, SessionLocal=SessionLocal, settings_service=settings_service,
                      registry=registry, documents=documents_svc, rag=rag, chat=chat_svc, suites=suites,
                      tracker=tracker, benchmarks=bench, baseline=IntentBaseline(settings.ml_dir / ARTIFACT_NAME),
                      job_lock=job_lock, arena_store=arena_store, arena=arena_mgr, agent_suites=agent_suites,
                      agents=agent_mgr, started_at=datetime.now(), migration=migration,
                      setup=SetupService(settings, registry, settings_service, engine),
                      scheduler=ScheduleManager(settings, SessionLocal, bench, agent_mgr, engine),
                      work_agent=WorkAgentService(settings, SessionLocal, registry, settings_service, rag))


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json,
                      None if settings.app_env == "test" else settings.data_dir / "logs")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        ctx: AppContext = app.state.ctx
        ctx.benchmarks.start()
        ctx.arena.start()
        ctx.agents.start()
        if settings.scheduler_enabled:
            ctx.scheduler.start()
        if not settings.auth_enabled and settings.host not in ("127.0.0.1", "localhost", "::1"):
            log.warning("API is bound to a non-local address without authentication; set AUTH_ENABLED=true")
        log.info("Local AI Lab started", extra={"version": __version__, "offline_only": guard.offline_only})
        yield
        ctx.benchmarks.stop()
        ctx.arena.stop()
        ctx.agents.stop()
        ctx.scheduler.stop()
        ctx.documents.shutdown()
        ctx.registry.local.close()
        ctx.engine.dispose()

    app = FastAPI(title="Local AI Lab API", version=__version__, lifespan=lifespan,
                  description="Offline AI assistant, private document Q&A, model benchmarking and MLOps.")
    app.state.ctx = build_context(settings)

    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=False,
                       allow_methods=["*"], allow_headers=["*"])

    local_nets = [ipaddress.ip_network(n, strict=False) for n in settings.unauthenticated_networks]

    def is_local_client(host: str | None) -> bool:
        if host is None:
            return False
        if host == "testclient" and settings.app_env == "test":
            return True
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            return False
        return any(ip in n for n in local_nets)

    host_names = {h.lower() for h in settings.local_host_names} | ({"testserver"} if settings.app_env == "test" else set())

    def _hostname(value: str) -> str:
        value = (value or "").strip().lower()
        if value.startswith("["):
            return value[1:value.find("]")] if "]" in value else value
        return value.rsplit(":", 1)[0] if value.count(":") == 1 else value

    def browser_guard(request: Request) -> JSONResponse | None:
        """Without authentication: refuse DNS-rebinding (wrong Host) and cross-site state changes (CSRF)."""
        if settings.auth_enabled:
            return None
        if _hostname(request.headers.get("host", "")) not in host_names:
            return JSONResponse({"error": {"code": "host_not_allowed",
                                           "message": "Requests must address Local AI Lab as localhost / 127.0.0.1.",
                                           "hint": "Add the host name to LOCAL_HOST_NAMES, or enable authentication "
                                                   "to serve it under another name."}}, status_code=403)
        origin = request.headers.get("origin")
        if request.method not in ("GET", "HEAD", "OPTIONS") and origin and origin != "null":
            if origin not in settings.cors_origins and _hostname(origin.split("://", 1)[-1]) not in host_names:
                return JSONResponse({"error": {"code": "cross_site_request",
                                               "message": "Cross-site requests are not accepted."}}, status_code=403)
        return None

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        blocked = browser_guard(request) if request.url.path != "/api/health" else None
        if blocked is not None:
            log.warning("request refused by browser guard", extra={"path": request.url.path,
                                                                    "host": request.headers.get("host"),
                                                                    "origin": request.headers.get("origin")})
            return blocked
        host = request.client.host if request.client else None
        if not settings.auth_enabled and not is_local_client(host) and request.url.path != "/api/health":
            log.warning("remote request refused (authentication disabled)", extra={"client": host,
                                                                                   "path": request.url.path})
            return JSONResponse({"error": {"code": "remote_access_requires_auth",
                                           "message": "This Local AI Lab only accepts requests from this computer "
                                                      "because authentication is disabled.",
                                           "hint": "Set AUTH_ENABLED=true (and use HTTPS) to allow other "
                                                   "computers, or add their network to UNAUTHENTICATED_NETWORKS."}},
                                status_code=403)
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        if not request.url.path.startswith(("/api/", "/v1/")):  # the bundled UI never needs another origin
            response.headers["Content-Security-Policy"] = CSP
        if request.url.path.startswith(("/api/", "/v1/")):
            log.info("request", extra={"request_id": rid, "method": request.method, "path": request.url.path,
                                       "status": response.status_code,
                                       "duration_ms": round((time.perf_counter() - t0) * 1000, 1)})
        return response

    @app.exception_handler(AppError)
    async def app_error_handler(_: Request, exc: AppError):
        return JSONResponse(exc.to_dict(), status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_: Request, exc: RequestValidationError):
        details = [{"loc": list(e.get("loc", [])), "msg": e.get("msg")} for e in exc.errors()]
        return JSONResponse({"error": {"code": "invalid_request", "message": "Invalid request.", "details": details}},
                            status_code=422)

    for r in (health, auth, system, setup, backups, models, chat, documents, benchmarks, ml, arena, agents,
              settings_routes, bundles, automation, work_agent, openai_compat):
        app.include_router(r.router)

    dist = settings.frontend_dist
    if (dist / "index.html").exists():  # serve the built React app (single-process mode)
        @app.get("/{path:path}", include_in_schema=False)
        async def spa(path: str):
            if path.startswith("api/"):
                return JSONResponse({"error": {"code": "not_found", "message": "Not found."}}, status_code=404)
            candidate = (dist / path).resolve()
            if path and candidate.is_file() and dist.resolve() in candidate.parents:
                return FileResponse(candidate)
            return FileResponse(dist / "index.html")

    return app


def _make_default_app() -> FastAPI:
    return create_app()


app = _make_default_app() if os.environ.get("LOCAL_AI_LAB_NO_APP") != "1" else None  # type: ignore[assignment]
