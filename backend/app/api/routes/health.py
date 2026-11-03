from __future__ import annotations

import shutil
import time

from fastapi import APIRouter, Depends
from sqlalchemy import text

from app import __version__
from app.api.deps import AppContext, get_ctx
from app.core.errors import AppError
from app.core.network_guard import guard

router = APIRouter(prefix="/api/health", tags=["health"])


@router.get("")
def liveness(ctx: AppContext = Depends(get_ctx)) -> dict:
    return {"status": "ok", "app": ctx.settings.app_name, "version": __version__,
            "uptime_s": round((time.time() - ctx.started_at.timestamp()), 1)}


@router.get("/ready")
def readiness(ctx: AppContext = Depends(get_ctx)) -> dict:
    checks: dict[str, dict] = {}
    t0 = time.perf_counter()
    try:
        with ctx.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["database"] = {"ok": True, "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
    except Exception as exc:  # noqa: BLE001
        checks["database"] = {"ok": False, "error": type(exc).__name__}

    t0 = time.perf_counter()
    try:
        version = ctx.registry.local.version()
        checks["ollama"] = {"ok": True, "version": version, "base_url": ctx.registry.local.base_url,
                            "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
    except AppError as exc:
        checks["ollama"] = {"ok": False, "base_url": ctx.registry.local.base_url, "error": exc.message, "hint": exc.hint}

    emb = ctx.documents.embedder_status()
    checks["embeddings"] = {"ok": True, "backend": emb["backend"], "model": emb["model"], "note": emb["note"]}
    free_gb = round(shutil.disk_usage(ctx.settings.data_dir).free / 1024**3, 1)
    checks["storage"] = {"ok": free_gb > 1, "data_dir": str(ctx.settings.data_dir), "free_gb": free_gb}
    checks["mlflow"] = {"ok": True, **ctx.tracker.status()}
    checks["privacy"] = {"ok": True, "offline_only": guard.offline_only,
                         "socket_guard_installed": guard.audit_hook_installed}
    checks["benchmark_worker"] = {"ok": True, "busy": ctx.benchmarks.is_busy()}
    overall = "ok" if checks["database"]["ok"] and checks["ollama"]["ok"] else "degraded"
    return {"status": overall, "checks": checks}
