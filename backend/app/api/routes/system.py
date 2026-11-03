from __future__ import annotations

import os

from fastapi import APIRouter, Depends

from app.api.deps import AppContext, current_user, get_ctx, require_admin
from app.core.network_guard import guard
from app.models.core import User
from app.services.system_monitor import recommendations, snapshot

router = APIRouter(prefix="/api", tags=["system"], dependencies=[Depends(current_user)])

TELEMETRY_ENV = ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "MLFLOW_DISABLE_TELEMETRY",
                 "DO_NOT_TRACK", "ANONYMIZED_TELEMETRY", "ORT_DISABLE_TELEMETRY", "OTEL_SDK_DISABLED"]


@router.get("/system/audit")
def audit_log(limit: int = 200, action: str | None = None, ctx: AppContext = Depends(get_ctx),
              _: User = Depends(require_admin)) -> dict:
    """Security-relevant actions (sign-ins, tokens, settings, backups, model downloads, agent approvals)."""
    from sqlalchemy import select

    from app.models.core import AuditEvent

    with ctx.SessionLocal() as db:
        q = select(AuditEvent).order_by(AuditEvent.id.desc()).limit(max(1, min(limit, 2000)))
        if action:
            q = q.where(AuditEvent.action.like(f"{action}%"))
        rows = db.scalars(q).all()
        return {"events": [{"id": e.id, "at": e.created_at.isoformat() + "Z", "user": e.username, "action": e.action,
                            "target": e.target, "outcome": e.outcome, "ip": e.ip, "details": e.details}
                           for e in rows]}


@router.get("/system/resources")
def resources(ctx: AppContext = Depends(get_ctx)) -> dict:
    snap = snapshot(ctx.settings.data_dir)
    snap["loaded_models"] = [
        {"name": m.get("name"), "size_mb": round((m.get("size") or 0) / 1024**2, 1),
         "size_vram_mb": round((m.get("size_vram") or 0) / 1024**2, 1), "expires_at": m.get("expires_at"),
         "context_length": m.get("context_length")}
        for m in ctx.registry.loaded_models()
    ]
    return snap


@router.get("/system/recommendations")
def model_recommendations(ctx: AppContext = Depends(get_ctx)) -> dict:
    return recommendations(snapshot(ctx.settings.data_dir))


@router.get("/privacy/status")
def privacy_status(ctx: AppContext = Depends(get_ctx)) -> dict:
    rs = ctx.settings_service.get()
    _, status = ctx.registry.list_models()
    local_url = ctx.registry.local.base_url
    from urllib.parse import urlsplit

    local_host = urlsplit(local_url).hostname or ""
    checks = [
        {"id": "offline_mode", "label": "Offline-Only mode", "ok": rs.offline_only,
         "detail": "Cloud models and non-local network requests are blocked." if rs.offline_only
         else "Disabled – cloud models and remote hosts can be used."},
        {"id": "socket_guard", "label": "Socket-level network guard", "ok": guard.audit_hook_installed,
         "detail": "Every outbound TCP connection from this backend is checked against the allow-list."
         if guard.audit_hook_installed else "Disabled (NETWORK_GUARD_ENABLED=false)."},
        {"id": "runtime_local", "label": "Model runtime address", "ok": guard.is_host_allowed(local_host) if rs.offline_only else True,
         "detail": f"Ollama at {local_url}"},
        {"id": "embeddings_local", "label": "Embeddings computed locally", "ok": True,
         "detail": f"{ctx.documents.embedder_status()['backend']} embedder – never sent to a cloud provider."},
        {"id": "cloud_configured", "label": "Ollama Cloud access", "ok": not status["cloud"].get("enabled") or rs.offline_only,
         "detail": "Not configured." if not status["cloud"].get("enabled")
         else ("Configured but blocked by Offline-Only mode." if rs.offline_only else "Configured and ENABLED.")},
        {"id": "docs_with_cloud", "label": "Documents never sent to cloud models", "ok": not rs.allow_documents_with_cloud,
         "detail": "Blocked." if not rs.allow_documents_with_cloud else "Allowed by the user in Settings."},
        {"id": "telemetry", "label": "Library telemetry opt-outs set",
         "ok": all(os.environ.get(k) for k in TELEMETRY_ENV),
         "detail": ", ".join(f"{k}={os.environ.get(k, 'unset')}" for k in TELEMETRY_ENV)},
        {"id": "auth", "label": "Authentication", "ok": ctx.settings.auth_enabled or ctx.settings.host in ("127.0.0.1", "localhost"),
         "detail": "Enabled." if ctx.settings.auth_enabled else f"Disabled – API bound to {ctx.settings.host} (local use only)."},
    ]
    return {"checks": checks, "guard": guard.status(),
            "limitations": [
                "Offline-Only is an application-level restriction, not an operating-system firewall. The guard covers "
                "Python code in this backend process. Native libraries (e.g. ONNX Runtime), the Ollama server, the "
                "browser and the OS can open their own connections – known telemetry is switched off with the "
                "variables above, and Ollama's cloud features with OLLAMA_NO_CLOUD=1. For a hard guarantee, block "
                "outbound traffic with a firewall rule or run on a machine without a network.",
                "Prompt-injection defences reduce, but cannot eliminate, the risk that document text influences answers.",
                "Data at rest (SQLite database, uploaded files) is not encrypted by this application; use disk encryption.",
            ]}
