from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.api.deps import AppContext, current_user, get_ctx, require_admin
from app.core.audit import audit
from app.models.core import User
from app.services.settings_service import RuntimeSettingsUpdate

router = APIRouter(prefix="/api/settings", tags=["settings"])


def _out(ctx: AppContext) -> dict:
    rs = ctx.settings_service.get()
    s = ctx.settings
    return {
        "runtime": rs.model_dump(),
        "static": {  # from environment variables / .env – restart required to change
            "ollama_base_url": s.ollama_base_url,
            "ollama_cloud_enabled": s.ollama_cloud_enabled,
            "ollama_cloud_api_key_set": bool(s.ollama_api_key),
            "offline_allowed_hosts": s.offline_allowed_hosts,
            "network_guard_enabled": s.network_guard_enabled,
            "auth_enabled": s.auth_enabled,
            "data_dir": str(s.data_dir),
            "chunk_size": s.chunk_size,
            "chunk_overlap": s.chunk_overlap,
            "ml_baseline_enabled": s.ml_baseline_enabled,
            "mlflow_enabled": s.mlflow_enabled,
            "doc_processing_mode": s.doc_processing_mode,
        },
    }


@router.get("")
def get_settings(ctx: AppContext = Depends(get_ctx), _: User = Depends(current_user)) -> dict:
    return _out(ctx)


@router.patch("")
def update_settings(body: RuntimeSettingsUpdate, request: Request, ctx: AppContext = Depends(get_ctx),
                    user: User = Depends(require_admin)) -> dict:
    changes = body.model_dump(exclude_unset=True)
    before = ctx.settings_service.get().model_dump()
    ctx.settings_service.update(body)
    diff = {k: {"from": before.get(k), "to": v} for k, v in changes.items() if before.get(k) != v}
    if diff:
        audit(ctx, user, "settings.update", ", ".join(sorted(diff))[:300], request=request, details=diff)
    ctx.registry.invalidate()
    ctx.documents.invalidate()
    return _out(ctx)
