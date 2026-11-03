from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import AppContext, current_user, get_ctx

router = APIRouter(prefix="/api/models", tags=["models"], dependencies=[Depends(current_user)])


@router.get("")
def list_models(refresh: bool = False, ctx: AppContext = Depends(get_ctx)) -> dict:
    models, status = ctx.registry.list_models(force=refresh)
    rs = ctx.settings_service.get()
    return {"models": models, "status": status, "default_model": rs.default_model,
            "embedding_model": rs.embedding_model, "offline_only": rs.offline_only}


@router.get("/loaded")
def loaded(ctx: AppContext = Depends(get_ctx)) -> dict:
    return {"models": ctx.registry.loaded_models()}
