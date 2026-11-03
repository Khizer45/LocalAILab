"""First-run setup: prerequisite checks, model catalogue with memory fit, model download / removal."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.api.deps import AppContext, current_user, get_ctx, require_admin
from app.core.audit import audit
from app.core.errors import ValidationFailed
from app.models.core import User
from app.services.setup_service import MODEL_NAME

router = APIRouter(prefix="/api/setup", tags=["setup"])


class PullRequest(BaseModel):
    name: str = Field(min_length=1, max_length=160)


@router.get("/check")
def check(ctx: AppContext = Depends(get_ctx), _: User = Depends(current_user)) -> dict:
    return ctx.setup.check()


@router.get("/models")
def catalog(num_ctx: int = 4096, ctx: AppContext = Depends(get_ctx), _: User = Depends(current_user)) -> dict:
    return ctx.setup.catalog(max(512, min(num_ctx, 131072)))


@router.post("/models/pull")
def pull(body: PullRequest, request: Request, ctx: AppContext = Depends(get_ctx), user: User = Depends(require_admin)):
    """Stream Ollama's download progress as NDJSON. Ollama resumes a partly downloaded model on the next pull."""
    if not MODEL_NAME.match(body.name):
        raise ValidationFailed("Invalid model name.")
    ctx.setup._require_online()  # fail fast with a normal JSON error
    audit(ctx, user, "model.pull", body.name, request=request)

    def gen():
        for ev in ctx.setup.pull(body.name):
            yield json.dumps(ev) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson", headers={"Cache-Control": "no-cache"})


@router.delete("/models/{name:path}", status_code=204)
def delete(name: str, request: Request, ctx: AppContext = Depends(get_ctx), user: User = Depends(require_admin)) -> None:
    ctx.setup.delete(name)
    audit(ctx, user, "model.delete", name, request=request)
