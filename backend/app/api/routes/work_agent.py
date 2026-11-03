"""Work Agent API: real tools with permissions, human approval and an audit trail (see app/agents/work.py)."""

from __future__ import annotations

import json
import threading

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agents.work import DEFAULT_POLICY, tool_specs
from app.api.deps import AppContext, current_user, get_ctx, require_admin
from app.core.audit import audit
from app.core.errors import AppError
from app.models.core import User

router = APIRouter(prefix="/api/work-agent", tags=["work-agent"])


class WorkRun(BaseModel):
    model: str = Field(min_length=1, max_length=200)
    prompt: str = Field(min_length=1, max_length=20000)
    tools: list[str] | None = None
    max_steps: int = Field(6, ge=1, le=12)
    interactive: bool = True


class Decision(BaseModel):
    approve: bool
    note: str | None = Field(None, max_length=300)


def _enabled(ctx: AppContext) -> None:
    if not ctx.settings.work_agent_enabled:
        raise AppError("The Work Agent is disabled (WORK_AGENT_ENABLED=false).", code="work_agent_disabled",
                       status_code=404)


@router.get("/config")
def config(ctx: AppContext = Depends(get_ctx), _: User = Depends(current_user)) -> dict:
    _enabled(ctx)
    svc = ctx.work_agent
    policy = svc.policy()
    return {"workspace": str(ctx.settings.workspace_dir), "webhooks": svc.agent_webhooks(),
            "approval_timeout_s": svc.approval_timeout_s,
            "tools": [{"name": t.name, "description": t.description, "policy": policy.get(t.name, "ask"),
                       "writes": t.name in ("write_file", "call_webhook", "draft_email")}
                      for t in tool_specs(svc.agent_webhooks())]}


@router.put("/policy")
def set_policy(body: dict[str, str], request: Request, ctx: AppContext = Depends(get_ctx),
               user: User = Depends(require_admin)) -> dict:
    _enabled(ctx)
    merged = ctx.work_agent.set_policy(body)
    audit(ctx, user, "work_agent.policy", ", ".join(f"{k}={v}" for k, v in body.items())[:300], request=request)
    return {"policy": merged, "defaults": DEFAULT_POLICY}


@router.post("/run")
def run(body: WorkRun, ctx: AppContext = Depends(get_ctx), user: User = Depends(current_user)):
    """Interactive runs stream NDJSON events (including ``approval_required``); non-interactive runs return JSON
    and refuse every tool that needs approval."""
    _enabled(ctx)
    stop = threading.Event()
    events = ctx.work_agent.run(user, body.model, body.prompt, tools=body.tools, max_steps=body.max_steps,
                                interactive=body.interactive, cancel=stop)
    if not body.interactive:
        trace = list(events)
        last = trace[-1] if trace else {}
        return {"answer": last.get("answer", ""), "stop_reason": last.get("stop_reason"), "trace": trace}

    def gen():
        try:
            for ev in events:
                yield json.dumps(ev, default=str) + "\n"
        finally:
            stop.set()  # the browser went away: pending approvals are cancelled

    return StreamingResponse(gen(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/approvals")
def pending(ctx: AppContext = Depends(get_ctx), user: User = Depends(current_user)) -> dict:
    return {"approvals": ctx.work_agent.approvals.pending(user.id)}


@router.post("/approvals/{approval_id}")
def decide(approval_id: str, body: Decision, request: Request, ctx: AppContext = Depends(get_ctx),
           user: User = Depends(current_user)) -> dict:
    out = ctx.work_agent.approvals.decide(user.id, approval_id, body.approve, body.note)
    audit(ctx, user, "work_agent.approve" if body.approve else "work_agent.reject", approval_id, request=request,
          details={"note": body.note})
    return out


@router.get("/files")
def files(folder: str = ".", ctx: AppContext = Depends(get_ctx), _: User = Depends(current_user)) -> dict:
    _enabled(ctx)
    return ctx.work_agent.workspace.list(folder)


@router.get("/files/content")
def file_content(path: str, ctx: AppContext = Depends(get_ctx), _: User = Depends(current_user)) -> dict:
    _enabled(ctx)
    return ctx.work_agent.workspace.read(path)
