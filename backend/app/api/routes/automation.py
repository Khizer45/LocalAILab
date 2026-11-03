"""Schedules (recurring evaluations and backups) and outgoing webhooks."""

from __future__ import annotations

import re

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import AppContext, current_user, get_ctx, get_db, require_admin
from app.core.audit import audit
from app.core.errors import ConflictError, ForbiddenError, NotFoundError, ValidationFailed
from app.db.base import utcnow
from app.models.automation import Schedule, Webhook
from app.models.core import User
from app.services import webhooks as W
from app.services.scheduler import next_time

router = APIRouter(tags=["automation"])
HHMM = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


class ScheduleIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    kind: str = Field(pattern="^(benchmark|agent|backup)$")
    request: dict = Field(default_factory=dict)
    every_hours: int | None = Field(None, ge=1, le=24 * 30)
    daily_at: str | None = None
    gates: dict | None = None
    notify_webhook: str | None = Field(None, max_length=100)
    enabled: bool = True

    @field_validator("daily_at")
    @classmethod
    def _hhmm(cls, v):
        if v is not None and not HHMM.match(v):
            raise ValueError("daily_at must be HH:MM (24-hour)")
        return v


class WebhookIn(BaseModel):
    name: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_.-]+$")
    url: str = Field(min_length=8, max_length=500)
    description: str = Field("", max_length=500)
    allow_agent: bool = False


def _schedule_out(s: Schedule) -> dict:
    iso = lambda d: d.isoformat() + "Z" if d else None  # noqa: E731
    return {"id": s.id, "name": s.name, "kind": s.kind, "request": s.request, "every_hours": s.every_hours,
            "daily_at": s.daily_at, "gates": s.gates, "notify_webhook": s.notify_webhook, "enabled": s.enabled,
            "next_run_at": iso(s.next_run_at), "last_run_at": iso(s.last_run_at), "last_run_id": s.last_run_id,
            "last_status": s.last_status, "last_message": s.last_message}


def _validate(body: ScheduleIn, ctx: AppContext, user: User) -> dict:
    if not (body.every_hours or body.daily_at) or (body.every_hours and body.daily_at):
        raise ValidationFailed("Give either every_hours or daily_at.")
    req = dict(body.request)
    if body.kind == "backup":
        if not user.is_admin:
            raise ForbiddenError("Only administrators can schedule backups.")
        return {"keep": int(req.get("keep", 7)), "include_mlflow": bool(req.get("include_mlflow", True))}
    req.setdefault("name", f"[scheduled] {body.name}"[:200])
    if body.kind == "benchmark":
        from app.benchmark.runner import BenchmarkRequest

        parsed = BenchmarkRequest(**req)
        ctx.suites.get(parsed.suite_id, user.id)
        return parsed.model_dump(exclude_none=True)
    from app.agents.runner import AgentRunRequest

    parsed = AgentRunRequest(**req)
    ctx.agent_suites.get(user.id, parsed.suite_id)
    return parsed.model_dump(exclude_none=True)


@router.get("/api/schedules")
def list_schedules(db: Session = Depends(get_db), user: User = Depends(current_user)) -> dict:
    rows = db.scalars(select(Schedule).where(Schedule.owner_id == user.id).order_by(Schedule.id)).all()
    return {"schedules": [_schedule_out(s) for s in rows]}


@router.post("/api/schedules", status_code=201)
def create_schedule(body: ScheduleIn, request: Request, ctx: AppContext = Depends(get_ctx),
                    db: Session = Depends(get_db), user: User = Depends(current_user)) -> dict:
    req = _validate(body, ctx, user)
    s = Schedule(owner_id=user.id, name=body.name, kind=body.kind, request=req, every_hours=body.every_hours,
                 daily_at=body.daily_at, gates=body.gates, notify_webhook=body.notify_webhook, enabled=body.enabled)
    s.next_run_at = next_time(s, utcnow()) if body.enabled else None
    db.add(s)
    db.commit()
    audit(ctx, user, "schedule.create", body.name, request=request, details={"kind": body.kind})
    return _schedule_out(s)


def _own(db: Session, sid: int, user: User) -> Schedule:
    s = db.get(Schedule, sid)
    if s is None or s.owner_id != user.id:
        raise NotFoundError("Schedule not found.")
    return s


@router.put("/api/schedules/{sid}")
def update_schedule(sid: int, body: ScheduleIn, ctx: AppContext = Depends(get_ctx), db: Session = Depends(get_db),
                    user: User = Depends(current_user)) -> dict:
    s = _own(db, sid, user)
    s.request = _validate(body, ctx, user)
    s.name, s.kind, s.every_hours, s.daily_at = body.name, body.kind, body.every_hours, body.daily_at
    s.gates, s.notify_webhook, s.enabled = body.gates, body.notify_webhook, body.enabled
    s.next_run_at = next_time(s, utcnow()) if body.enabled else None
    db.commit()
    return _schedule_out(s)


@router.delete("/api/schedules/{sid}", status_code=204)
def delete_schedule(sid: int, db: Session = Depends(get_db), user: User = Depends(current_user)) -> None:
    db.delete(_own(db, sid, user))
    db.commit()


@router.post("/api/schedules/{sid}/run-now")
def run_schedule_now(sid: int, ctx: AppContext = Depends(get_ctx), db: Session = Depends(get_db),
                     user: User = Depends(current_user)) -> dict:
    _own(db, sid, user)
    return ctx.scheduler.run_now(sid)


# ------------------------------------------------------------------ webhooks (admin)
def _hook_out(h: Webhook, with_secret: bool = False) -> dict:
    out = {"id": h.id, "name": h.name, "url": h.url, "description": h.description, "allow_agent": h.allow_agent}
    if with_secret:
        out["secret"] = h.secret
    return out


@router.get("/api/system/webhooks")
def list_webhooks(db: Session = Depends(get_db), _: User = Depends(require_admin)) -> dict:
    return {"webhooks": [_hook_out(h) for h in db.scalars(select(Webhook).order_by(Webhook.name))]}


@router.post("/api/system/webhooks", status_code=201)
def create_webhook(body: WebhookIn, request: Request, ctx: AppContext = Depends(get_ctx), db: Session = Depends(get_db),
                   user: User = Depends(require_admin)) -> dict:
    if db.scalar(select(Webhook).where(Webhook.name == body.name)):
        raise ConflictError("A webhook with this name exists.")
    h = Webhook(name=body.name, url=W.validate_url(body.url), description=body.description,
                allow_agent=body.allow_agent, secret=W.new_secret())
    db.add(h)
    db.commit()
    audit(ctx, user, "webhook.create", body.name, request=request, details={"url": body.url,
                                                                            "allow_agent": body.allow_agent})
    return _hook_out(h, with_secret=True) | {"note": "Copy the signing secret now; it is used to verify requests."}


@router.delete("/api/system/webhooks/{name}", status_code=204)
def delete_webhook(name: str, request: Request, ctx: AppContext = Depends(get_ctx), db: Session = Depends(get_db),
                   user: User = Depends(require_admin)) -> None:
    h = db.scalar(select(Webhook).where(Webhook.name == name))
    if h is None:
        raise NotFoundError("Webhook not found.")
    db.delete(h)
    db.commit()
    audit(ctx, user, "webhook.delete", name, request=request)


@router.post("/api/system/webhooks/{name}/test")
def test_webhook(name: str, request: Request, ctx: AppContext = Depends(get_ctx), user: User = Depends(require_admin)) -> dict:
    res = W.send(W.get(ctx.SessionLocal, name), "test", {"message": "Test from Local AI Lab"})
    audit(ctx, user, "webhook.test", name, request=request, details={"status": res["status"]})
    return res
