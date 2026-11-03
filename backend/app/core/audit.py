"""Append-only audit log of security-relevant actions.

Recorded: sign-in successes and failures, account creation, API-token creation and revocation, settings changes
(offline mode, cloud access…), backups and restores, model downloads/deletions, trusted-pickle uploads and
every Work-Agent tool approval or rejection. Stored in the ``audit_events`` table; administrators can read
it at ``GET /api/system/audit``. Request bodies, documents and prompts are never written to it.
"""

from __future__ import annotations

import logging

from fastapi import Request
from sqlalchemy import delete, func, select

from app.db.base import utcnow
from app.models.core import AuditEvent

log = logging.getLogger("app.audit")
MAX_EVENTS = 50_000


def client_ip(request: Request | None) -> str | None:
    if request is None or request.client is None:
        return None
    return request.client.host


def record(SessionLocal, *, user_id: int | None, username: str | None, action: str, target: str | None = None,
           details: dict | None = None, ip: str | None = None, outcome: str = "ok") -> None:
    try:
        with SessionLocal() as db:
            db.add(AuditEvent(user_id=user_id, username=(username or "")[:64] or None, action=action[:64],
                              target=(target or "")[:300] or None, details=details or None, ip=ip, outcome=outcome,
                              created_at=utcnow()))
            db.commit()
            count = db.scalar(select(func.count(AuditEvent.id))) or 0
            if count > MAX_EVENTS:
                cutoff = db.scalar(select(AuditEvent.id).order_by(AuditEvent.id.desc()).offset(MAX_EVENTS).limit(1))
                if cutoff:
                    db.execute(delete(AuditEvent).where(AuditEvent.id <= cutoff))
                    db.commit()
    except Exception:  # noqa: BLE001 - auditing must never break the request
        log.exception("audit write failed", extra={"action": action})
    log.info("audit", extra={"action": action, "target": target, "user": username, "outcome": outcome})


def audit(ctx, user, action: str, target: str | None = None, *, request: Request | None = None,
          details: dict | None = None, outcome: str = "ok") -> None:
    record(ctx.SessionLocal, user_id=getattr(user, "id", None), username=getattr(user, "username", None),
           action=action, target=target, details=details, ip=client_ip(request), outcome=outcome)
