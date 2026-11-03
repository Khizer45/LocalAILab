"""Outgoing webhooks (schedules, Work Agent) – signed, named, and subject to Offline-Only mode.

Each request carries ``X-LocalAILab-Signature: sha256=<hex>`` = HMAC-SHA256(secret, raw body) and
``X-LocalAILab-Event``. In n8n, verify it in a Code node (see docs/AUTOMATION.md). Requests go through the
offline network guard: with Offline-Only on, only loopback / allowed hosts can be called (e.g. an n8n instance
on the same computer).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from urllib.parse import urlsplit

from sqlalchemy import select

from app.core.errors import AppError, NotFoundError, OfflineModeViolation, ValidationFailed
from app.core.network_guard import OfflineNetworkBlocked, guarded_http_client
from app.models.automation import Webhook


def new_secret() -> str:
    return secrets.token_urlsafe(32)


def validate_url(url: str) -> str:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValidationFailed("Webhook URLs must start with http:// or https://.")
    if parts.username or parts.password:
        raise ValidationFailed("Put credentials in the receiving workflow, not in the URL.")
    return url


def sign(secret: str, body: bytes) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def get(SessionLocal, name: str) -> Webhook:
    with SessionLocal() as db:
        row = db.scalar(select(Webhook).where(Webhook.name == name))
        if row is None:
            raise NotFoundError(f"Webhook '{name}' is not configured.")
        db.expunge(row)
        return row


def send(hook: Webhook, event: str, payload: dict, timeout: float = 20.0) -> dict:
    body = json.dumps({"event": event, "payload": payload}, default=str).encode()
    headers = {"Content-Type": "application/json", "X-LocalAILab-Event": event,
               "X-LocalAILab-Signature": sign(hook.secret, body), "User-Agent": "LocalAILab-Webhook/1"}
    parts = urlsplit(hook.url)
    base = f"{parts.scheme}://{parts.netloc}"
    path = parts.path or "/"
    if parts.query:
        path += "?" + parts.query
    try:
        with guarded_http_client(base, timeout=timeout) as http:
            r = http.post(path, content=body, headers=headers)
    except AppError:
        raise
    except OfflineNetworkBlocked as exc:
        raise OfflineModeViolation(f"Webhook '{hook.name}' is not reachable in Offline-Only mode: {exc}",
                                   hint="Point it at a service on this computer (e.g. n8n on localhost) or switch "
                                        "Offline-Only off.") from exc
    except Exception as exc:  # noqa: BLE001 - network errors
        raise AppError(f"Webhook '{hook.name}' failed: {type(exc).__name__}", code="webhook_failed",
                       status_code=502) from exc
    text = r.text[:2000]
    try:
        data = r.json()
    except ValueError:
        data = None
    return {"status": r.status_code, "ok": r.status_code < 400, "response": data if data is not None else text}
