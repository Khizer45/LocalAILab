"""Public demo entry point (Hugging Face Spaces).

Wraps the normal Local AI Lab app for an anonymous, public audience:

* models are the built-in Ollama simulator, so no real model and no API key exists in the container;
* the shared "local" user is NOT an administrator, so every admin-only action (settings, backups, restores,
  webhooks, model downloads, user management, pickle/joblib models) is refused by the app's own checks;
* a short deny-list closes the remaining routes that make no sense in a public demo (schedules, bundle import,
  the OpenAI-compatible API, sign-in, backbone downloads, the Work Agent policy);
* state-changing requests are rate-limited per visitor IP;
* the page may be embedded only by huggingface.co.

Run:  uvicorn public_demo:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import ipaddress
import json
import os
import threading
import time
from collections import defaultdict, deque

# --- configuration that must be in place before the app reads its settings --------------------------------------
_space_host = (os.environ.get("SPACE_HOST") or "").strip().lower()
_hosts = ["localhost", "127.0.0.1", "::1"] + ([_space_host] if _space_host else [])
_extra = [h.strip().lower() for h in os.environ.get("DEMO_EXTRA_HOSTS", "").split(",") if h.strip()]
os.environ["LOCAL_HOST_NAMES"] = ",".join(dict.fromkeys(_hosts + _extra))
# Visitors arrive through the Spaces proxy from anywhere; access control is done below instead of by network.
os.environ["UNAUTHENTICATED_NETWORKS"] = "0.0.0.0/0,::/0"
os.environ["AUTH_ENABLED"] = "false"

from app.main import create_app  # noqa: E402
from app.models.core import User  # noqa: E402
from sqlalchemy import select  # noqa: E402

GITHUB_URL = os.environ.get("DEMO_GITHUB_URL", "https://github.com/Khizer45/LocalAILab")
RATE_PER_MIN = int(os.environ.get("DEMO_WRITES_PER_MINUTE", "30"))

# (method or "*", path prefix) – refused in the public demo
DENY: list[tuple[str, str]] = [
    ("*", "/v1/"),                              # OpenAI-compatible API: not a free public endpoint
    ("POST", "/api/auth/"), ("DELETE", "/api/auth/"),
    ("*", "/api/system/backups"),               # admin-only anyway; never expose other visitors' data
    ("*", "/api/system/audit"),
    ("POST", "/api/system/"), ("PUT", "/api/system/"), ("DELETE", "/api/system/"), ("PATCH", "/api/system/"),
    ("PATCH", "/api/settings"),
    ("POST", "/api/setup/"), ("DELETE", "/api/setup/"),
    ("POST", "/api/schedules"), ("PUT", "/api/schedules"), ("DELETE", "/api/schedules"),
    ("POST", "/api/bundles/import"), ("POST", "/api/bundles/rerun"),
    ("POST", "/api/arena/backbones/"),
    ("PUT", "/api/work-agent/policy"),
]


def _denied(method: str, path: str) -> bool:
    return any((m == "*" or m == method) and path.startswith(p) for m, p in DENY)


class _RateLimiter:
    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self.hits: dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self.lock:
            q = self.hits[key]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_minute:
                return False
            q.append(now)
            if len(self.hits) > 10_000:  # forget idle visitors
                for k in [k for k, v in self.hits.items() if not v or now - v[-1] > 60]:
                    self.hits.pop(k, None)
            return True


def _visitor_ip(scope) -> str:
    """The visitor's address: the right-most public address in X-Forwarded-For (added by the Spaces proxy;
    anything a client puts there itself ends up further left), else the socket peer."""
    for k, v in scope.get("headers", []):
        if k == b"x-forwarded-for":
            for part in reversed(v.decode("latin-1").split(",")):
                part = part.strip()
                try:
                    ip = ipaddress.ip_address(part)
                except ValueError:
                    continue
                if ip.is_global:
                    return part
    return (scope.get("client") or ("unknown", 0))[0]


def _demote_local_user(inner) -> None:
    ctx = inner.state.ctx
    with ctx.SessionLocal() as db:
        user = db.scalar(select(User).where(User.username == "local"))
        if user is not None and user.is_admin:
            user.is_admin = False
            db.commit()


async def _send_json(send, status: int, code: str, message: str) -> None:
    body = json.dumps({"error": {"code": code, "message": message,
                                 "hint": f"Install Local AI Lab on your own computer for the full version: {GITHUB_URL}"}}
                      ).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()),
                            (b"cache-control", b"no-store")]})
    await send({"type": "http.response.body", "body": body})


class PublicDemo:
    """ASGI wrapper: deny-list, per-IP write limit, frame policy for huggingface.co."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.limiter = _RateLimiter(RATE_PER_MIN)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.inner(scope, receive, send)
        method, path = scope["method"].upper(), scope["path"]
        if _denied(method, path):
            return await _send_json(send, 403, "disabled_in_public_demo",
                                    "This action is disabled in the public online demo.")
        if method not in ("GET", "HEAD", "OPTIONS"):
            if not self.limiter.allow(_visitor_ip(scope)):
                return await _send_json(send, 429, "rate_limited",
                                        "Too many actions in a short time. Please wait a minute.")

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = []
                for k, v in message.get("headers", []):
                    if k.lower() == b"content-security-policy":
                        v = v.replace(b"frame-ancestors 'none'", b"frame-ancestors https://huggingface.co")
                    if k.lower() == b"x-frame-options":
                        continue
                    headers.append((k, v))
                headers.append((b"strict-transport-security", b"max-age=31536000"))
                headers.append((b"permissions-policy", b"camera=(), microphone=(), geolocation=()"))
                message = {**message, "headers": headers}
            await send(message)

        return await self.inner(scope, receive, send_wrapper)


_inner = create_app()
_demote_local_user(_inner)
app = PublicDemo(_inner)
