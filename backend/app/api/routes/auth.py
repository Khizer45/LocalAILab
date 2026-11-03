"""Authentication (AUTH_ENABLED=true), sessions and API tokens.

Needed when the API is reachable from other machines or shared by several people. Each user only sees their own
conversations, documents and runs. Session and API tokens are random, stored hashed, expire, and can be revoked.
API tokens carry *scopes* (read, chat, documents, evaluate, agent, admin) so a CI job or an n8n workflow gets only
the access it needs. Sign-in attempts are rate-limited per account and per client address.
"""

from __future__ import annotations

from datetime import timedelta

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import SCOPES, AppContext, current_user, get_ctx, get_db, require_admin
from app.core.audit import audit, client_ip, record
from app.core.errors import AppError, AuthError, ConflictError, ForbiddenError, NotFoundError, ValidationFailed
from app.core.ratelimit import LoginLimiter
from app.core.security import hash_password, hash_token, new_token, verify_password
from app.db.base import utcnow
from app.models.core import ApiToken, User
from app.schemas import Credentials

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _limiter(ctx: AppContext) -> LoginLimiter:
    lim = getattr(ctx, "_login_limiter", None)
    if lim is None:
        lim = LoginLimiter()
        ctx._login_limiter = lim  # type: ignore[attr-defined]
    return lim


class TokenRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    scopes: list[str] = Field(min_length=1)
    expires_days: int = Field(90, ge=1, le=366)


class PasswordChange(BaseModel):
    current_password: str = Field(min_length=1, max_length=256)
    new_password: str = Field(min_length=8, max_length=256)


def _token_out(t: ApiToken) -> dict:
    iso = lambda d: d.isoformat() + "Z" if d else None  # noqa: E731
    return {"id": t.id, "name": t.name, "prefix": t.prefix, "scopes": t.scopes, "kind": t.kind,
            "created_at": iso(t.created_at), "expires_at": iso(t.expires_at), "last_used_at": iso(t.last_used_at),
            "revoked_at": iso(t.revoked_at), "active": t.revoked_at is None and t.expires_at > utcnow()}


@router.get("/config")
def auth_config(ctx: AppContext = Depends(get_ctx), db: Session = Depends(get_db)) -> dict:
    users = db.scalar(select(func.count(User.id)).where(User.password_hash.is_not(None))) or 0
    return {"auth_enabled": ctx.settings.auth_enabled, "registration_open": ctx.settings.allow_registration or users == 0,
            "has_users": users > 0, "scopes": SCOPES}


@router.post("/register", status_code=201)
def register(body: Credentials, request: Request, ctx: AppContext = Depends(get_ctx),
             db: Session = Depends(get_db)) -> dict:
    if not ctx.settings.auth_enabled:
        raise ForbiddenError("Authentication is disabled on this server.")
    users = db.scalar(select(func.count(User.id)).where(User.password_hash.is_not(None))) or 0
    if users > 0 and not ctx.settings.allow_registration:
        raise ForbiddenError("Registration is closed. Ask the administrator to create an account.")
    if body.username.lower() == "local" or db.scalar(select(User).where(User.username == body.username)):
        raise ConflictError("Username is already taken.")
    user = User(username=body.username, password_hash=hash_password(body.password), is_admin=users == 0)
    db.add(user)
    db.commit()
    audit(ctx, user, "auth.register", user.username, request=request, details={"admin": user.is_admin})
    return {"id": user.id, "username": user.username, "is_admin": user.is_admin}


@router.post("/login")
def login(body: Credentials, request: Request, ctx: AppContext = Depends(get_ctx), db: Session = Depends(get_db)) -> dict:
    if not ctx.settings.auth_enabled:
        raise ForbiddenError("Authentication is disabled on this server.")
    ip = client_ip(request)
    limiter = _limiter(ctx)
    wait = limiter.retry_after(body.username, ip)
    if wait:
        record(ctx.SessionLocal, user_id=None, username=body.username, action="auth.login", ip=ip, outcome="denied",
               details={"reason": "rate limited"})
        raise AppError(f"Too many failed sign-in attempts. Try again in {wait // 60 + 1} minute(s).",
                       code="too_many_attempts", status_code=429)
    user = db.scalar(select(User).where(User.username == body.username))
    if user is None or not user.password_hash or not verify_password(body.password, user.password_hash):
        limiter.failed(body.username, ip)
        record(ctx.SessionLocal, user_id=user.id if user else None, username=body.username, action="auth.login",
               ip=ip, outcome="failed")
        raise AuthError("Invalid username or password.")
    limiter.succeeded(body.username)
    token = new_token()
    db.add(ApiToken(user_id=user.id, token_hash=hash_token(token), kind="session",
                    expires_at=utcnow() + timedelta(hours=ctx.settings.token_ttl_hours)))
    db.commit()
    audit(ctx, user, "auth.login", user.username, request=request)
    return {"token": token, "user": {"id": user.id, "username": user.username, "is_admin": user.is_admin},
            "expires_in_hours": ctx.settings.token_ttl_hours}


@router.post("/logout", status_code=204)
def logout(request: Request, db: Session = Depends(get_db), _: User = Depends(current_user)) -> None:
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        db.query(ApiToken).filter(ApiToken.token_hash == hash_token(header.split(" ", 1)[1].strip()),
                                  ApiToken.kind == "session").delete()
        db.commit()


@router.get("/me")
def me(request: Request, user: User = Depends(current_user), ctx: AppContext = Depends(get_ctx)) -> dict:
    return {"id": user.id, "username": user.username, "is_admin": user.is_admin,
            "auth_enabled": ctx.settings.auth_enabled, "token": getattr(request.state, "token", None)}


@router.post("/password", status_code=204)
def change_password(body: PasswordChange, request: Request, ctx: AppContext = Depends(get_ctx),
                    db: Session = Depends(get_db), user: User = Depends(current_user)) -> None:
    if not ctx.settings.auth_enabled or not user.password_hash:
        raise ForbiddenError("Passwords are only used when authentication is enabled.")
    if not verify_password(body.current_password, user.password_hash):
        audit(ctx, user, "auth.password_change", user.username, request=request, outcome="failed")
        raise AuthError("The current password is wrong.")
    db_user = db.get(User, user.id)
    db_user.password_hash = hash_password(body.new_password)
    current = (getattr(request.state, "token", None) or {}).get("id")
    for t in db.scalars(select(ApiToken).where(ApiToken.user_id == user.id, ApiToken.revoked_at.is_(None))):
        if t.id != current:
            t.revoked_at = utcnow()
    db.commit()
    audit(ctx, user, "auth.password_change", user.username, request=request, details={"other_tokens_revoked": True})


# ------------------------------------------------------------------ API tokens
@router.get("/tokens")
def list_tokens(user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    rows = db.scalars(select(ApiToken).where(ApiToken.user_id == user.id, ApiToken.kind == "api")
                      .order_by(ApiToken.created_at.desc())).all()
    return {"tokens": [_token_out(t) for t in rows], "scopes": SCOPES}


@router.post("/tokens", status_code=201)
def create_token(body: TokenRequest, request: Request, ctx: AppContext = Depends(get_ctx),
                 db: Session = Depends(get_db), user: User = Depends(current_user)) -> dict:
    unknown = set(body.scopes) - set(SCOPES)
    if unknown:
        raise ValidationFailed(f"Unknown scope(s): {', '.join(sorted(unknown))}.")
    if "admin" in body.scopes and not user.is_admin:
        raise ForbiddenError("Only administrators can create tokens with the 'admin' scope.")
    tok = getattr(request.state, "token", None)
    if tok and tok.get("kind") == "api":
        raise ForbiddenError("API tokens cannot create other tokens – sign in to the web interface.")
    raw = "lal_" + new_token()
    row = ApiToken(user_id=user.id, token_hash=hash_token(raw), kind="api", name=body.name, prefix=raw[:12],
                   scopes=sorted(set(body.scopes)), expires_at=utcnow() + timedelta(days=body.expires_days))
    db.add(row)
    db.commit()
    audit(ctx, user, "token.create", body.name, request=request, details={"scopes": row.scopes, "id": row.id})
    return {"token": raw, "note": "Copy the token now – it is shown only once.", **_token_out(row)}


@router.delete("/tokens/{token_id}", status_code=204)
def revoke_token(token_id: int, request: Request, ctx: AppContext = Depends(get_ctx), db: Session = Depends(get_db),
                 user: User = Depends(current_user)) -> None:
    row = db.get(ApiToken, token_id)
    if row is None or row.user_id != user.id or row.kind != "api":
        raise NotFoundError("Token not found.")
    row.revoked_at = row.revoked_at or utcnow()
    db.commit()
    audit(ctx, user, "token.revoke", row.name, request=request, details={"id": row.id})


@router.post("/sessions/revoke-all")
def revoke_all(request: Request, ctx: AppContext = Depends(get_ctx), db: Session = Depends(get_db),
               user: User = Depends(current_user)) -> dict:
    """Sign out everywhere: revoke every session and API token of this account except the current session."""
    current = (getattr(request.state, "token", None) or {}).get("id")
    n = 0
    for t in db.scalars(select(ApiToken).where(ApiToken.user_id == user.id, ApiToken.revoked_at.is_(None))):
        if t.id != current:
            t.revoked_at = utcnow()
            n += 1
    db.commit()
    audit(ctx, user, "token.revoke_all", user.username, request=request, details={"revoked": n})
    return {"revoked": n}


@router.get("/users")
def list_users(db: Session = Depends(get_db), _: User = Depends(require_admin)) -> dict:
    rows = db.scalars(select(User).order_by(User.id)).all()
    return {"users": [{"id": u.id, "username": u.username, "is_admin": u.is_admin,
                       "created_at": u.created_at.isoformat() + "Z"} for u in rows if u.username != "local"]}
