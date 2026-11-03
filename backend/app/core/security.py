"""Security helpers: password hashing, tokens, safe file paths."""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import secrets
import unicodedata
from pathlib import Path

from app.core.errors import ValidationFailed

# --- passwords (stdlib scrypt, no extra dependency) ---------------------------
_SCRYPT = {"n": 2**14, "r": 8, "p": 1, "dklen": 32}


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, **_SCRYPT)
    return "scrypt$" + base64.b64encode(salt).decode() + "$" + base64.b64encode(digest).decode()


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, salt_b64, digest_b64 = stored.split("$", 2)
        if scheme != "scrypt":
            return False
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(digest_b64)
    except (ValueError, TypeError):
        return False
    actual = hashlib.scrypt(password.encode("utf-8"), salt=salt, **_SCRYPT)
    return hmac.compare_digest(actual, expected)


# --- API tokens ----------------------------------------------------------------
def new_token() -> str:
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


# --- files -----------------------------------------------------------------------
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._\- ()\[\]]+")


def sanitize_filename(name: str, max_len: int = 150) -> str:
    """Return a display-safe file name (never used to build storage paths)."""
    name = unicodedata.normalize("NFKC", name or "")
    name = name.replace("\\", "/").split("/")[-1]  # drop any directory part
    name = _UNSAFE_CHARS.sub("_", name).strip(" .")
    if not name:
        name = "document"
    return name[:max_len]


def safe_join(base: Path, *parts: str) -> Path:
    """Join path parts under `base` and refuse anything that escapes it."""
    base_resolved = base.resolve()
    candidate = base_resolved.joinpath(*parts).resolve()
    if candidate != base_resolved and base_resolved not in candidate.parents:
        raise ValidationFailed("Invalid file path.", code="path_traversal")
    return candidate
