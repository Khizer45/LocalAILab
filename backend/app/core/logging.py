"""Structured logging with redaction.

Rules enforced here (and relied on by the privacy tests):
* Log records are JSON lines (or plain text when LOG_JSON=false).
* Secrets that look like bearer tokens / API keys are masked.
* Any single log message is truncated, so a bug that logs a whole document
  or prompt cannot dump confidential content into the log file.
Application code must log *metadata* (ids, lengths, timings), never content.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

MAX_MESSAGE_CHARS = 500

_SECRET_PATTERNS = [
    (re.compile(r"(?i)(bearer\s+)[a-z0-9._\-]{8,}"), r"\1***"),
    (re.compile(r"(?i)((?:api[_-]?key|token|password|secret)[\"']?\s*[:=]\s*[\"']?)[^\s\"',}]{4,}"), r"\1***"),
]


def redact(text: str) -> str:
    for pattern, repl in _SECRET_PATTERNS:
        text = pattern.sub(repl, text)
    if len(text) > MAX_MESSAGE_CHARS:
        text = text[:MAX_MESSAGE_CHARS] + f"…[truncated {len(text) - MAX_MESSAGE_CHARS} chars]"
    return text


class RedactionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001  # pragma: no cover - defensive
            return True
        record.msg = redact(msg)
        record.args = None
        return True


class JsonFormatter(logging.Formatter):
    RESERVED = set(vars(logging.LogRecord("x", 0, "x", 0, "x", None, None)).keys()) | {"message"}

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in self.RESERVED and not key.startswith("_"):
                payload[key] = value if isinstance(value, (int, float, bool, type(None))) else redact(str(value))
        if record.exc_info:
            payload["exc"] = redact(self.formatException(record.exc_info))
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO", json_logs: bool = True, log_dir: Path | None = None) -> None:
    root = logging.getLogger()
    root.setLevel(level.upper())
    for h in list(root.handlers):
        if getattr(h, "_local_ai_lab", False):
            root.removeHandler(h)

    formatter: logging.Formatter = (
        JsonFormatter() if json_logs else logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(RotatingFileHandler(log_dir / "app.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"))
    for h in handlers:
        h.setFormatter(formatter)
        h.addFilter(RedactionFilter())
        h._local_ai_lab = True  # type: ignore[attr-defined]
        root.addHandler(h)

    # Quieten noisy libraries; never log request bodies.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
