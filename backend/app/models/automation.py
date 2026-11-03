"""Automation tables: scheduled evaluations/backups and outgoing webhooks (e.g. to n8n)."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, utcnow


class Schedule(Base):
    __tablename__ = "schedules"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    kind: Mapped[str] = mapped_column(String(16))  # benchmark | agent | backup
    request: Mapped[dict] = mapped_column(JSON, default=dict)  # run request (benchmark/agent) or backup options
    every_hours: Mapped[int | None] = mapped_column(Integer, nullable=True)
    daily_at: Mapped[str | None] = mapped_column(String(5), nullable=True)  # "HH:MM", server local time
    gates: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # e.g. {"min_quality": 0.7}
    notify_webhook: Mapped[str | None] = mapped_column(String(100), nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    next_run_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_run_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_run_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    last_status: Mapped[str | None] = mapped_column(String(16), nullable=True)  # submitted|passed|failed|error
    last_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Webhook(Base):
    """A named outgoing webhook. Tools and schedules refer to it by name – never by a free-form URL."""

    __tablename__ = "webhooks"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    url: Mapped[str] = mapped_column(String(500))
    description: Mapped[str] = mapped_column(String(500), default="")
    secret: Mapped[str] = mapped_column(String(100))  # HMAC-SHA256 signing key (X-LocalAILab-Signature)
    allow_agent: Mapped[bool] = mapped_column(Boolean, default=False)  # may the Work Agent call it (with approval)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
