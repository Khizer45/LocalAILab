"""Agent Arena tables: user-defined suites, runs and per-task results."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, utcnow


class AgentSuiteRecord(Base):
    __tablename__ = "agent_suites"
    __table_args__ = (UniqueConstraint("owner_id", "suite_id", name="uq_agent_suite_owner"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    suite_id: Mapped[str] = mapped_column(String(100))
    name: Mapped[str] = mapped_column(String(200))
    version: Mapped[str] = mapped_column(String(32))
    sha256: Mapped[str] = mapped_column(String(64))
    data: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class AgentRun(Base):
    __tablename__ = "agent_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    suite_id: Mapped[str] = mapped_column(String(100))
    suite_version: Mapped[str] = mapped_column(String(32))
    suite_hash: Mapped[str] = mapped_column(String(64))
    suite_snapshot: Mapped[dict] = mapped_column(JSON)  # the exact suite that was run
    models: Mapped[list] = mapped_column(JSON)
    config: Mapped[dict] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(16), default="queued")
    progress_done: Mapped[int] = mapped_column(Integer, default=0)
    progress_total: Mapped[int] = mapped_column(Integer, default=0)
    current_item: Mapped[str | None] = mapped_column(String(300), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    environment: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    model_info: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    mlflow_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    simulated: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class AgentResult(Base):
    __tablename__ = "agent_results"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("agent_runs.id", ondelete="CASCADE"), index=True)
    model: Mapped[str] = mapped_column(String(200), index=True)
    task_id: Mapped[str] = mapped_column(String(100))
    category: Mapped[str] = mapped_column(String(50))
    repeat_index: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(16))  # ok | error | timeout
    mode: Mapped[str] = mapped_column(String(16))  # native | prompted
    success: Mapped[bool] = mapped_column(Boolean, default=False)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    hard_fail: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")  # severe safety violation
    safety_violations: Mapped[list | None] = mapped_column(JSON, nullable=True)
    protocol_ok: Mapped[bool | None] = mapped_column(Boolean, nullable=True)  # no format errors / invalid calls
    checks: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    steps: Mapped[int] = mapped_column(Integer, default=0)
    tool_calls: Mapped[int] = mapped_column(Integer, default=0)
    invalid_calls: Mapped[int] = mapped_column(Integer, default=0)
    format_errors: Mapped[int] = mapped_column(Integer, default=0)
    stop_reason: Mapped[str | None] = mapped_column(String(32), nullable=True)
    final_answer: Mapped[str] = mapped_column(Text, default="")
    trace: Mapped[list | None] = mapped_column(JSON, nullable=True)
    total_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tokens_per_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    peak_ram_used_mb: Mapped[float | None] = mapped_column(Float, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
