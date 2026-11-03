from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, utcnow


class BenchmarkRun(Base):
    __tablename__ = "benchmark_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    suite_id: Mapped[str] = mapped_column(String(100))
    suite_version: Mapped[str] = mapped_column(String(32))
    suite_hash: Mapped[str] = mapped_column(String(64))
    models: Mapped[list] = mapped_column(JSON)
    config: Mapped[dict] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(16), default="queued")  # queued|running|completed|failed|cancelled
    progress_done: Mapped[int] = mapped_column(Integer, default=0)
    progress_total: Mapped[int] = mapped_column(Integer, default=0)
    current_item: Mapped[str | None] = mapped_column(String(300), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    environment: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    model_info: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    mlflow_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # exact cases, prompts, references and evaluator specs used (frozen at submit time)
    suite_snapshot: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    simulated: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")  # Ollama simulator
    integrity: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # expected vs stored results
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    results: Mapped[list[BenchmarkResult]] = relationship(
        back_populates="run", cascade="all, delete-orphan", passive_deletes=True, order_by="BenchmarkResult.id"
    )


class BenchmarkResult(Base):
    __tablename__ = "benchmark_results"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("benchmark_runs.id", ondelete="CASCADE"), index=True)
    model: Mapped[str] = mapped_column(String(200), index=True)
    case_id: Mapped[str] = mapped_column(String(100))
    category: Mapped[str] = mapped_column(String(50))
    repeat_index: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(16))  # ok | error | timeout | cancelled
    response: Mapped[str] = mapped_column(Text, default="")
    thinking_chars: Mapped[int] = mapped_column(Integer, default=0)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    passed: Mapped[bool] = mapped_column(Boolean, default=False)
    eval_details: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # --- timing (ms). ttft/total are measured by this application ------------
    ttft_ms: Mapped[float | None] = mapped_column(Float, nullable=True)  # first token incl. thinking
    ttft_answer_ms: Mapped[float | None] = mapped_column(Float, nullable=True)  # first visible answer token
    total_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    order_index: Mapped[int | None] = mapped_column(Integer, nullable=True)  # execution order within the run
    evaluator_version: Mapped[str | None] = mapped_column(String(200), nullable=True)
    # --- reported by the model runtime (Ollama) --------------------------------
    load_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tokens_per_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    tokens_source: Mapped[str] = mapped_column(String(16), default="runtime")  # runtime | estimated
    # --- resources (measured by sampler) / reported by runtime ----------------
    peak_ram_used_mb: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_cpu_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    runtime_rss_mb: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_mem_mb: Mapped[float | None] = mapped_column(Float, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    run: Mapped[BenchmarkRun] = relationship(back_populates="results")


class SuiteVersion(Base):
    """Every saved version of a user-defined benchmark or agent suite (history is never overwritten)."""

    __tablename__ = "suite_versions"
    __table_args__ = (UniqueConstraint("owner_id", "kind", "suite_id", "version", name="uq_suite_version"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(16))  # benchmark | agent
    suite_id: Mapped[str] = mapped_column(String(100), index=True)
    version: Mapped[str] = mapped_column(String(32))
    name: Mapped[str] = mapped_column(String(200))
    sha256: Mapped[str] = mapped_column(String(64))
    data: Mapped[dict] = mapped_column(JSON)
    note: Mapped[str | None] = mapped_column(String(300), nullable=True)
    deleted: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
