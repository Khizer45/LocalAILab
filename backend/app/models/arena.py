"""Model Arena tables: user datasets, uploaded models, comparison runs and results."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, utcnow


class ArenaDataset(Base):
    __tablename__ = "arena_datasets"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    kind: Mapped[str] = mapped_column(String(16))  # tabular | text | image
    task: Mapped[str] = mapped_column(String(16))  # classification | regression
    target: Mapped[str | None] = mapped_column(String(200), nullable=True)
    text_column: Mapped[str | None] = mapped_column(String(200), nullable=True)
    columns: Mapped[dict] = mapped_column(JSON, default=dict)  # {"features": [...], "ignored": [...], "types": {...}}
    labels: Mapped[list] = mapped_column(JSON, default=list)
    n_rows: Mapped[int] = mapped_column(Integer, default=0)
    n_train: Mapped[int] = mapped_column(Integer, default=0)
    n_test: Mapped[int] = mapped_column(Integer, default=0)
    n_val: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    test_size: Mapped[float] = mapped_column(default=0.2)
    val_size: Mapped[float] = mapped_column(default=0.0, server_default="0")
    seed: Mapped[int] = mapped_column(Integer, default=42)
    split_source: Mapped[str] = mapped_column(String(16), default="random")  # random | group | time | provided
    group_column: Mapped[str | None] = mapped_column(String(200), nullable=True)
    time_column: Mapped[str | None] = mapped_column(String(200), nullable=True)
    split_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    filename: Mapped[str] = mapped_column(String(255))
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    sha256: Mapped[str] = mapped_column(String(64))
    profile: Mapped[dict] = mapped_column(JSON, default=dict)
    warnings: Mapped[list] = mapped_column(JSON, default=list)
    path: Mapped[str] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class ArenaModel(Base):
    __tablename__ = "arena_models"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    filename: Mapped[str] = mapped_column(String(255))
    format: Mapped[str] = mapped_column(String(16))  # onnx | joblib
    kind: Mapped[str] = mapped_column(String(16))  # tabular | text | image
    task: Mapped[str] = mapped_column(String(16))  # classification | regression | features
    role: Mapped[str] = mapped_column(String(24), default="predictor")  # predictor | feature_extractor
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    sha256: Mapped[str] = mapped_column(String(64))
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    inspection: Mapped[dict] = mapped_column(JSON, default=dict)
    trusted: Mapped[bool] = mapped_column(Boolean, default=False)
    # what the uploader declared about the training data (overlap with arena splits is otherwise unknown)
    provenance: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    path: Mapped[str] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class ArenaRun(Base):
    __tablename__ = "arena_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    dataset_id: Mapped[int | None] = mapped_column(ForeignKey("arena_datasets.id", ondelete="SET NULL"), nullable=True)
    name: Mapped[str] = mapped_column(String(200))
    dataset_info: Mapped[dict] = mapped_column(JSON, default=dict)  # snapshot, survives dataset deletion
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    contenders: Mapped[list] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(16), default="queued")
    # "v1" = v1.1 runs (chosen on the test split, exploratory); "v2" = chosen on validation, test used once
    protocol: Mapped[str] = mapped_column(String(16), default="v2", server_default="v1")
    progress_done: Mapped[int] = mapped_column(Integer, default=0)
    progress_total: Mapped[int] = mapped_column(Integer, default=0)
    current_item: Mapped[str | None] = mapped_column(String(300), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    environment: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    summary: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    best_model_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    mlflow_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class ArenaResult(Base):
    __tablename__ = "arena_results"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("arena_runs.id", ondelete="CASCADE"), index=True)
    key: Mapped[str] = mapped_column(String(200))
    source: Mapped[str] = mapped_column(String(16))  # baseline | builtin | uploaded | llm
    name: Mapped[str] = mapped_column(String(300))
    status: Mapped[str] = mapped_column(String(16))  # ok | error | cancelled
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # test split (+ performance)
    val_metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # validation split (selection)
    details: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    predictions: Mapped[list | None] = mapped_column(JSON, nullable=True)  # aligned with evaluated test indices
    evaluated_idx: Mapped[list | None] = mapped_column(JSON, nullable=True)
    val_predictions: Mapped[list | None] = mapped_column(JSON, nullable=True)
    val_idx: Mapped[list | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
