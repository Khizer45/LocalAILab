from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, utcnow


class MLTrainingRun(Base):
    __tablename__ = "ml_training_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[str] = mapped_column(String(100))
    dataset_version: Mapped[str] = mapped_column(String(32))
    dataset_hash: Mapped[str] = mapped_column(String(64))
    selected_model: Mapped[str] = mapped_column(String(100))
    metrics: Mapped[dict] = mapped_column(JSON)
    artifact_path: Mapped[str] = mapped_column(String(500))
    mlflow_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
