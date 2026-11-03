from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, LargeBinary, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, utcnow


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    filename: Mapped[str] = mapped_column(String(200))  # sanitized display name
    stored_name: Mapped[str] = mapped_column(String(80))  # uuid-based, never user controlled
    file_type: Mapped[str] = mapped_column(String(8))  # pdf | txt | md
    size_bytes: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(16), default="pending")  # pending|processing|ready|failed
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    char_count: Mapped[int] = mapped_column(Integer, default=0)
    flagged_chunks: Mapped[int] = mapped_column(Integer, default=0)
    embedding_backend: Mapped[str | None] = mapped_column(String(32), nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(200), nullable=True)
    embedding_dim: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )


class Chunk(Base):
    __tablename__ = "chunks"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[bytes] = mapped_column(LargeBinary)
    flagged: Mapped[bool] = mapped_column(Boolean, default=False)

    document: Mapped[Document] = relationship(back_populates="chunks")
