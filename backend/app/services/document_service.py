"""Document storage, ingestion (parse → chunk → embed → index) and retrieval."""

from __future__ import annotations

import hashlib
import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.core.errors import AppError, ConflictError, NotFoundError, ValidationFailed
from app.core.security import safe_join, sanitize_filename
from app.db.base import utcnow
from app.models.document import Chunk, Document
from app.services.chunker import chunk_pages
from app.services.document_parser import detect_type, parse_document
from app.services.embeddings import Embedder, HashingEmbedder, OllamaEmbedder, from_bytes, to_bytes
from app.services.model_registry import ModelRegistry
from app.services.prompt_safety import looks_like_injection
from app.services.settings_service import SettingsService

log = logging.getLogger("app.documents")


@dataclass
class _Index:
    chunk_ids: np.ndarray
    doc_ids: np.ndarray
    matrix: np.ndarray


class DocumentService:
    def __init__(self, settings: Settings, SessionLocal: sessionmaker[Session], settings_service: SettingsService,
                 registry: ModelRegistry):
        self.settings = settings
        self.SessionLocal = SessionLocal
        self.settings_service = settings_service
        self.registry = registry
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="doc-ingest")
        self._lock = threading.Lock()
        self._index: dict[tuple[int, str], _Index] = {}
        self._hashing = HashingEmbedder()

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------ embedder
    def embedder_status(self) -> dict:
        emb, note = self.current_embedder()
        return {"backend": emb.backend, "model": emb.model, "key": emb.key, "note": note}

    def current_embedder(self) -> tuple[Embedder, str | None]:
        wanted = (self.settings_service.get().embedding_model or "").strip()
        if not wanted or wanted == "hashing-v1":
            return self._hashing, "Lexical hashing embeddings (offline fallback) are selected."
        models, status = self.registry.list_models()
        for m in models:
            if m["provider"] == "local" and m["name"] in (wanted, f"{wanted}:latest"):
                return OllamaEmbedder(self.registry.local, m["name"]), None
        if not status["local"].get("reachable"):
            note = "Ollama is not reachable – using lexical hashing embeddings instead."
        else:
            note = f"Embedding model '{wanted}' is not installed (ollama pull {wanted}) – using lexical hashing embeddings."
        return self._hashing, note

    # ------------------------------------------------------------ upload
    def save_upload(self, db: Session, owner_id: int, filename: str, data: bytes) -> Document:
        rs = self.settings_service.get()
        if not data:
            raise ValidationFailed("The file is empty.", code="empty_file")
        if len(data) > rs.max_upload_mb * 1024 * 1024:
            raise ValidationFailed(f"File is larger than the {rs.max_upload_mb} MB limit.", code="file_too_large")
        file_type = detect_type(filename, data[:2048])
        digest = hashlib.sha256(data).hexdigest()
        existing = db.scalar(
            select(Document).where(Document.owner_id == owner_id, Document.sha256 == digest, Document.status != "failed")
        )
        if existing:
            raise ConflictError(f"This file is already in your library as '{existing.filename}'.", code="duplicate_document")

        stored_name = f"{uuid.uuid4().hex}.{file_type}"
        path = safe_join(self.settings.uploads_dir, str(owner_id), stored_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        doc = Document(owner_id=owner_id, filename=sanitize_filename(filename), stored_name=stored_name,
                       file_type=file_type, size_bytes=len(data), sha256=digest, status="pending")
        db.add(doc)
        db.commit()
        db.refresh(doc)
        log.info("document uploaded", extra={"document_id": doc.id, "bytes": len(data), "type": file_type})
        self.schedule(doc.id)
        return doc

    def schedule(self, doc_id: int) -> None:
        if self.settings.doc_processing_mode == "inline":
            self.process(doc_id)
        else:
            self._executor.submit(self.process, doc_id)

    # ------------------------------------------------------------ ingestion
    def process(self, doc_id: int) -> None:
        with self.SessionLocal() as db:
            doc = db.get(Document, doc_id)
            if doc is None:
                return
            doc.status, doc.error = "processing", None
            db.commit()
            try:
                path = safe_join(self.settings.uploads_dir, str(doc.owner_id), doc.stored_name)
                pages = parse_document(path, doc.file_type, max_pages=self.settings.max_pdf_pages,
                                       pdf_parser=self.settings.pdf_parser)
                chunks = chunk_pages(pages, self.settings.chunk_size, self.settings.chunk_overlap)
                if not chunks:
                    raise ValidationFailed(
                        "No extractable text found. Scanned/image-only PDFs need OCR, which is not supported yet.",
                        code="no_text",
                    )
                embedder, _ = self.current_embedder()
                vectors = embedder.embed_documents([c.text for c in chunks])
                db.query(Chunk).filter(Chunk.document_id == doc.id).delete()
                flagged = 0
                for i, (c, v) in enumerate(zip(chunks, vectors, strict=True)):
                    is_flagged = looks_like_injection(c.text)
                    flagged += int(is_flagged)
                    db.add(Chunk(document_id=doc.id, chunk_index=i, page=c.page, text=c.text,
                                 embedding=to_bytes(v), flagged=is_flagged))
                doc.page_count = sum(1 for p, t in pages if t.strip()) if doc.file_type == "pdf" else 1
                doc.chunk_count = len(chunks)
                doc.char_count = sum(len(c.text) for c in chunks)
                doc.flagged_chunks = flagged
                doc.embedding_backend = embedder.backend
                doc.embedding_model = embedder.model
                doc.embedding_dim = int(vectors.shape[1])
                doc.status = "ready"
                doc.processed_at = utcnow()
                db.commit()
                log.info("document indexed", extra={"document_id": doc.id, "chunks": len(chunks),
                                                    "embedder": embedder.key, "flagged": flagged})
            except Exception as exc:  # noqa: BLE001
                db.rollback()
                doc = db.get(Document, doc_id)
                if doc is not None:
                    doc.status = "failed"
                    doc.error = exc.message if isinstance(exc, AppError) else f"Processing failed ({type(exc).__name__})."
                    db.commit()
                log.warning("document processing failed",
                            extra={"document_id": doc_id, "error_type": type(exc).__name__})
            finally:
                self.invalidate()

    # ------------------------------------------------------------ management
    def get_owned(self, db: Session, owner_id: int, doc_id: int) -> Document:
        doc = db.get(Document, doc_id)
        if doc is None or doc.owner_id != owner_id:
            raise NotFoundError("Document not found.")
        return doc

    def delete(self, db: Session, owner_id: int, doc_id: int) -> None:
        doc = self.get_owned(db, owner_id, doc_id)
        try:
            path = safe_join(self.settings.uploads_dir, str(owner_id), doc.stored_name)
            path.unlink(missing_ok=True)
        except AppError:
            pass
        db.query(Chunk).filter(Chunk.document_id == doc.id).delete()
        db.delete(doc)
        db.commit()
        self.invalidate()
        log.info("document deleted", extra={"document_id": doc_id})

    def reindex(self, db: Session, owner_id: int, doc_id: int) -> Document:
        doc = self.get_owned(db, owner_id, doc_id)
        if doc.status == "processing":
            raise ConflictError("Document is already being processed.")
        doc.status = "pending"
        db.commit()
        self.schedule(doc.id)
        return doc

    def invalidate(self) -> None:
        with self._lock:
            self._index.clear()

    # ------------------------------------------------------------ retrieval
    def _load_index(self, owner_id: int, key: str) -> _Index | None:
        with self._lock:
            if (owner_id, key) in self._index:
                return self._index[(owner_id, key)]
        backend, _, model = key.partition(":")
        with self.SessionLocal() as db:
            rows = db.execute(
                select(Chunk.id, Chunk.document_id, Chunk.embedding)
                .join(Document, Document.id == Chunk.document_id)
                .where(Document.owner_id == owner_id, Document.status == "ready",
                       Document.embedding_backend == backend, Document.embedding_model == model)
            ).all()
        if not rows:
            return None
        idx = _Index(
            chunk_ids=np.array([r[0] for r in rows]),
            doc_ids=np.array([r[1] for r in rows]),
            matrix=np.vstack([from_bytes(r[2]) for r in rows]),
        )
        with self._lock:
            self._index[(owner_id, key)] = idx
        return idx

    def retrieve(self, owner_id: int, question: str, *, top_k: int, document_ids: list[int] | None = None,
                 min_score: float | None = None) -> dict:
        """Return {sources, embedder, min_score, stale_documents}. Only the owner's documents are searched.

        ``min_score`` overrides the threshold used to *filter* chunks (retrieval evaluation passes -inf to see the
        full ranking); the returned ``min_score`` is always the threshold used in production."""
        embedder, note = self.current_embedder()
        rs = self.settings_service.get()
        threshold = rs.rag_min_score if rs.rag_min_score is not None else embedder.default_min_score
        min_score_used = threshold if min_score is None else min_score
        with self.SessionLocal() as db:
            ready = db.scalars(select(Document).where(Document.owner_id == owner_id, Document.status == "ready")).all()
        stale = [d.id for d in ready if f"{d.embedding_backend}:{d.embedding_model}" != embedder.key]
        result = {"sources": [], "embedder": embedder.key, "embedder_note": note, "min_score": threshold,
                  "stale_documents": stale, "searched_documents": len(ready) - len(stale)}
        idx = self._load_index(owner_id, embedder.key)
        if idx is None or not question.strip():
            return result
        mask = np.ones(len(idx.chunk_ids), dtype=bool)
        if document_ids:
            mask = np.isin(idx.doc_ids, np.array(document_ids))
        if not mask.any():
            return result
        q = embedder.embed_query(question)
        if q.shape[0] != idx.matrix.shape[1]:
            return result
        scores = idx.matrix[mask] @ q
        chunk_ids = idx.chunk_ids[mask]
        order = np.argsort(-scores)[:top_k]
        picked = [(int(chunk_ids[i]), float(scores[i])) for i in order if scores[i] >= min_score_used]
        if not picked:
            return result
        with self.SessionLocal() as db:
            chunk_map = {c.id: c for c in db.scalars(select(Chunk).where(Chunk.id.in_([p[0] for p in picked])))}
            doc_map = {d.id: d for d in db.scalars(
                select(Document).where(Document.id.in_({c.document_id for c in chunk_map.values()}),
                                       Document.owner_id == owner_id))}
        for n, (cid, score) in enumerate(picked, start=1):
            c = chunk_map.get(cid)
            d = doc_map.get(c.document_id) if c else None
            if c is None or d is None:  # defensive: never return another owner's chunk
                continue
            result["sources"].append({
                "n": n, "chunk_id": c.id, "document_id": d.id, "filename": d.filename, "page": c.page,
                "chunk_index": c.chunk_index, "score": round(score, 4), "text": c.text, "flagged": c.flagged,
            })
        return result
