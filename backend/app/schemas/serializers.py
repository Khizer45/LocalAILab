from __future__ import annotations

from datetime import datetime

from app.models.chat import Conversation, Message
from app.models.document import Document


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() + "Z" if dt else None


def conversation_out(c: Conversation, *, with_messages: bool = False) -> dict:
    d = {
        "id": c.id, "title": c.title, "model": c.model, "system_prompt": c.system_prompt, "options": c.options or {},
        "use_documents": c.use_documents, "created_at": _iso(c.created_at), "updated_at": _iso(c.updated_at),
    }
    if with_messages:
        d["messages"] = [message_out(m) for m in c.messages]
    return d


def message_out(m: Message) -> dict:
    return {
        "id": m.id, "role": m.role, "content": m.content, "thinking": m.thinking, "model": m.model,
        "status": m.status, "metrics": m.metrics, "sources": m.sources, "created_at": _iso(m.created_at),
    }


def document_out(d: Document) -> dict:
    return {
        "id": d.id, "filename": d.filename, "file_type": d.file_type, "size_bytes": d.size_bytes,
        "status": d.status, "error": d.error, "page_count": d.page_count, "chunk_count": d.chunk_count,
        "char_count": d.char_count, "flagged_chunks": d.flagged_chunks, "embedding_backend": d.embedding_backend,
        "embedding_model": d.embedding_model, "embedding_dim": d.embedding_dim, "sha256": d.sha256,
        "created_at": _iso(d.created_at), "processed_at": _iso(d.processed_at),
    }
