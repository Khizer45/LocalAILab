from __future__ import annotations

from fastapi import APIRouter, Depends, File, Request, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import AppContext, current_user, get_ctx, get_db
from app.core.errors import ValidationFailed
from app.models.core import User
from app.models.document import Chunk, Document
from app.schemas import AskDocuments
from app.schemas.serializers import document_out
from app.services.chat_service import PreparedTurn, run_stream
from app.services.inference import clamp_options
from app.services.rag_service import NOT_FOUND_ANSWER
from app.services.retrieval_eval import RetrievalEvalRequest
from app.services.retrieval_eval import evaluate as retrieval_evaluate

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.get("")
def list_documents(user: User = Depends(current_user), db: Session = Depends(get_db),
                   ctx: AppContext = Depends(get_ctx)) -> dict:
    docs = db.scalars(select(Document).where(Document.owner_id == user.id).order_by(Document.created_at.desc())).all()
    emb = ctx.documents.embedder_status()
    return {"documents": [document_out(d) | {"needs_reindex": d.status == "ready" and
                                              f"{d.embedding_backend}:{d.embedding_model}" != emb["key"]} for d in docs],
            "embedder": emb, "max_upload_mb": ctx.settings_service.get().max_upload_mb}


@router.post("", status_code=201)
async def upload_document(request: Request, file: UploadFile = File(...), user: User = Depends(current_user),
                          ctx: AppContext = Depends(get_ctx)) -> dict:
    limit = ctx.settings_service.get().max_upload_mb * 1024 * 1024
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > limit + 1024 * 1024:
        raise ValidationFailed(f"File is larger than the {limit // 1024 // 1024} MB limit.", code="file_too_large")
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise ValidationFailed(f"File is larger than the {limit // 1024 // 1024} MB limit.", code="file_too_large")
    with ctx.SessionLocal() as db:
        doc = ctx.documents.save_upload(db, user.id, file.filename or "document", data)
        db.expire_all()  # processing may have run inline in another session
        return document_out(db.get(Document, doc.id))


@router.post("/retrieval-eval")
def retrieval_eval(body: RetrievalEvalRequest, user: User = Depends(current_user),
                   ctx: AppContext = Depends(get_ctx)) -> dict:
    """Recall@k / MRR / nDCG@k of the document search on questions with known relevant passages."""
    return retrieval_evaluate(ctx.documents, user.id, body)


@router.get("/{document_id}")
def get_document(document_id: int, user: User = Depends(current_user), db: Session = Depends(get_db),
                 ctx: AppContext = Depends(get_ctx)) -> dict:
    doc = ctx.documents.get_owned(db, user.id, document_id)
    chunks = db.scalars(select(Chunk).where(Chunk.document_id == doc.id).order_by(Chunk.chunk_index).limit(200)).all()
    return document_out(doc) | {"chunks": [
        {"index": c.chunk_index, "page": c.page, "chars": len(c.text), "flagged": c.flagged, "preview": c.text[:300]}
        for c in chunks]}


@router.delete("/{document_id}", status_code=204)
def delete_document(document_id: int, user: User = Depends(current_user), db: Session = Depends(get_db),
                    ctx: AppContext = Depends(get_ctx)) -> None:
    ctx.documents.delete(db, user.id, document_id)


@router.post("/{document_id}/reindex")
def reindex_document(document_id: int, user: User = Depends(current_user), db: Session = Depends(get_db),
                     ctx: AppContext = Depends(get_ctx)) -> dict:
    return document_out(ctx.documents.reindex(db, user.id, document_id))


@router.post("/ask")
def ask_documents(body: AskDocuments, user: User = Depends(current_user), ctx: AppContext = Depends(get_ctx)):
    """Single-turn grounded Q&A over the user's documents, streamed as NDJSON."""
    rs = ctx.settings_service.get()
    model = ctx.registry.resolve(body.model or rs.default_model)
    prep = ctx.rag.prepare(user.id, body.question, model, body.document_ids)
    opts = clamp_options(body.options.model_dump(exclude_none=True) if body.options else {"temperature": 0.1},
                         max_ctx=rs.max_context_tokens, max_out=rs.max_output_tokens)
    turn = PreparedTurn(
        model=model, options=opts, timeout_s=rs.request_timeout_s, think=None,
        messages=[{"role": "system", "content": prep["system"]}, {"role": "user", "content": prep["user_message"] or ""}],
        sources=prep["public_sources"], warnings=prep["warnings"], retrieval=prep["retrieval"],
        fixed_answer=None if prep["grounded"] else NOT_FOUND_ANSWER,
    )
    return StreamingResponse(run_stream(turn), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
