from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.api.deps import AppContext, current_user, get_ctx, get_db
from app.models.chat import Conversation, Message
from app.models.core import User
from app.schemas import ConversationCreate, ConversationUpdate, SendMessage
from app.schemas.serializers import conversation_out

router = APIRouter(prefix="/api/chat", tags=["chat"])

STREAM_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


@router.get("/conversations")
def list_conversations(user: User = Depends(current_user), db: Session = Depends(get_db),
                       ctx: AppContext = Depends(get_ctx)) -> dict:
    return {"conversations": [conversation_out(c) for c in ctx.chat.list(db, user.id)]}


@router.post("/conversations", status_code=201)
def create_conversation(body: ConversationCreate, user: User = Depends(current_user), db: Session = Depends(get_db),
                        ctx: AppContext = Depends(get_ctx)) -> dict:
    rs = ctx.settings_service.get()
    conv = Conversation(owner_id=user.id, title=body.title or "New chat", model=body.model or rs.default_model,
                        system_prompt=body.system_prompt or "", use_documents=body.use_documents,
                        options=body.options.model_dump(exclude_none=True) if body.options else {})
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conversation_out(conv, with_messages=True)


@router.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: int, user: User = Depends(current_user), db: Session = Depends(get_db),
                     ctx: AppContext = Depends(get_ctx)) -> dict:
    return conversation_out(ctx.chat.get_owned(db, user.id, conversation_id), with_messages=True)


@router.patch("/conversations/{conversation_id}")
def update_conversation(conversation_id: int, body: ConversationUpdate, user: User = Depends(current_user),
                        db: Session = Depends(get_db), ctx: AppContext = Depends(get_ctx)) -> dict:
    conv = ctx.chat.get_owned(db, user.id, conversation_id)
    data = body.model_dump(exclude_unset=True)
    if "options" in data:
        data["options"] = body.options.model_dump(exclude_none=True) if body.options else {}
    for k, v in data.items():
        if v is not None or k == "options":
            setattr(conv, k, v)
    db.commit()
    db.refresh(conv)
    return conversation_out(conv)


@router.delete("/conversations/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: int, user: User = Depends(current_user), db: Session = Depends(get_db),
                        ctx: AppContext = Depends(get_ctx)) -> None:
    conv = ctx.chat.get_owned(db, user.id, conversation_id)
    db.query(Message).filter(Message.conversation_id == conv.id).delete()
    db.delete(conv)
    db.commit()


@router.delete("/conversations/{conversation_id}/messages", status_code=204)
def clear_conversation(conversation_id: int, user: User = Depends(current_user), db: Session = Depends(get_db),
                       ctx: AppContext = Depends(get_ctx)) -> None:
    conv = ctx.chat.get_owned(db, user.id, conversation_id)
    db.query(Message).filter(Message.conversation_id == conv.id).delete()
    db.commit()


@router.delete("/conversations", status_code=204)
def delete_all_conversations(user: User = Depends(current_user), db: Session = Depends(get_db)) -> None:
    ids = [c.id for c in db.query(Conversation.id).filter(Conversation.owner_id == user.id)]
    if ids:
        db.query(Message).filter(Message.conversation_id.in_(ids)).delete(synchronize_session=False)
        db.query(Conversation).filter(Conversation.id.in_(ids)).delete(synchronize_session=False)
        db.commit()


@router.post("/conversations/{conversation_id}/messages")
def send_message(conversation_id: int, body: SendMessage, user: User = Depends(current_user),
                 ctx: AppContext = Depends(get_ctx)) -> StreamingResponse:
    """Stream the assistant reply as NDJSON events (start, sources, thinking, token, done | error)."""
    turn = ctx.chat.prepare(
        user.id, conversation_id, content=body.content, model=body.model,
        options=body.options.model_dump(exclude_none=True) if body.options else None,
        use_documents=body.use_documents, document_ids=body.document_ids, think=body.think,
    )
    return StreamingResponse(ctx.chat.stream(turn), media_type="application/x-ndjson", headers=STREAM_HEADERS)
