"""Conversations, history management and streamed replies."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.core.errors import AppError, NotFoundError
from app.models.chat import Conversation, Message
from app.services.inference import GenerationResult, clamp_options, estimate_tokens, stream_generate, strip_think
from app.services.model_registry import ModelRegistry, ResolvedModel
from app.services.rag_service import NOT_FOUND_ANSWER, RagService
from app.services.settings_service import SettingsService

log = logging.getLogger("app.chat")


def ndjson(event: dict) -> bytes:
    return (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")


@dataclass
class PreparedTurn:
    model: ResolvedModel
    messages: list[dict]
    options: dict
    timeout_s: float
    think: bool | None
    conversation_id: int | None = None
    user_message_id: int | None = None
    sources: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    fixed_answer: str | None = None  # answer without calling the model (e.g. nothing retrieved)
    retrieval: dict | None = None


def run_stream(turn: PreparedTurn, on_finish=None) -> Iterator[bytes]:
    """Stream NDJSON events for a prepared turn. `on_finish(result, status, error)` persists."""
    yield ndjson({"type": "start", "conversation_id": turn.conversation_id, "user_message_id": turn.user_message_id,
                  "model": turn.model.ref, "provider": turn.model.provider})
    if turn.sources or turn.warnings or turn.retrieval:
        yield ndjson({"type": "sources", "sources": turn.sources, "warnings": turn.warnings,
                      "retrieval": turn.retrieval})
    result = GenerationResult()
    status, error = "complete", None
    try:
        if turn.fixed_answer is not None:
            result.content = turn.fixed_answer
            result.tokens_source = "none"
            yield ndjson({"type": "token", "content": turn.fixed_answer})
        else:
            for kind, text in stream_generate(turn.model, turn.messages, options=turn.options,
                                              timeout_s=turn.timeout_s, think=turn.think, result=result):
                yield ndjson({"type": kind, "content": text})
    except AppError as exc:
        status, error = ("partial" if result.content else "error"), exc
        yield ndjson({"type": "error", "code": exc.code, "message": exc.message, "hint": exc.hint})
    except GeneratorExit:  # client disconnected / pressed Stop
        status = "partial"
        if on_finish:
            on_finish(result, status, None)
        raise
    except Exception as exc:  # noqa: BLE001
        log.exception("unexpected streaming failure")
        status, error = "error", AppError(f"Unexpected error: {type(exc).__name__}")
        yield ndjson({"type": "error", "code": "internal_error", "message": error.message})
    message_id = on_finish(result, status, error) if on_finish else None
    if status != "error":
        yield ndjson({"type": "done", "message_id": message_id, "status": status, "metrics": result.metrics()})


class ChatService:
    def __init__(self, SessionLocal: sessionmaker[Session], settings_service: SettingsService,
                 registry: ModelRegistry, rag: RagService):
        self.SessionLocal = SessionLocal
        self.settings_service = settings_service
        self.registry = registry
        self.rag = rag

    # ------------------------------------------------------------ CRUD
    @staticmethod
    def get_owned(db: Session, owner_id: int, conversation_id: int) -> Conversation:
        conv = db.get(Conversation, conversation_id)
        if conv is None or conv.owner_id != owner_id:
            raise NotFoundError("Conversation not found.")
        return conv

    def list(self, db: Session, owner_id: int) -> list[Conversation]:
        return list(db.scalars(select(Conversation).where(Conversation.owner_id == owner_id)
                               .order_by(Conversation.updated_at.desc())))

    # ------------------------------------------------------------ turns
    def _history(self, conv: Conversation, system_prompt: str, budget_tokens: int) -> list[dict]:
        msgs = [m for m in conv.messages if m.role in ("user", "assistant") and m.status != "error" and m.content]
        picked: list[dict] = []
        used = estimate_tokens(system_prompt)
        for m in reversed(msgs):
            content = strip_think(m.content) if m.role == "assistant" else m.content
            t = estimate_tokens(content)
            if used + t > budget_tokens and picked:
                break
            picked.append({"role": m.role, "content": content})
            used += t
        picked.reverse()
        return ([{"role": "system", "content": system_prompt}] if system_prompt else []) + picked

    def prepare(self, owner_id: int, conversation_id: int, *, content: str, model: str | None,
                options: dict | None, use_documents: bool | None, document_ids: list[int] | None,
                think: bool | None) -> PreparedTurn:
        rs = self.settings_service.get()
        with self.SessionLocal() as db:
            conv = self.get_owned(db, owner_id, conversation_id)
            resolved = self.registry.resolve(model or conv.model or rs.default_model)
            opts = clamp_options(options if options is not None else conv.options,
                                 max_ctx=rs.max_context_tokens, max_out=rs.max_output_tokens)
            conv.model = resolved.ref
            conv.options = options if options is not None else conv.options
            if use_documents is not None:
                conv.use_documents = use_documents
            if conv.title in ("", "New chat"):
                conv.title = (content.strip().splitlines() or ["New chat"])[0][:60]
            system_prompt = conv.system_prompt or rs.default_system_prompt
            rag = None
            if conv.use_documents:
                rag = self.rag.prepare(owner_id, content, resolved, document_ids, user_system_prompt=system_prompt)
                system_prompt = rag["system"]
            user_msg = Message(conversation_id=conv.id, role="user", content=content)
            db.add(user_msg)
            db.commit()
            db.refresh(conv)
            num_ctx = int(opts.get("num_ctx") or rs.max_context_tokens)
            budget = max(256, num_ctx - int(opts["num_predict"]) - 64)
            messages = self._history(conv, system_prompt, budget)
            turn = PreparedTurn(model=resolved, messages=messages, options=opts, timeout_s=rs.request_timeout_s,
                                think=think, conversation_id=conv.id, user_message_id=user_msg.id)
            if rag is not None:
                turn.sources, turn.warnings, turn.retrieval = rag["public_sources"], rag["warnings"], rag["retrieval"]
                if rag["grounded"]:
                    messages[-1] = {"role": "user", "content": rag["user_message"]}
                else:
                    turn.fixed_answer = NOT_FOUND_ANSWER
            log.info("chat turn prepared", extra={"conversation_id": conv.id, "model": resolved.ref,
                                                 "history_messages": len(messages), "rag": rag is not None})
            return turn

    def stream(self, turn: PreparedTurn) -> Iterator[bytes]:
        def persist(result: GenerationResult, status: str, error: AppError | None) -> int | None:
            if status == "error" and not result.content:
                return None
            with self.SessionLocal() as db:
                msg = Message(conversation_id=turn.conversation_id, role="assistant", content=result.content,
                              thinking=result.thinking or None, model=turn.model.ref, status=status,
                              metrics=result.metrics(), sources=turn.sources or None)
                db.add(msg)
                conv = db.get(Conversation, turn.conversation_id)
                if conv is not None:
                    from app.db.base import utcnow

                    conv.updated_at = utcnow()
                db.commit()
                return msg.id

        return run_stream(turn, persist)
