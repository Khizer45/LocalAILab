"""OpenAI-compatible endpoints (``/v1``) on top of the local models.

Point any tool that speaks the OpenAI Chat Completions format – n8n's "OpenAI" credential, LangChain,
LlamaIndex, Continue, curl scripts – at ``http://127.0.0.1:8000/v1``. Compared with calling Ollama directly,
requests go through Local AI Lab's policies: Offline-Only mode and the model allow-list are enforced, cloud
models stay blocked when they should be, users authenticate with scoped API tokens ("chat" scope) when
authentication is on, and document Q&A can be switched on per request with the ``local_ai_lab`` extension::

    {"model": "qwen3:1.7b", "messages": [...], "local_ai_lab": {"use_documents": true}}

Supported: ``GET /v1/models``, ``POST /v1/chat/completions`` (streaming and non-streaming, ``tools`` /
``tool_calls``), ``POST /v1/embeddings``. "OpenAI" is used only to describe the request format; this is not an
OpenAI product and no request is sent to OpenAI.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Literal

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.api.deps import AppContext, current_user, get_ctx
from app.core.errors import AppError, ValidationFailed
from app.models.core import User
from app.services.inference import clamp_options, strip_think

router = APIRouter(prefix="/v1", tags=["openai-compatible"])


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool", "developer"]
    content: str | list[dict] | None = None
    name: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None


class LabOptions(BaseModel):
    use_documents: bool = False
    document_ids: list[int] | None = None


class ChatCompletionRequest(BaseModel):
    model: str = Field(min_length=1, max_length=200)
    messages: list[ChatMessage] = Field(min_length=1, max_length=500)
    temperature: float | None = Field(None, ge=0, le=2)
    top_p: float | None = Field(None, ge=0, le=1)
    max_tokens: int | None = Field(None, ge=1, le=32768)
    max_completion_tokens: int | None = Field(None, ge=1, le=32768)
    seed: int | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    tools: list[dict] | None = None
    tool_choice: Any = None
    local_ai_lab: LabOptions | None = None


class EmbeddingRequest(BaseModel):
    model: str = Field(min_length=1, max_length=200)
    input: str | list[str]


def _text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") in ("text", None))


def _to_ollama_messages(messages: list[ChatMessage]) -> list[dict]:
    out = []
    for m in messages:
        role = "system" if m.role == "developer" else m.role
        msg: dict = {"role": role, "content": _text(m.content)}
        if m.tool_calls:
            calls = []
            for c in m.tool_calls:
                fn = c.get("function") or {}
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except ValueError:
                        args = {}
                calls.append({"function": {"name": fn.get("name"), "arguments": args or {}}})
            msg["tool_calls"] = calls
        if m.role == "tool" and m.name:
            msg["tool_name"] = m.name
        out.append(msg)
    return out


def _openai_tool_calls(calls: list[dict] | None) -> list[dict] | None:
    if not calls:
        return None
    out = []
    for i, c in enumerate(calls):
        fn = c.get("function") or {}
        args = fn.get("arguments")
        out.append({"id": f"call_{uuid.uuid4().hex[:12]}", "type": "function", "index": i,
                    "function": {"name": fn.get("name"), "arguments": args if isinstance(args, str)
                                 else json.dumps(args or {}, ensure_ascii=False)}})
    return out


@router.get("/models")
def list_models(ctx: AppContext = Depends(get_ctx), _: User = Depends(current_user)) -> dict:
    models, _status = ctx.registry.list_models()
    data = [{"id": m["id"], "object": "model", "created": 0, "owned_by": f"local-ai-lab:{m['provider']}"}
            for m in models if m.get("allowed", True)]
    return {"object": "list", "data": data}


@router.post("/chat/completions")
def chat_completions(body: ChatCompletionRequest, ctx: AppContext = Depends(get_ctx),
                     user: User = Depends(current_user)):
    model = ctx.registry.resolve(body.model)
    rs = ctx.settings_service.get()
    messages = _to_ollama_messages(body.messages)
    sources: list[dict] = []
    if body.local_ai_lab and body.local_ai_lab.use_documents:
        question = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        prep = ctx.rag.prepare(user.id, question, model, body.local_ai_lab.document_ids)
        sources = prep["public_sources"]
        if prep["grounded"]:
            messages = [{"role": "system", "content": prep["system"]}] + \
                [m for m in messages if m["role"] != "system"][:-1] + [{"role": "user", "content": prep["user_message"]}]
    stop = [body.stop] if isinstance(body.stop, str) else body.stop
    options = clamp_options({"temperature": body.temperature, "top_p": body.top_p, "seed": body.seed, "stop": stop,
                             "num_predict": body.max_completion_tokens or body.max_tokens},
                            max_ctx=rs.max_context_tokens, max_out=rs.max_output_tokens)
    payload: dict = {"model": model.name, "messages": messages, "options": options}
    if body.tools:
        payload["tools"] = [t for t in body.tools if t.get("type") == "function"]
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    extra = {"local_ai_lab": {"sources": sources, "provider": model.provider}} if body.local_ai_lab else {}

    if not body.stream:
        resp = model.client.chat(payload, timeout=float(rs.request_timeout_s))
        msg = resp.get("message") or {}
        calls = _openai_tool_calls(msg.get("tool_calls"))
        usage = {"prompt_tokens": resp.get("prompt_eval_count") or 0, "completion_tokens": resp.get("eval_count") or 0}
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        message = {"role": "assistant", "content": strip_think(msg.get("content") or "") if not calls else
                   (msg.get("content") or None)}
        if calls:
            message["tool_calls"] = calls
        return {"id": cid, "object": "chat.completion", "created": created, "model": body.model,
                "choices": [{"index": 0, "message": message,
                             "finish_reason": "tool_calls" if calls else (resp.get("done_reason") or "stop")}],
                "usage": usage, **extra}

    def gen():
        def chunk(delta: dict, finish: str | None = None, **more) -> str:
            return "data: " + json.dumps({"id": cid, "object": "chat.completion.chunk", "created": created,
                                          "model": body.model, "choices": [{"index": 0, "delta": delta,
                                                                            "finish_reason": finish}], **more}) + "\n\n"

        yield chunk({"role": "assistant", "content": ""})
        finish = "stop"
        in_think = False
        try:
            for obj in model.client.chat_stream(payload, timeout=float(rs.request_timeout_s)):
                msg = obj.get("message") or {}
                if msg.get("tool_calls"):
                    finish = "tool_calls"
                    yield chunk({"tool_calls": _openai_tool_calls(msg["tool_calls"])})
                token = msg.get("content") or ""
                if "<think>" in token:
                    in_think = True
                if token and not in_think:
                    yield chunk({"content": token})
                if "</think>" in token:
                    in_think = False
                if obj.get("done"):
                    finish = "tool_calls" if finish == "tool_calls" else (obj.get("done_reason") or "stop")
                    break
        except AppError as exc:
            yield "data: " + json.dumps({"error": {"message": exc.message, "code": exc.code}}) + "\n\n"
            yield "data: [DONE]\n\n"
            return
        yield chunk({}, finish, **extra)
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/embeddings")
def embeddings(body: EmbeddingRequest, ctx: AppContext = Depends(get_ctx), _: User = Depends(current_user)) -> dict:
    inputs = [body.input] if isinstance(body.input, str) else list(body.input)
    if not inputs or len(inputs) > 512:
        raise ValidationFailed("Send between 1 and 512 inputs.")
    model = ctx.registry.resolve(body.model, purpose="embedding")
    vectors = model.client.embed(model.name, inputs)
    return {"object": "list", "model": body.model,
            "data": [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vectors)],
            "usage": {"prompt_tokens": sum(len(t) // 4 for t in inputs), "total_tokens": sum(len(t) // 4 for t in inputs)}}
