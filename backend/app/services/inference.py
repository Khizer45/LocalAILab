"""Shared streaming-generation logic used by chat, document Q&A and benchmarks.

Metric provenance (shown in the UI and the reports):
* ``ttft_ms`` / ``total_ms``  – measured by this application around the HTTP
  stream (includes network + model load time). ``ttft_ms`` stops at the first
  token of any kind (including hidden "thinking" tokens); ``ttft_answer_ms`` at
  the first visible answer token – what a user actually waits for.
* ``load_ms``, ``prompt_tokens``, ``output_tokens``, ``tokens_per_s`` –
  reported by the runtime (Ollama ``eval_count`` / ``eval_duration``).
* If the runtime does not report token counts, tokens are *estimated* as
  characters / 4 and ``tokens_source`` is set to ``"estimated"``.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

from app.core.errors import InferenceTimeout
from app.services.model_registry import ResolvedModel

THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
ALLOWED_OPTIONS = {"temperature", "top_p", "top_k", "num_ctx", "num_predict", "seed", "repeat_penalty", "stop", "min_p"}


def strip_think(text: str) -> str:
    """Remove <think>…</think> blocks some reasoning models emit inline."""
    text = THINK_BLOCK.sub("", text)
    if "</think>" in text.lower():  # unterminated opening tag cut at the start
        text = text[text.lower().rfind("</think>") + len("</think>"):]
    return text.strip()


def clamp_options(options: dict | None, *, max_ctx: int, max_out: int) -> dict:
    opts = {k: v for k, v in (options or {}).items() if k in ALLOWED_OPTIONS and v is not None and v != ""}
    if "num_ctx" in opts:
        opts["num_ctx"] = max(256, min(int(opts["num_ctx"]), max_ctx))
    opts["num_predict"] = max(1, min(int(opts.get("num_predict") or max_out), max_out))
    return opts


@dataclass
class GenerationResult:
    content: str = ""
    thinking: str = ""
    ttft_ms: float | None = None  # first token of any kind (thinking or answer)
    ttfa_ms: float | None = None  # first visible answer token
    total_ms: float | None = None
    load_ms: float | None = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    tokens_per_s: float | None = None
    tokens_source: str = "runtime"
    done_reason: str | None = None
    raw_final: dict = field(default_factory=dict)

    def metrics(self) -> dict:
        return {
            "ttft_ms": _r(self.ttft_ms),
            "ttft_answer_ms": _r(self.ttfa_ms),
            "total_ms": _r(self.total_ms),
            "load_ms": _r(self.load_ms),
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "tokens_per_s": _r(self.tokens_per_s),
            "tokens_source": self.tokens_source,
            "done_reason": self.done_reason,
        }


def _r(v: float | None) -> float | None:
    return None if v is None else round(float(v), 2)


def stream_generate(
    model: ResolvedModel,
    messages: list[dict],
    *,
    options: dict,
    timeout_s: float,
    think: bool | None = None,
    result: GenerationResult | None = None,
) -> Iterator[tuple[str, str]]:
    """Stream ("thinking"|"token", text) tuples; fills `result` with metrics.

    Raises InferenceTimeout if the total time exceeds `timeout_s`.
    """
    res = result if result is not None else GenerationResult()
    payload: dict = {"model": model.name, "messages": messages, "options": options}
    if think is not None:
        payload["think"] = think
    start = time.perf_counter()
    deadline = start + timeout_s
    for obj in model.client.chat_stream(payload, timeout=timeout_s):
        now = time.perf_counter()
        msg = obj.get("message") or {}
        thinking = msg.get("thinking") or ""
        token = msg.get("content") or ""
        if (thinking or token) and res.ttft_ms is None:
            res.ttft_ms = (now - start) * 1000
        if thinking:
            res.thinking += thinking
            yield "thinking", thinking
        if token:
            if res.ttfa_ms is None:
                res.ttfa_ms = (now - start) * 1000
            res.content += token
            yield "token", token
        if obj.get("done"):
            res.raw_final = {k: v for k, v in obj.items() if k != "message"}
            res.done_reason = obj.get("done_reason")
            break
        if now > deadline:
            res.total_ms = (now - start) * 1000
            raise InferenceTimeout(f"Generation exceeded the {timeout_s:.0f}s timeout.",
                                   hint="Increase the timeout in Settings or lower the max output tokens.")
    res.total_ms = (time.perf_counter() - start) * 1000
    _fill_runtime_metrics(res)


def _fill_runtime_metrics(res: GenerationResult) -> None:
    f = res.raw_final
    ns = 1e6
    if f.get("load_duration") is not None:
        res.load_ms = f["load_duration"] / ns
    if f.get("prompt_eval_count") is not None:
        res.prompt_tokens = int(f["prompt_eval_count"])
    eval_count, eval_duration = f.get("eval_count"), f.get("eval_duration")
    if eval_count is not None:
        res.output_tokens = int(eval_count)
    if eval_count and eval_duration:
        res.tokens_per_s = eval_count / (eval_duration / 1e9)
        res.tokens_source = "runtime"
    else:
        # Runtime did not report counts (e.g. some cloud responses): estimate.
        est = max(1, round(len(res.content + res.thinking) / 4))
        res.output_tokens = res.output_tokens or est
        gen_ms = (res.total_ms or 0) - (res.ttft_ms or 0)
        res.tokens_per_s = (est / (gen_ms / 1000)) if gen_ms > 0 else None
        res.tokens_source = "estimated"


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)
