"""The agent loop: model → tool calls → tool results → model … → final answer.

Two tool-calling modes:

* **native** – Ollama's tool-calling API (``tools`` in ``/api/chat``). The model
  returns structured ``tool_calls``. Works with models whose capabilities list
  "tools" (Llama 3.1+/3.2, Qwen 2.5/3, Mistral, …).
* **prompted** – for models without native support: the tools are described in
  the system prompt and the model must answer with a JSON object
  ``{"tool": …, "arguments": …}`` or ``{"final_answer": …}``. This is how every
  model can be compared, and it measures instruction following as well.

Protocol compliance and task success are kept apart: with ``lenient=True``
(default since v1.2) a malformed reply whose intent is clear – single quotes,
trailing commas, Python literals, unquoted keys, a missing closing brace or a
``tool_name(arg="x")`` call – is repaired and executed, and counted as a
*format error*. Task success then measures what the model tried to do; the
protocol-compliance rate measures whether it followed the format.

The loop is a generator of events so the same code drives the benchmark runner
and the live Playground stream.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass

from app.agents.tools import ToolExecutor, ToolSpec
from app.benchmark.evaluators import extract_json
from app.core.errors import AppError, InferenceTimeout
from app.services.inference import strip_think
from app.services.model_registry import ResolvedModel

PROMPTED_PROTOCOL = """You can use tools to answer the user.

TOOL PROTOCOL – follow it exactly:
* To call a tool, reply with ONLY this JSON on one line and nothing else:
  {{"tool": "<tool name>", "arguments": {{<arguments>}}}}
* You will then receive the tool result and can call another tool.
* When you can answer the user, reply with ONLY:
  {{"final_answer": "<your answer>"}}
* Only call tools that are listed below. Do not invent tool results.

Available tools:
{tools}"""

NO_TOOLS_SUPPORT = ("does not support tools", "tools are not supported", "tool use is not supported")


class ToolsUnsupported(AppError):
    status_code = 400
    code = "tools_unsupported"


@dataclass
class AgentConfig:
    mode: str = "native"  # native | prompted
    max_steps: int = 6
    temperature: float = 0.0
    seed: int | None = 42
    num_ctx: int = 4096
    max_tokens: int = 512
    timeout_s: float = 180.0
    lenient: bool = True  # repair malformed prompted-mode replies (counted as format errors)


def _metrics(resp: dict) -> dict:
    out: dict = {}
    if resp.get("prompt_eval_count") is not None:
        out["prompt_tokens"] = int(resp["prompt_eval_count"])
    if resp.get("eval_count") is not None:
        out["output_tokens"] = int(resp["eval_count"])
    if resp.get("eval_count") and resp.get("eval_duration"):
        out["tokens_per_s"] = round(resp["eval_count"] / (resp["eval_duration"] / 1e9), 2)
    if resp.get("load_duration") is not None:
        out["load_ms"] = round(resp["load_duration"] / 1e6, 1)
    return out


def _call_model(model: ResolvedModel, messages: list[dict], cfg: AgentConfig, tools: list[ToolSpec] | None,
                deadline: float) -> tuple[dict, float]:
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        raise InferenceTimeout("The agent exceeded the task time limit.")
    payload: dict = {"model": model.name, "messages": messages,
                     "options": {k: v for k, v in {"temperature": cfg.temperature, "seed": cfg.seed,
                                                   "num_ctx": cfg.num_ctx, "num_predict": cfg.max_tokens}.items()
                                 if v is not None}}
    if tools:
        payload["tools"] = [t.ollama_schema() for t in tools]
    t0 = time.perf_counter()
    try:
        resp = model.client.chat(payload, timeout=max(1.0, remaining))
    except AppError as exc:
        if tools and any(s in exc.message.lower() for s in NO_TOOLS_SUPPORT):
            raise ToolsUnsupported(f"{model.ref} does not support native tool calling.") from exc
        raise
    return resp, (time.perf_counter() - t0) * 1000


def _parse_args(raw) -> dict | str:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            val = json.loads(raw) if raw.strip() else {}
            return val if isinstance(val, dict) else raw
        except ValueError:
            return raw
    return {} if raw is None else raw


def parse_prompted(content: str) -> tuple[str, list[dict], str | None]:
    """Return ("tool"|"final"|"invalid", calls, final_answer)."""
    try:
        obj = extract_json(content)
    except ValueError:
        return "invalid", [], None
    if isinstance(obj, dict):
        if "tool" in obj or "name" in obj and "arguments" in obj:
            return "tool", [{"name": str(obj.get("tool") or obj.get("name")), "arguments": _parse_args(obj.get("arguments"))}], None
        if isinstance(obj.get("tool_calls"), list):
            calls = [{"name": str(c.get("tool") or c.get("name")), "arguments": _parse_args(c.get("arguments"))}
                     for c in obj["tool_calls"] if isinstance(c, dict)]
            return ("tool", calls, None) if calls else ("invalid", [], None)
        if "final_answer" in obj:
            fa = obj["final_answer"]
            return "final", [], fa if isinstance(fa, str) else json.dumps(fa, ensure_ascii=False)
    return "invalid", [], None


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")
_BARE_KEY = re.compile(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:")
_CALL_STYLE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*$", re.DOTALL)
_KWARG = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(\"[^\"]*\"|'[^']*'|-?\d+(?:\.\d+)?|true|false|True|False|None|null)")


def _json_candidates(text: str) -> list[str]:
    t = text.strip()
    m = _FENCE.search(t)
    if m:
        t = m.group(1).strip()
    start = t.find("{")
    if start < 0:
        return []
    t = t[start:]
    end = t.rfind("}")
    base = t[: end + 1] if end >= 0 else t
    out = []
    fixed = _TRAILING_COMMA.sub(r"\1", base)
    fixed = re.sub(r"\bTrue\b", "true", re.sub(r"\bFalse\b", "false", re.sub(r"\bNone\b", "null", fixed)))
    if '"' not in fixed:
        fixed = fixed.replace("'", '"')
    fixed = _BARE_KEY.sub(r'\1"\2":', fixed)
    out.append(fixed)
    missing = fixed.count("{") - fixed.count("}")
    if missing > 0:
        out.append(fixed + "}" * missing)
    return out


def repair_prompted(content: str, tool_names: list[str] | None = None) -> tuple[str, list[dict], str | None, bool]:
    """Second chance for a reply :func:`parse_prompted` rejected. Returns (kind, calls, final, repaired)."""
    for cand in _json_candidates(content):
        try:
            obj = json.loads(cand)
        except ValueError:
            continue
        kind, calls, final = parse_prompted(json.dumps(obj))
        if kind != "invalid":
            return kind, calls, final, True
    m = _CALL_STYLE.match(content.strip().strip("`"))
    if m and tool_names and m.group(1) in tool_names:
        args = {}
        for k, v in _KWARG.findall(m.group(2)):
            try:
                args[k] = json.loads(v.replace("'", '"').replace("True", "true").replace("False", "false")
                                     .replace("None", "null"))
            except ValueError:
                args[k] = v.strip("'\"")
        return "tool", [{"name": m.group(1), "arguments": args}], None, True
    return "invalid", [], None, False


def run_agent(model: ResolvedModel, prompt: str, tools: list[ToolSpec], executor: ToolExecutor, cfg: AgentConfig,
              *, system: str | None = None, cancel=None) -> Iterator[dict]:
    """Run one agent episode, yielding trace events. The last event is always "final" or "error"."""
    start = time.perf_counter()
    deadline = start + cfg.timeout_s
    totals = {"steps": 0, "tool_calls": 0, "invalid_calls": 0, "format_errors": 0, "repaired": 0,
              "prompt_tokens": 0, "output_tokens": 0, "model_ms": 0.0, "tool_ms": 0.0}
    tps: list[float] = []
    messages: list[dict] = []
    if cfg.mode == "prompted":
        protocol = PROMPTED_PROTOCOL.format(tools="\n".join(t.prompt_line() for t in tools) or "(none)")
        messages.append({"role": "system", "content": (system + "\n\n" if system else "") + protocol})
    elif system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    yield {"type": "start", "mode": cfg.mode, "model": model.ref, "tools": [t.name for t in tools]}

    def finish(answer: str, reason: str) -> dict:
        totals["total_ms"] = round((time.perf_counter() - start) * 1000, 1)
        totals["model_ms"] = round(totals["model_ms"], 1)
        totals["tool_ms"] = round(totals["tool_ms"], 2)
        totals["tokens_per_s"] = round(sum(tps) / len(tps), 2) if tps else None
        return {"type": "final", "answer": answer, "stop_reason": reason, "totals": totals}

    try:
        for step in range(1, cfg.max_steps + 1):
            if cancel is not None and cancel.is_set():
                yield finish("", "cancelled")
                return
            resp, ms = _call_model(model, messages, cfg, tools if cfg.mode == "native" else None, deadline)
            totals["steps"] = step
            totals["model_ms"] += ms
            m = _metrics(resp)
            totals["prompt_tokens"] += m.get("prompt_tokens", 0)
            totals["output_tokens"] += m.get("output_tokens", 0)
            if m.get("tokens_per_s"):
                tps.append(m["tokens_per_s"])
            msg = resp.get("message") or {}
            content = msg.get("content") or ""
            event = {"type": "model", "step": step, "ms": round(ms, 1), "content": strip_think(content)[:4000],
                     "thinking_chars": len(msg.get("thinking") or ""), **m}
            if cfg.mode == "native":
                calls = [{"name": str((c.get("function") or {}).get("name", "")),
                          "arguments": _parse_args((c.get("function") or {}).get("arguments"))}
                         for c in (msg.get("tool_calls") or [])]
                event["tool_calls"] = calls
                yield event
                if not calls:
                    yield finish(strip_think(content), "final")
                    return
                messages.append({"role": "assistant", "content": content, "tool_calls": msg.get("tool_calls")})
            else:
                text = strip_think(content)
                kind, calls, final = parse_prompted(text)
                if kind == "invalid" and cfg.lenient:
                    kind, calls, final, repaired = repair_prompted(text, [t.name for t in tools])
                    if repaired:
                        totals["format_errors"] += 1
                        totals["repaired"] += 1
                        event["repaired"] = True
                event["tool_calls"] = calls
                event["parsed"] = kind
                yield event
                if kind == "final":
                    yield finish(final or "", "final")
                    return
                if kind == "invalid":
                    totals["format_errors"] += 1
                    yield finish(strip_think(content), "final")  # plain text is taken as the answer
                    return
                messages.append({"role": "assistant", "content": content})
            for call in calls:
                t0 = time.perf_counter()
                result, meta = executor.call(call["name"], call["arguments"])
                tool_ms = (time.perf_counter() - t0) * 1000
                totals["tool_calls"] += 1
                totals["tool_ms"] += tool_ms
                if not meta["valid"]:
                    totals["invalid_calls"] += 1
                yield {"type": "tool", "step": step, "name": call["name"], "arguments": call["arguments"],
                       "result": result, "ms": round(tool_ms, 2), **meta}
                text = json.dumps(result, ensure_ascii=False, default=str)[:6000]
                if cfg.mode == "native":
                    messages.append({"role": "tool", "content": text, "tool_name": call["name"]})
                else:
                    messages.append({"role": "user", "content": f"Tool result for {call['name']}: {text}\n"
                                     "Reply with the next tool call JSON or the final_answer JSON."})
        yield finish("", "max_steps")
    except ToolsUnsupported:
        raise
    except InferenceTimeout as exc:
        out = finish("", "timeout")
        out["error"] = exc.message
        yield out
    except AppError as exc:
        yield {"type": "error", "code": exc.code, "message": exc.message, "hint": exc.hint,
               "totals": finish("", "error")["totals"]}
