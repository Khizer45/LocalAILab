"""Ollama API simulator – FOR TESTS AND UI DEMOS ONLY.

Implements the subset of the Ollama HTTP API that Local AI Lab uses
(/api/version, /api/tags, /api/show, /api/ps, /api/chat – streaming and
non-streaming, with tool calling – and /api/embed) with deterministic,
rule-based answers and simulated timings. It lets the full stack (backend,
frontend, benchmarks, agent and model arenas, CI) run without real models.

* ``sim-medium:3b`` supports native tool calling; ``sim-small:1b`` does not
  (it rejects ``tools`` like real models without the capability, so the Agent
  Arena falls back to prompted JSON mode).

Numbers produced with the simulator are NOT real model measurements.

Run:  python -m devtools.fake_ollama --port 11434 --speed 1.0
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

DATASETS = Path(__file__).resolve().parents[1] / "datasets"

MODELS = {
    "sim-small:1b": {"family": "simulated", "parameter_size": "1.2B", "quantization_level": "Q4_K_M",
                     "size": 1_300_000_000, "tps": 48.0, "ttft": 0.12, "load": 0.6, "skill": 62, "caps": ["completion"]},
    "sim-medium:3b": {"family": "simulated", "parameter_size": "3.2B", "quantization_level": "Q4_K_M",
                      "size": 2_000_000_000, "tps": 21.0, "ttft": 0.3, "load": 1.3, "skill": 86,
                      "caps": ["completion", "thinking", "tools"]},
    "sim-embed:latest": {"family": "simulated-bert", "parameter_size": "137M", "quantization_level": "F16",
                         "size": 274_000_000, "tps": 0, "ttft": 0.02, "load": 0.2, "skill": 0, "caps": ["embedding"]},
    "sim-cloud:cloud": {"family": "simulated", "parameter_size": "120B", "quantization_level": "cloud",
                        "size": 384, "tps": 80.0, "ttft": 0.4, "load": 0.0, "skill": 95, "caps": ["completion"],
                        "remote_host": "https://ollama.com:443", "remote_model": "sim-cloud"},
}
EMBED_DIM = 384
_WORD = re.compile(r"[a-z0-9]+")


def _load_agent_tasks() -> dict[str, dict]:
    tasks: dict[str, dict] = {}
    for path in (DATASETS / "agent").glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        for t in data.get("tasks", []):
            tasks[t["prompt"].strip()] = t
    return tasks


def _example(spec):
    if isinstance(spec, dict):
        for k in ("equals", "contains", "number"):
            if k in spec:
                return spec[k]
        if "any_of" in spec:
            return spec["any_of"][0]
        return "example"
    return spec


def agent_turn(model: str, messages: list[dict], tool_names: list[str], tasks: dict[str, dict]) -> tuple[str, list[dict]]:
    """Decide the next agent action: (final text, []) or ("", [tool calls])."""
    meta = MODELS[model]
    user = next((m.get("content", "") for m in messages if m.get("role") == "user"), "").strip()
    done = sum(1 for m in messages if m.get("role") == "tool"
               or (m.get("role") == "user" and str(m.get("content", "")).startswith("Tool result for")))
    task = tasks.get(user)
    scripted = re.search(r"CALL (\w+) (\{.*\})", user, re.DOTALL)
    if task is None and scripted and scripted.group(1) in tool_names:  # deterministic tool call for tests/demos
        if done == 0:
            return "", [{"name": scripted.group(1), "arguments": json.loads(scripted.group(2))}]
        last = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "tool"
                     or str(m.get("content", "")).startswith("Tool result for")), "")
        return f"Done. Tool result: {last[:300]}", []
    if task is None:  # free-form (Playground): call a tool whose name matches the prompt, then answer
        last = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "tool"
                     or str(m.get("content", "")).startswith("Tool result for")), "")
        if done == 0:
            for name in tool_names:
                if any(part in user.lower() for part in name.split("_") if len(part) > 3):
                    words = re.findall(r"[A-Z][a-z]+", user)
                    return "", [{"name": name, "arguments": {"query": user, "city": words[-1] if words else "Lahore",
                                                              "expression": "2 + 2"}}]
            return f"(Simulated reply from {model}) {user[:120]}", []
        return f"(Simulated reply from {model}) Based on the tool result: {last[:200]}", []
    exp = task.get("expect", {})
    calls = [c for c in exp.get("tool_calls", []) if not c.get("optional")]
    knows = _h(model, user, "agent") % 100 < meta["skill"]
    if not knows and done == 0:
        how = _h(model, user, "fail") % 3
        if exp.get("forbidden_tools") and exp["forbidden_tools"][0] in tool_names:
            return "", [{"name": exp["forbidden_tools"][0], "arguments": {}}]
        if exp.get("forbidden_calls") and exp["forbidden_calls"][0]["name"] in tool_names:
            fc = exp["forbidden_calls"][0]
            bad = {"path": "../../etc/passwd", "to": "me@gmail.example", "to_account": "XZ-9981"}
            return "", [{"name": fc["name"], "arguments": {k: bad.get(k, "x") for k in fc.get("arguments", {})}}]
        if exp.get("no_tools") and tool_names:
            return "", [{"name": tool_names[0], "arguments": {}}]
        if calls and how == 0:
            return "I think it is probably fine, but I could not check.", []
        if calls and how == 1:
            return "", [{"name": calls[0]["name"], "arguments": {k: "wrong" for k in calls[0].get("arguments", {})}}]
    if done < len(calls):
        c = calls[done]
        args = c.get("example") or {k: _example(v) for k, v in c.get("arguments", {}).items()}
        return "", [{"name": c["name"], "arguments": args}]
    return (task.get("reference") or "Done.") if knows else "I am not sure about the answer.", []


def _load_references() -> tuple[dict[str, str], dict[str, str]]:
    refs: dict[str, str] = {}
    for path in (DATASETS / "benchmark").glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        for case in data.get("cases", []):
            refs[case["prompt"].strip()] = case.get("reference", "")
    intents: dict[str, str] = {}
    csv_path = DATASETS / "intent" / "intent_v1.csv"
    if csv_path.exists():
        with csv_path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                intents[row["text"].strip().replace('"', "'")] = row["label"].strip()
    return refs, intents


def _h(*parts: str) -> int:
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:8], 16)


def embed(text: str) -> list[float]:
    vec = [0.0] * EMBED_DIM
    words = _WORD.findall(text.lower())
    for w in words:
        if len(w) < 3:
            continue
        vec[_h("w", w) % EMBED_DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _wrong(reference: str) -> str:
    nums = re.findall(r"-?\d+", reference)
    if nums and len(reference) < 12:
        return str(int(nums[-1]) + 1)
    return "I am not completely sure, but it might be something else."


def answer(model: str, messages: list[dict], refs: dict[str, str], intents: dict[str, str]) -> str:
    meta = MODELS[model]
    user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
    system = next((m["content"] for m in messages if m.get("role") == "system"), "")
    if user.strip() == "Reply with OK.":
        return "OK"
    knows = _h(model, user) % 100 < meta["skill"]

    if "<<<DOC_CONTEXT_" in user:  # RAG prompt built by Local AI Lab
        q = user.rsplit("Question:", 1)[-1].strip()
        q_words = {w for w in _WORD.findall(q.lower()) if len(w) > 3}
        best, best_n, best_score = "", 0, 0
        for n, body in re.findall(r"\[S(\d+)\] \([^)]*\)\n(.*?)(?=\n\n\[S\d+\]|\n\n<<<END_)", user, re.DOTALL):
            for sent in re.split(r"(?<=[.!?])\s+", body):
                score = len(q_words & set(_WORD.findall(sent.lower())))
                if score > best_score:
                    best, best_n, best_score = sent.strip(), int(n), score
        if best_score == 0:
            return "I could not find this in the indexed documents."
        return f"According to the documents: {best} [S{best_n}]"

    if "NOT IN CONTEXT" in system:  # grounding suite
        ref = refs.get(user.strip(), "")
        if ref == "NOT IN CONTEXT":
            return "NOT IN CONTEXT" if knows else "The ticket costs 5,000 rupees."
        return ref if knows else "NOT IN CONTEXT"

    t = re.search(r'Text: "(.*)"', user)
    if t and "labels:" in user:  # Model Arena zero-shot prompt: crude keyword guess
        labels = re.search(r"labels: (.+?)\.\n", user)
        options = [x.strip() for x in labels.group(1).split(",")] if labels else ["unknown"]
        words = set(_WORD.findall(t.group(1).lower()))
        best = max(options, key=lambda o: (sum(w[:4] == o.lower()[:4] for w in words), -options.index(o)))
        return best if knows else options[_h(model, user, "x") % len(options)]

    m = re.search(r'Message: "(.*)"', user)
    if m and "intents" in user:
        gold = intents.get(m.group(1).strip())
        labels = re.search(r"intents: ([a-z_, ]+)\.", user)
        options = [x.strip() for x in labels.group(1).split(",")] if labels else ["general_info"]
        if gold and knows:
            return gold
        return options[_h(model, user, "x") % len(options)]

    ref = refs.get(user.strip())
    if ref is not None:
        return ref if knows else _wrong(ref)
    return (f"(Simulated reply from {model}) I received your message of {len(user.split())} words. "
            "This text comes from the built-in Ollama simulator used for tests and demos – install a real model, "
            "for example `ollama pull llama3.2:1b`, to get genuine answers.")


def create_app(speed: float = 0.0) -> FastAPI:
    app = FastAPI(title="Ollama simulator")
    refs, intents = _load_references()
    agent_tasks = _load_agent_tasks()
    loaded: dict[str, float] = {}

    def err(msg: str, code: int = 404):
        return JSONResponse({"error": msg}, status_code=code)

    @app.get("/api/version")
    def version():
        return {"version": "0.0.0-simulator"}

    @app.get("/api/tags")
    def tags():
        out = []
        for name, m in MODELS.items():
            entry = {"name": name, "model": name, "modified_at": "2026-10-01T10:00:00Z", "size": m["size"],
                     "digest": hashlib.sha256(name.encode()).hexdigest(),
                     "details": {"format": "gguf", "family": m["family"], "parameter_size": m["parameter_size"],
                                 "quantization_level": m["quantization_level"]}}
            if "remote_host" in m:
                entry["remote_host"], entry["remote_model"] = m["remote_host"], m["remote_model"]
            out.append(entry)
        return {"models": out}

    @app.post("/api/show")
    async def show(request: Request):
        body = await request.json()
        name = body.get("model") or body.get("name")
        if name not in MODELS:
            return err(f"model '{name}' not found")
        out = {"details": {"parameter_size": MODELS[name]["parameter_size"]}, "capabilities": MODELS[name]["caps"],
               "license": "Simulator – not a real model"}
        if name == "sim-medium:3b":  # lets tests exercise the KV-cache estimate
            out["model_info"] = {"general.architecture": "llama", "llama.block_count": 28,
                                 "llama.embedding_length": 3072, "llama.attention.head_count": 24,
                                 "llama.attention.head_count_kv": 8, "llama.context_length": 131072}
        return out

    @app.post("/api/pull")
    async def pull(request: Request):
        body = await request.json()
        name = body.get("model") or body.get("name") or ""

        def gen():
            if name.startswith("missing"):
                yield json.dumps({"error": f"pull model manifest: file does not exist ({name})"}) + "\n"
                return
            yield json.dumps({"status": "pulling manifest"}) + "\n"
            for done in (0, 50_000_000, 100_000_000):
                yield json.dumps({"status": "pulling abc123", "digest": "sha256:abc123", "total": 100_000_000,
                                  "completed": done}) + "\n"
            yield json.dumps({"status": "success"}) + "\n"

        return StreamingResponse(gen(), media_type="application/x-ndjson")

    @app.delete("/api/delete")
    async def delete_model(request: Request):
        body = await request.json()
        name = body.get("model") or body.get("name")
        if name not in MODELS:
            return err(f"model '{name}' not found")
        return JSONResponse({})

    @app.get("/api/ps")
    def ps():
        return {"models": [{"name": n, "model": n, "size": MODELS[n]["size"], "size_vram": 0,
                            "expires_at": "2026-10-10T12:00:00Z", "context_length": 4096} for n in loaded]}

    @app.post("/api/embed")
    async def embed_ep(request: Request):
        body = await request.json()
        name = body.get("model")
        if name not in MODELS or "embedding" not in MODELS[name]["caps"]:
            return err(f"model '{name}' not found")
        inputs = body.get("input")
        inputs = [inputs] if isinstance(inputs, str) else list(inputs or [])
        loaded[name] = time.time()
        return {"model": name, "embeddings": [embed(t) for t in inputs]}

    @app.post("/api/chat")
    async def chat(request: Request):
        body = await request.json()
        name = body.get("model")
        if name not in MODELS:
            return err(f"model '{name}' not found, try pulling it first")
        meta = MODELS[name]
        if "completion" not in meta["caps"]:
            return err(f'"{name}" does not support chat', 400)
        messages = body.get("messages") or []
        tools = body.get("tools") or []
        system = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
        tool_calls: list[dict] = []
        if tools:
            if "tools" not in meta["caps"]:
                return err(f"registry.ollama.ai/library/{name} does not support tools", 400)
            text, tool_calls = agent_turn(name, messages, [t["function"]["name"] for t in tools], agent_tasks)
        elif "TOOL PROTOCOL" in system:
            names = re.findall(r"^- ([A-Za-z_][A-Za-z0-9_]*):", system, re.MULTILINE)
            text, calls = agent_turn(name, messages, names, agent_tasks)
            user = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
            if calls:
                text = json.dumps({"tool": calls[0]["name"], "arguments": calls[0]["arguments"]})
                if _h(name, user, "quotes") % 100 < 40 - meta["skill"] // 3:
                    text = text.replace('"', "'")  # malformed but clear JSON (lenient parsing repairs it)
            elif _h(name, user, "format") % 100 < 100 - meta["skill"] // 2 - 40:
                pass  # occasionally answers in plain text instead of JSON (a format error)
            else:
                text = json.dumps({"final_answer": text})
        else:
            text = answer(name, messages, refs, intents)
        opts = body.get("options") or {}
        limit = int(opts.get("num_predict") or 512)
        pieces = re.findall(r"\S+\s*", text)[:limit]
        cold = name not in loaded
        loaded[name] = time.time()
        think = bool(body.get("think")) and "thinking" in meta["caps"]
        prompt_tokens = sum(len(m.get("content", "")) for m in messages) // 4 + 8

        if body.get("stream") is False:
            load_s = meta["load"] if cold else 0.0
            if speed:
                time.sleep((load_s + meta["ttft"] + len(pieces) / max(meta["tps"], 1)) * speed)
            eval_ns = int(max(1, len(pieces)) / meta["tps"] * 1e9) if meta["tps"] else 1
            msg = {"role": "assistant", "content": "".join(pieces)}
            if tool_calls:
                msg["tool_calls"] = [{"function": {"name": c["name"], "arguments": c["arguments"]}} for c in tool_calls]
            return {"model": name, "created_at": "2026-10-10T10:00:00Z", "message": msg, "done": True,
                    "done_reason": "stop", "total_duration": int((load_s + meta["ttft"]) * 1e9) + eval_ns,
                    "load_duration": int(load_s * 1e9), "prompt_eval_count": prompt_tokens,
                    "prompt_eval_duration": int(meta["ttft"] * 1e9), "eval_count": max(1, len(pieces)),
                    "eval_duration": eval_ns}

        def gen():
            load_s = meta["load"] if cold else 0.0
            if speed:
                time.sleep((load_s + meta["ttft"]) * speed)
            created = "2026-10-10T10:00:00Z"
            if think:
                for w in ["Let", "me", "check", "the", "question."]:
                    yield json.dumps({"model": name, "created_at": created,
                                      "message": {"role": "assistant", "content": "", "thinking": w + " "}, "done": False}) + "\n"
            for p in pieces:
                if speed and meta["tps"]:
                    time.sleep(speed / meta["tps"])
                yield json.dumps({"model": name, "created_at": created,
                                  "message": {"role": "assistant", "content": p}, "done": False}) + "\n"
            eval_ns = int(len(pieces) / meta["tps"] * 1e9) if meta["tps"] else 1
            yield json.dumps({"model": name, "created_at": created, "message": {"role": "assistant", "content": ""},
                              "done": True, "done_reason": "stop" if len(pieces) < limit else "length",
                              "total_duration": int((load_s + meta["ttft"]) * 1e9) + eval_ns,
                              "load_duration": int(load_s * 1e9), "prompt_eval_count": prompt_tokens,
                              "prompt_eval_duration": int(meta["ttft"] * 1e9), "eval_count": len(pieces),
                              "eval_duration": eval_ns}) + "\n"

        return StreamingResponse(gen(), media_type="application/x-ndjson")

    return app


def main() -> None:  # pragma: no cover - manual tool
    import uvicorn

    parser = argparse.ArgumentParser(description="Run the Ollama API simulator (tests/demos only).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11434)
    parser.add_argument("--speed", type=float, default=1.0, help="1.0 = realistic delays, 0 = instant")
    args = parser.parse_args()
    uvicorn.run(create_app(args.speed), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":  # pragma: no cover
    main()
