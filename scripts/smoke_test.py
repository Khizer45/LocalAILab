"""End-to-end smoke test against a RUNNING Local AI Lab backend.

Checks health, the setup check, model discovery, a streamed chat reply, the
OpenAI-compatible API, document indexing + cited answer + retrieval evaluation,
a one-model quick benchmark, the ML baseline, a Model Arena comparison on a small
generated CSV (protocol v2), an Agent Arena run, the Work Agent policy and a
backup that is created, verified and deleted again.

    python scripts/smoke_test.py --base-url http://127.0.0.1:8000 --model llama3.2:1b

Only the Python standard library is used, so it runs with any Python 3.9+.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
import uuid

OK, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []
AUTH: dict[str, str] = {}  # {"Authorization": "Bearer lal_…"} when --token is given


def call(base: str, method: str, path: str, body=None, *, raw: bytes | None = None, headers=None, timeout=600):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    h = {"Content-Type": "application/json"} if body is not None else {}
    h.update(AUTH)
    h.update(headers or {})
    req = urllib.request.Request(base + path, data=data, method=method, headers=h)  # noqa: S310 - http(s) checked in main()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never use a proxy for localhost
    with opener.open(req, timeout=timeout) as r:
        text = r.read().decode("utf-8")
        ctype = r.headers.get("content-type", "")
    if "ndjson" in ctype:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text) if text and "json" in ctype else text


def check(name: str, fn):
    try:
        detail = fn() or ""
        results.append((OK, name, str(detail)))
    except Exception as exc:  # noqa: BLE001
        results.append((FAIL, name, f"{type(exc).__name__}: {exc}"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True, help="an installed chat model, e.g. llama3.2:1b")
    ap.add_argument("--token", help="API token (admin scope) when authentication is on; or set LAL_TOKEN")
    args = ap.parse_args()
    token = args.token or os.environ.get("LAL_TOKEN")
    if token:
        AUTH["Authorization"] = f"Bearer {token}"
    b, model = args.base_url.rstrip("/"), args.model
    if not b.startswith(("http://", "https://")):
        ap.error("--base-url must start with http:// or https://")

    def health():
        r = call(b, "GET", "/api/health/ready")
        assert r["checks"]["ollama"]["ok"], r["checks"]["ollama"]
        return f"ollama {r['checks']['ollama']['version']}, embeddings {r['checks']['embeddings']['backend']}"

    def models():
        ids = [m["id"] for m in call(b, "GET", "/api/models")["models"]]
        assert model in ids, f"{model} not installed; found {ids}"
        return f"{len(ids)} models"

    def chat():
        conv = call(b, "POST", "/api/chat/conversations", {"model": model})
        ev = call(b, "POST", f"/api/chat/conversations/{conv['id']}/messages", {"content": "Reply with one short sentence."})
        assert ev[-1]["type"] == "done", ev[-1]
        call(b, "DELETE", f"/api/chat/conversations/{conv['id']}")
        m = ev[-1]["metrics"]
        return f"{m['output_tokens']} tokens, {m['tokens_per_s']} tok/s ({m['tokens_source']})"

    def documents():
        boundary = uuid.uuid4().hex
        text = f"Smoke test note {boundary[:6]}: the backup generator is tested every Friday at 09:00."
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"smoke_{boundary[:6]}.txt\"\r\n"
                f"Content-Type: text/plain\r\n\r\n{text}\r\n--{boundary}--\r\n").encode()
        doc = call(b, "POST", "/api/documents", raw=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        for _ in range(120):
            d = call(b, "GET", f"/api/documents/{doc['id']}")
            if d["status"] in ("ready", "failed"):
                break
            time.sleep(1)
        assert d["status"] == "ready", d.get("error")
        ev = call(b, "POST", "/api/documents/ask", {"question": "When is the backup generator tested?", "model": model,
                                                     "document_ids": [doc["id"]]})
        src = next(e for e in ev if e["type"] == "sources")["sources"]
        ev_ = call(b, "POST", "/api/documents/retrieval-eval", {"questions": [
            {"id": "q1", "question": "When is the backup generator tested?",
             "relevant": [{"filename": f"smoke_{boundary[:6]}.txt", "contains": "every Friday"}]}],
            "document_ids": [doc["id"]]})
        call(b, "DELETE", f"/api/documents/{doc['id']}")
        assert src, "no source retrieved"
        assert ev_["summary"]["hit@1"] == 1.0, ev_["summary"]
        return f"{d['chunk_count']} chunk(s), cited {src[0]['filename']}, retrieval hit@1 {ev_['summary']['hit@1']}"

    def setup():
        r = call(b, "GET", "/api/setup/check")
        bad = [c["label"] for c in r["checks"] if c["status"] == "fail"]
        assert not any(c["id"] == "database" and c["status"] == "fail" for c in r["checks"]), bad
        return f"overall {r['status']}" + (f" – needs attention: {', '.join(bad)}" if bad else "")

    def openai_api():
        r = call(b, "POST", "/v1/chat/completions", {"model": model, "messages": [
            {"role": "user", "content": "Reply with one short sentence."}], "max_tokens": 40})
        text = r["choices"][0]["message"]["content"]
        assert text.strip(), r
        return f"{r['usage']['completion_tokens']} tokens via /v1/chat/completions"

    def work_agent():
        cfg = call(b, "GET", "/api/work-agent/config")
        rules = {t["name"]: t["policy"] for t in cfg["tools"]}
        risky = {k: rules.get(k) for k in ("write_file", "call_webhook")}
        return "policy " + ", ".join(f"{k}={v}" for k, v in risky.items()) + (
            "" if all(v in ("ask", "deny") for v in risky.values()) else " – WARNING: writes run without approval")

    def backup():
        r = call(b, "POST", "/api/system/backups", {"label": "smoke-test", "include_mlflow": False})
        v = call(b, "POST", f"/api/system/backups/{r['name']}/verify")
        call(b, "DELETE", f"/api/system/backups/{r['name']}")
        assert v["ok"], v
        return f"{v['files']} files verified (schema {v['schema_revision']})"

    def benchmark():
        run = call(b, "POST", "/api/benchmarks/runs", {"suite_id": "quick", "models": [model], "max_cases": 3, "name": "smoke test"})
        for _ in range(600):
            r = call(b, "GET", f"/api/benchmarks/runs/{run['id']}?include_responses=false")
            if r["status"] not in ("queued", "running"):
                break
            time.sleep(1)
        assert r["status"] == "completed", r["status"]
        s = r["summary"][0]
        return f"quality {s['quality_score']}, latency p50 {s['latency_ms_p50']} ms"

    def ml():
        r = call(b, "POST", "/api/ml/baseline/train")
        return f"{r['selected_model']} test accuracy {r['metrics']['test']['accuracy']}"

    def wait(path: str) -> dict:
        for _ in range(900):
            r = call(b, "GET", path)
            if r["status"] not in ("queued", "running"):
                return r
            time.sleep(1)
        raise TimeoutError(path)

    def model_arena():
        boundary = uuid.uuid4().hex
        rows = ["hours,absences,passed"] + [f"{h},{a},{'yes' if h * 2 - a + (i % 3) > 8 else 'no'}"
                                            for i, (h, a) in enumerate((h, a) for h in range(1, 11) for a in range(0, 8))]
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"smoke_students.csv\"\r\n"
                f"Content-Type: text/csv\r\n\r\n" + "\n".join(rows) + f"\r\n--{boundary}\r\n"
                f"Content-Disposition: form-data; name=\"target\"\r\n\r\npassed\r\n--{boundary}--\r\n").encode()
        ds = call(b, "POST", "/api/arena/datasets", raw=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        run = call(b, "POST", "/api/arena/runs", {"dataset_id": ds["id"], "algorithms": ["logistic_regression", "random_forest"],
                                                  "cv_folds": 3, "name": "smoke test"})
        r = wait(f"/api/arena/runs/{run['id']}")
        call(b, "DELETE", f"/api/arena/runs/{run['id']}")
        call(b, "DELETE", f"/api/arena/datasets/{ds['id']}")
        assert r["status"] == "completed", r.get("error")
        bad = [x["error"] for x in r["results"] if x["status"] != "ok"]
        assert not bad, bad
        s = r["summary"]
        assert s.get("protocol") == "v2" and s.get("final"), "expected a protocol v2 result with a final estimate"
        return (f"recommended {s['recommendation']['name']} ({s['recommendation']['status']}), "
                f"final test {s['final']['label']} {s['final']['value']:.3f}")

    def agent_arena():
        run = call(b, "POST", "/api/agents/runs", {"suite_id": "agent-quick", "models": [model], "categories": ["single_tool"],
                                                   "name": "smoke test"})
        r = wait(f"/api/agents/runs/{run['id']}")
        call(b, "DELETE", f"/api/agents/runs/{run['id']}")
        assert r["status"] == "completed", r.get("error")
        s = r["summary"][0]
        return f"{s['modes']} mode, success {s['success_rate']}, latency p50 {s['latency_ms_p50']} ms"

    for name, fn in [("Health", health), ("Setup check", setup), ("Model discovery", models),
                     ("Streaming chat", chat), ("OpenAI-compatible API", openai_api), ("Document Q&A", documents),
                     ("Benchmark", benchmark), ("ML baseline", ml), ("Model Arena", model_arena),
                     ("Agent Arena", agent_arena), ("Work Agent policy", work_agent), ("Backup", backup)]:
        check(name, fn)
    for status, name, detail in results:
        print(f"[{status}] {name}: {detail}")
    return 0 if all(s == OK for s, _, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
