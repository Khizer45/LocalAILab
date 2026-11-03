"""Fill a fresh public demo with example content so visitors see results straight away.

Uses only the standard library and the normal API (no admin rights needed). Every step is optional:
a failure is logged and the next step runs.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
import uuid
from pathlib import Path

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7860"
ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "samples"
MODELS = ["sim-small:1b", "sim-medium:3b"]
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def call(method: str, path: str, body=None, *, raw: bytes | None = None, ctype: str | None = None, timeout=300):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    headers = {"Content-Type": ctype or "application/json"} if data is not None else {}
    req = urllib.request.Request(BASE + path, data=data, method=method, headers=headers)
    with _opener.open(req, timeout=timeout) as r:
        text = r.read().decode("utf-8")
        kind = r.headers.get("content-type", "")
    if "ndjson" in kind:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text) if text and "json" in kind else text


def multipart(file_path: Path, mime: str, fields: dict[str, str] | None = None) -> tuple[bytes, str]:
    b = uuid.uuid4().hex
    parts = [f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{file_path.name}\"\r\n"
             f"Content-Type: {mime}\r\n\r\n".encode() + file_path.read_bytes() + b"\r\n"]
    for k, v in (fields or {}).items():
        parts.append(f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    return b"".join(parts) + f"--{b}--\r\n".encode(), f"multipart/form-data; boundary={b}"


def wait(path: str, done=("completed", "failed", "cancelled", "ready"), limit_s=600) -> dict:
    end = time.time() + limit_s
    while time.time() < end:
        r = call("GET", path)
        if r.get("status") in done:
            return r
        time.sleep(2)
    return r


def step(name: str, fn) -> None:
    try:
        fn()
        print(f"[seed] {name}: ok", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[seed] {name}: skipped ({type(exc).__name__}: {exc})", flush=True)
    time.sleep(2.5)  # stay well inside the per-minute write limit


def documents():
    ids = []
    for f, mime in [("northwind_clinic_leave_policy.txt", "text/plain"), ("project_falcon_notes.md", "text/markdown"),
                    ("it_security_handbook.pdf", "application/pdf")]:
        raw, ct = multipart(SAMPLES / "documents" / f, mime)
        ids.append(call("POST", "/api/documents", raw=raw, ctype=ct)["id"])
    for i in ids:
        wait(f"/api/documents/{i}")
    call("POST", "/api/documents/ask", {"question": "How many days of paid annual leave do full-time employees get?",
                                        "model": MODELS[0]})


def chat():
    conv = call("POST", "/api/chat/conversations", {"model": MODELS[1], "title": "Welcome"})
    call("POST", f"/api/chat/conversations/{conv['id']}/messages",
         {"content": "In two sentences, what can I do with Local AI Lab?"})


def benchmark():
    run = call("POST", "/api/benchmarks/runs", {"suite_id": "quick", "models": MODELS, "name": "Quick suite: 1B vs 3B"})
    wait(f"/api/benchmarks/runs/{run['id']}?include_responses=false")


def ml_baseline():
    call("POST", "/api/ml/baseline/train")


def model_arena():
    raw, ct = multipart(SAMPLES / "arena" / "customer_churn.csv", "text/csv",
                        {"name": "Customer churn (sample)", "target": "churn", "ignore": "customer_id"})
    ds = call("POST", "/api/arena/datasets", raw=raw, ctype=ct)
    run = call("POST", "/api/arena/runs", {
        "dataset_id": ds["id"], "name": "Churn: classic algorithms",
        "algorithms": ["logistic_regression", "random_forest", "gradient_boosting", "knn", "decision_tree"],
        "cv_folds": 3})
    wait(f"/api/arena/runs/{run['id']}")


def agent_arena():
    run = call("POST", "/api/agents/runs", {"suite_id": "agent-quick", "models": MODELS, "name": "Agent quick: 1B vs 3B"})
    wait(f"/api/agents/runs/{run['id']}")


def main() -> None:
    for _ in range(90):
        try:
            if call("GET", "/api/health").get("status"):
                break
        except Exception:  # noqa: BLE001
            time.sleep(2)
    for name, fn in [("documents", documents), ("chat", chat), ("benchmark", benchmark), ("ml baseline", ml_baseline),
                     ("model arena", model_arena), ("agent arena", agent_arena)]:
        step(name, fn)
    print("[seed] done", flush=True)


if __name__ == "__main__":
    main()
