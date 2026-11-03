"""User benchmark suites with version history, evaluator try-it, experiment bundles (export/verify/import/rerun)."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

from conftest import wait_for

SUITE = {"id": "my-support", "name": "Support answers", "version": "1.0.0", "default_max_tokens": 50,
         "cases": [{"id": "c1", "category": "qa", "prompt": "What is the capital of Japan? Answer with the city name only.",
                    "evaluator": {"type": "exact_match", "answers": ["Tokyo"]}},
                   {"id": "c2", "category": "math", "prompt": "What is 17 multiplied by 6? Reply with just the number.",
                    "evaluator": {"type": "numeric", "value": 102}}]}


def _wait(c, path):
    return wait_for(lambda: (d := c.get(path).json())["status"] not in ("queued", "running") and d, timeout=90)


def test_user_suites_keep_every_version(client):
    r = client.post("/api/benchmarks/suites", json={"suite": SUITE})
    assert r.status_code == 201 and r.json()["version"] == "1.0.0" and r.json()["source"] == "custom v1.0.0"
    assert client.post("/api/benchmarks/suites", json={"suite": SUITE}).json()["version"] == "1.0.0"  # unchanged
    changed = json.loads(json.dumps(SUITE))
    changed["cases"][0]["evaluator"]["answers"].append("Tokyo, Japan")
    assert client.post("/api/benchmarks/suites", json={"suite": changed, "note": "accept long form"}).json()["version"] == "1.0.1"
    versions = client.get("/api/benchmarks/suites/my-support/versions").json()["versions"]
    assert [v["version"] for v in versions] == ["1.0.0", "1.0.1"] and versions[1]["note"] == "accept long form"
    old = client.get("/api/benchmarks/suites/my-support?version=1.0.0").json()
    assert old["cases"][0]["evaluator"]["answers"] == ["Tokyo"]
    run = _wait(client, f"/api/benchmarks/runs/{client.post('/api/benchmarks/runs', json={'suite_id': 'my-support', 'models': ['sim-medium:3b']}).json()['id']}")
    assert run["status"] == "completed" and run["suite_version"] == "1.0.1"
    bad = client.post("/api/benchmarks/suites", json={"suite": {**SUITE, "id": "quick"}})
    assert bad.status_code == 422 and "built-in" in bad.json()["error"]["message"]
    bad = client.post("/api/benchmarks/suites", json={"suite": {**SUITE, "cases": [{**SUITE["cases"][0],
                                                       "evaluator": {"type": "label", "labels": ["a"], "answer": "b"}}]}})
    assert bad.status_code == 422
    assert client.delete("/api/benchmarks/suites/my-support").status_code == 204
    assert "my-support" not in {s["id"] for s in client.get("/api/benchmarks/suites").json()["suites"]}


def test_evaluator_try_it(client):
    r = client.post("/api/benchmarks/evaluators/try", json={"response": "Final answer: 42",
                                                            "evaluator": {"type": "numeric", "value": 42}}).json()
    assert r["passed"] and r["evaluator_version"] == "numeric@2"
    bad = client.post("/api/benchmarks/evaluators/try", json={"response": "x", "evaluator": {"type": "regex", "pattern": "("}})
    assert bad.status_code == 422


def test_benchmark_bundle_round_trip_and_metadata_scope(client):
    run = _wait(client, f"/api/benchmarks/runs/{client.post('/api/benchmarks/runs', json={'suite_id': 'quick', 'models': ['sim-small:1b'], 'max_cases': 4}).json()['id']}")
    full = client.get(f"/api/bundles/benchmark/{run['id']}?scope=full").content
    meta = client.get(f"/api/bundles/benchmark/{run['id']}?scope=metadata").content
    zf = zipfile.ZipFile(io.BytesIO(meta))
    manifest = json.loads(zf.read("manifest.json"))
    assert manifest["contains_user_data"] is False
    assert all("response" not in json.loads(line) for line in zf.read("results.jsonl").decode().splitlines())
    assert "prompt" not in zf.read("suite.json").decode()
    v = client.post("/api/bundles/verify", files={"file": ("b.zip", full)}).json()
    assert v["ok"] and v["kind"] == "benchmark" and v["contains_user_data"] is True
    imp = client.post("/api/bundles/import", files={"file": ("b.zip", full)})
    assert imp.status_code == 201
    copy = client.get(f"/api/benchmarks/runs/{imp.json()['run_id']}").json()
    assert copy["name"].startswith("[imported]") and len(copy["results"]) == len(run["results"])
    assert [s["quality_score"] for s in copy["summary"]] == [s["quality_score"] for s in run["summary"]]
    re = client.post("/api/bundles/rerun", files={"file": ("b.zip", full)}, data={"models": "sim-medium:3b"})
    assert re.status_code == 201 and re.json()["suite_id"] == "quick"
    again = _wait(client, f"/api/benchmarks/runs/{re.json()['run_id']}")
    assert again["models"] == ["sim-medium:3b"] and again["config"]["cases_sha256"] == run["config"]["cases_sha256"]


def test_tampered_bundle_fails_verification(client):
    run = _wait(client, f"/api/benchmarks/runs/{client.post('/api/benchmarks/runs', json={'suite_id': 'quick', 'models': ['sim-small:1b'], 'max_cases': 2}).json()['id']}")
    blob = client.get(f"/api/bundles/benchmark/{run['id']}").content
    src = zipfile.ZipFile(io.BytesIO(blob))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        for n in src.namelist():
            data = src.read(n)
            if n == "results.jsonl":
                data = data.replace(b'"sim-small:1b"', b'"sim-small:9b"')
            z.writestr(n, data)
    v = client.post("/api/bundles/verify", files={"file": ("t.zip", out.getvalue())}).json()
    assert not v["ok"] and any("checksum" in p for p in v["problems"])
    assert client.post("/api/bundles/import", files={"file": ("t.zip", out.getvalue())}).status_code == 422


def test_arena_and_agent_bundles_recompute_metrics(client):
    sample = Path(__file__).resolve().parents[2] / "samples" / "arena" / "customer_churn.csv"
    with sample.open("rb") as f:
        ds = client.post("/api/arena/datasets", files={"file": ("c.csv", f)}, data={"target": "churn"}).json()
    r = client.post("/api/arena/runs", json={"dataset_id": ds["id"], "algorithms": ["logistic_regression"],
                                             "cv_folds": 0, "bootstrap_resamples": 300})
    run = _wait(client, f"/api/arena/runs/{r.json()['id']}")
    blob = client.get(f"/api/bundles/arena/{run['id']}?include_dataset=true").content
    names = zipfile.ZipFile(io.BytesIO(blob)).namelist()
    assert "data/c.csv" in names and "dataset.json" in names
    v = client.post("/api/bundles/verify", files={"file": ("a.zip", blob)}).json()
    assert v["ok"] and any("recomputed" in c["detail"] for c in v["checks"])
    lineage = client.get(f"/api/arena/datasets/{ds['id']}/runs").json()
    assert lineage["runs"][0]["id"] == run["id"] and lineage["runs"][0]["protocol"] == "v2"

    ar = client.post("/api/agents/runs", json={"suite_id": "agent-quick", "models": ["sim-medium:3b"]}).json()
    agent_run = _wait(client, f"/api/agents/runs/{ar['id']}")
    ablob = client.get(f"/api/bundles/agent/{agent_run['id']}?scope=metadata").content
    assert client.post("/api/bundles/verify", files={"file": ("g.zip", ablob)}).json()["ok"]
    assert client.post("/api/bundles/rerun", files={"file": ("g.zip", ablob)}).status_code == 422  # needs full scope
