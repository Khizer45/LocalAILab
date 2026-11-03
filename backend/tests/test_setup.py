"""First-run checks, model catalogue with licences and memory fit, model download / removal."""

from __future__ import annotations

import json

from app.services.setup_service import fit, kv_cache_gb

SNAP = {"ram_total_mb": 8192, "ram_available_mb": 4096, "gpus": []}


def test_kv_cache_and_fit_estimates():
    show = {"model_info": {"general.architecture": "llama", "llama.block_count": 28, "llama.embedding_length": 3072,
                           "llama.attention.head_count": 24, "llama.attention.head_count_kv": 8}}
    kv = kv_cache_gb(show, 4096)
    assert kv == round(2 * 28 * 8 * 128 * 2 * 4096 / 1024**3, 2)  # 0.44 GB
    small = fit(1.3 * 1024**3, kv, 4096, SNAP)
    assert small["verdict"] == "fits" and small["kv_source"] == "model architecture"
    assert fit(6 * 1024**3, None, 4096, SNAP, params_b=8)["verdict"] in ("tight", "too_big")
    assert fit(20 * 1024**3, None, 4096, SNAP)["verdict"] == "too_big"
    assert fit(6 * 1024**3, None, 4096, {**SNAP, "gpus": [{"name": "GPU", "memory_total_mb": 12288}]})["where"] == "GPU"
    assert kv_cache_gb({}, 4096) is None


def test_setup_check_reports_prerequisites(client):
    r = client.get("/api/setup/check").json()
    ids = {c["id"]: c for c in r["checks"]}
    assert ids["ollama"]["status"] == "ok" and "simulator" in ids["ollama"]["detail"]
    assert ids["chat_model"]["status"] == "ok" and ids["database"]["status"] == "ok"
    assert ids["backup"]["status"] == "warn"  # no backup yet
    client.post("/api/system/backups", json={})
    ids = {c["id"]: c for c in client.get("/api/setup/check").json()["checks"]}
    assert ids["backup"]["status"] == "ok"


def test_catalog_has_licences_and_flags_research_only_models(client):
    cat = client.get("/api/setup/models?num_ctx=4096").json()
    rows = {m["name"]: m for m in cat["catalog"]}
    assert rows["qwen2.5:3b"]["commercial"].startswith("no") and rows["phi4-mini"]["license"] == "MIT"
    assert all(m["license"] and m["fit"]["verdict"] in ("fits", "tight", "too_big") for m in cat["catalog"])
    sim = {m["name"]: m for m in cat["other_installed"]}
    assert sim["sim-medium:3b"]["fit"]["kv_source"] == "model architecture"


def test_pull_is_blocked_offline_and_streams_progress_online(client):
    r = client.post("/api/setup/models/pull", json={"name": "qwen3:0.6b"})
    assert r.status_code == 403 and r.json()["error"]["code"] == "offline_mode_blocked"
    assert client.patch("/api/settings", json={"offline_only": False}).status_code == 200
    r = client.post("/api/setup/models/pull", json={"name": "qwen3:0.6b"})
    events = [json.loads(x) for x in r.text.splitlines() if x.strip()]
    assert events[-1] == {"type": "done", "model": "qwen3:0.6b"}
    assert any(e.get("percent") == 50.0 for e in events)
    bad = [json.loads(x) for x in client.post("/api/setup/models/pull", json={"name": "missing-model"}).text.splitlines()]
    assert bad[-1]["type"] == "error"
    assert client.post("/api/setup/models/pull", json={"name": "../../etc"}).status_code == 422
    assert client.delete("/api/setup/models/sim-small:1b").status_code == 204
    assert client.delete("/api/setup/models/not-there:1b").status_code == 404
    audit = client.get("/api/system/audit").json()["events"]
    assert {"model.pull", "model.delete"} <= {e["action"] for e in audit}
