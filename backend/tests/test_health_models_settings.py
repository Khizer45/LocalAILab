from __future__ import annotations

from app.core.network_guard import guard


def test_liveness(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert "X-Request-ID" in r.headers


def test_readiness_with_runtime(client):
    body = client.get("/api/health/ready").json()
    assert body["status"] == "ok"
    assert body["checks"]["database"]["ok"]
    assert body["checks"]["ollama"]["version"] == "0.0.0-simulator"
    assert body["checks"]["embeddings"]["backend"] == "ollama"


def test_readiness_degraded_when_runtime_down(make_app):
    c = make_app(ollama_base_url="http://127.0.0.1:9")  # nothing listens on port 9
    body = c.get("/api/health/ready").json()
    assert body["status"] == "degraded"
    assert body["checks"]["ollama"]["ok"] is False
    assert "ollama serve" in body["checks"]["ollama"]["hint"]
    # model listing degrades gracefully instead of failing
    models = c.get("/api/models").json()
    assert models["models"] == [] and models["status"]["local"]["reachable"] is False


def test_model_listing_classifies_models(client):
    data = client.get("/api/models").json()
    by_id = {m["id"]: m for m in data["models"]}
    assert by_id["sim-small:1b"]["provider"] == "local" and by_id["sim-small:1b"]["allowed"]
    assert by_id["sim-small:1b"]["parameter_size"] == "1.2B"
    assert by_id["sim-embed:latest"]["is_embedding"] is True
    cloud = by_id["sim-cloud:cloud"]
    assert cloud["provider"] == "cloud" and cloud["allowed"] is False
    assert "Offline-Only" in cloud["blocked_reason"]


def test_system_resources_and_recommendations(client):
    res = client.get("/api/system/resources").json()
    assert res["ram_total_mb"] > 0 and res["cpu_count_logical"] >= 1
    rec = client.get("/api/system/recommendations").json()
    assert rec["chat_models"] and rec["commands"][0].startswith("ollama pull")


def test_settings_roundtrip_and_validation(client):
    s = client.get("/api/settings").json()
    assert s["runtime"]["offline_only"] is True
    assert s["static"]["ollama_cloud_api_key_set"] is False

    r = client.patch("/api/settings", json={"max_upload_mb": 0})
    assert r.status_code == 422

    r = client.patch("/api/settings", json={"retrieval_top_k": 6, "offline_only": False})
    assert r.status_code == 200
    assert r.json()["runtime"]["retrieval_top_k"] == 6
    assert guard.offline_only is False
    client.patch("/api/settings", json={"offline_only": True})
    assert guard.offline_only is True


def test_allowed_models_list_is_enforced(client):
    client.patch("/api/settings", json={"allowed_models": ["sim-medium:3b"]})
    models = {m["id"]: m for m in client.get("/api/models").json()["models"]}
    assert models["sim-small:1b"]["allowed"] is False
    conv = client.post("/api/chat/conversations", json={}).json()
    r = client.post(f"/api/chat/conversations/{conv['id']}/messages", json={"content": "hi", "model": "sim-small:1b"})
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "model_not_allowed"


def test_privacy_status_endpoint(client):
    data = client.get("/api/privacy/status").json()
    checks = {c["id"]: c for c in data["checks"]}
    assert checks["offline_mode"]["ok"] and checks["socket_guard"]["ok"]
    assert checks["telemetry"]["ok"]
    assert data["limitations"]


def test_security_headers(client):
    ui = client.get("/")
    assert "default-src 'self'" in ui.headers["content-security-policy"]
    api = client.get("/api/health")
    assert api.headers["x-content-type-options"] == "nosniff" and api.headers["referrer-policy"] == "no-referrer"
