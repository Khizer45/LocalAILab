"""API tokens with scopes, revocation, sign-in rate limiting, password change, remote-access guard."""

from __future__ import annotations


def _session(c, user, password="password123"):
    c.post("/api/auth/register", json={"username": user, "password": password})
    r = c.post("/api/auth/login", json={"username": user, "password": password})
    return {"Authorization": f"Bearer {r.json()['token']}"}


def test_scoped_api_tokens(make_app):
    c = make_app(auth_enabled=True)
    admin = _session(c, "admin9")
    r = c.post("/api/auth/tokens", headers=admin, json={"name": "ci", "scopes": ["read", "evaluate"]})
    assert r.status_code == 201
    tok = r.json()
    assert tok["token"].startswith("lal_") and tok["prefix"] == tok["token"][:12]
    h = {"Authorization": f"Bearer {tok['token']}"}
    assert c.get("/api/benchmarks/runs", headers=h).status_code == 200  # read
    assert c.post("/api/benchmarks/runs", headers=h, json={"suite_id": "quick", "models": ["sim-small:1b"],
                                                           "max_cases": 1}).status_code == 201  # evaluate
    denied = c.post("/api/chat/conversations", headers=h, json={})
    assert denied.status_code == 403 and "'chat' scope" in denied.json()["error"]["message"]
    assert c.patch("/api/settings", headers=h, json={"retrieval_top_k": 3}).status_code == 403  # admin user, no scope
    assert c.post("/api/auth/tokens", headers=h, json={"name": "x", "scopes": ["read"]}).status_code == 403
    listed = c.get("/api/auth/tokens", headers=admin).json()["tokens"]
    assert listed[0]["name"] == "ci" and listed[0]["last_used_at"] and "token" not in listed[0]
    assert c.delete(f"/api/auth/tokens/{tok['id']}", headers=admin).status_code == 204
    assert c.get("/api/benchmarks/runs", headers=h).status_code == 401  # revoked
    events = {e["action"] for e in c.get("/api/system/audit", headers=admin).json()["events"]}
    assert {"token.create", "token.revoke", "auth.login"} <= events


def test_non_admin_cannot_mint_admin_tokens_and_unknown_scopes_fail(make_app):
    c = make_app(auth_enabled=True)
    _session(c, "first1")
    user = _session(c, "second1")
    assert c.post("/api/auth/tokens", headers=user, json={"name": "a", "scopes": ["admin"]}).status_code == 403
    assert c.post("/api/auth/tokens", headers=user, json={"name": "a", "scopes": ["root"]}).status_code == 422


def test_login_rate_limit(make_app):
    c = make_app(auth_enabled=True)
    _session(c, "dana1")
    for _ in range(5):
        assert c.post("/api/auth/login", json={"username": "dana1", "password": "wrong-pass"}).status_code == 401
    r = c.post("/api/auth/login", json={"username": "dana1", "password": "password123"})
    assert r.status_code == 429 and r.json()["error"]["code"] == "too_many_attempts"  # even the right password waits


def test_password_change_revokes_other_sessions(make_app):
    c = make_app(auth_enabled=True)
    a = _session(c, "erin1")
    b = {"Authorization": "Bearer " + c.post("/api/auth/login", json={"username": "erin1",
                                                                     "password": "password123"}).json()["token"]}
    assert c.post("/api/auth/password", headers=a, json={"current_password": "nope", "new_password": "newpassword1"}).status_code == 401
    assert c.post("/api/auth/password", headers=a, json={"current_password": "password123",
                                                          "new_password": "newpassword1"}).status_code == 204
    assert c.get("/api/auth/me", headers=a).status_code == 200 and c.get("/api/auth/me", headers=b).status_code == 401
    assert c.post("/api/auth/login", json={"username": "erin1", "password": "newpassword1"}).status_code == 200
    assert c.post("/api/auth/sessions/revoke-all", headers=a).json()["revoked"] >= 1


def test_remote_clients_are_refused_without_authentication(make_app):
    c = make_app()
    c.app.state.ctx.settings  # local test client is allowed
    assert c.get("/api/models").status_code == 200
    from fastapi.testclient import TestClient

    remote = TestClient(c.app, client=("203.0.113.7", 50000))
    r = remote.get("/api/models")
    assert r.status_code == 403 and r.json()["error"]["code"] == "remote_access_requires_auth"
    assert remote.get("/api/health").status_code == 200  # liveness stays reachable for monitoring
    lan = make_app(unauthenticated_networks="127.0.0.0/8,10.0.0.0/8")
    assert TestClient(lan.app, client=("10.1.2.3", 1)).get("/api/models").status_code == 200
    authed = make_app(auth_enabled=True)
    assert TestClient(authed.app, client=("203.0.113.7", 1)).get("/api/models").status_code == 401  # login needed


def test_dns_rebinding_and_cross_site_requests_are_refused_without_auth(client):
    r = client.get("/api/models", headers={"host": "evil.example:8000"})
    assert r.status_code == 403 and r.json()["error"]["code"] == "host_not_allowed"
    assert client.get("/api/models", headers={"host": "127.0.0.1:8000"}).status_code == 200
    assert client.get("/api/models", headers={"host": "[::1]:8000"}).status_code == 200
    r = client.post("/api/documents", headers={"origin": "https://evil.example"},
                    files={"file": ("x.txt", b"injected")})
    assert r.status_code == 403 and r.json()["error"]["code"] == "cross_site_request"
    ok = client.post("/api/documents", headers={"origin": "http://127.0.0.1:8000"}, files={"file": ("x.txt", b"fine")})
    assert ok.status_code == 201
    assert client.get("/api/health", headers={"host": "anything"}).status_code == 200
