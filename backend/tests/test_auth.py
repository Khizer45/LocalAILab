from __future__ import annotations

from conftest import stream_events, wait_for


def _login(c, user):
    c.post("/api/auth/register", json={"username": user, "password": "password123"})
    token = c.post("/api/auth/login", json={"username": user, "password": "password123"}).json()["token"]
    return {"Authorization": f"Bearer {token}"}


def test_auth_required_when_enabled(make_app):
    c = make_app(auth_enabled=True)
    assert c.get("/api/health").status_code == 200  # health stays public
    assert c.get("/api/auth/config").json()["auth_enabled"] is True
    for path in ("/api/chat/conversations", "/api/documents", "/api/benchmarks/runs", "/api/settings", "/api/models"):
        assert c.get(path).status_code == 401, path
    assert c.get("/api/chat/conversations", headers={"Authorization": "Bearer nonsense"}).status_code == 401


def test_first_user_is_admin_and_only_admin_changes_settings(make_app):
    c = make_app(auth_enabled=True)
    admin, user = _login(c, "admin1"), _login(c, "user1")
    assert c.get("/api/auth/me", headers=admin).json()["is_admin"] is True
    assert c.get("/api/auth/me", headers=user).json()["is_admin"] is False
    assert c.patch("/api/settings", headers=user, json={"retrieval_top_k": 3}).status_code == 403
    assert c.patch("/api/settings", headers=admin, json={"retrieval_top_k": 3}).status_code == 200


def test_bad_password_and_logout(make_app):
    c = make_app(auth_enabled=True)
    h = _login(c, "carol")
    assert c.post("/api/auth/login", json={"username": "carol", "password": "wrongpass1"}).status_code == 401
    assert c.post("/api/auth/logout", headers=h).status_code == 204
    assert c.get("/api/auth/me", headers=h).status_code == 401


def test_users_cannot_access_each_others_data(make_app):
    c = make_app(auth_enabled=True)
    a, b = _login(c, "alice"), _login(c, "bob")
    conv = c.post("/api/chat/conversations", headers=a, json={}).json()
    stream_events(c.post(f"/api/chat/conversations/{conv['id']}/messages", headers=a, json={"content": "private"}))
    doc = c.post("/api/documents", headers=a, files={"file": ("a.txt", b"Alice notes about the project.", "text/plain")}).json()
    run = c.post("/api/benchmarks/runs", headers=a, json={"suite_id": "quick", "models": ["sim-small:1b"], "max_cases": 1}).json()
    wait_for(lambda: c.get(f"/api/benchmarks/runs/{run['id']}", headers=a).json()["status"] == "completed")

    assert c.get("/api/chat/conversations", headers=b).json()["conversations"] == []
    assert c.get(f"/api/chat/conversations/{conv['id']}", headers=b).status_code == 404
    assert c.delete(f"/api/chat/conversations/{conv['id']}", headers=b).status_code == 404
    assert c.post(f"/api/chat/conversations/{conv['id']}/messages", headers=b, json={"content": "x"}).status_code == 404
    assert c.get("/api/documents", headers=b).json()["documents"] == []
    assert c.get(f"/api/documents/{doc['id']}", headers=b).status_code == 404
    assert c.delete(f"/api/documents/{doc['id']}", headers=b).status_code == 404
    assert c.get("/api/benchmarks/runs", headers=b).json()["runs"] == []
    assert c.get(f"/api/benchmarks/runs/{run['id']}", headers=b).status_code == 404
    events = stream_events(c.post("/api/documents/ask", headers=b, json={"question": "Alice notes about the project"}))
    assert next(e for e in events if e["type"] == "sources")["sources"] == []
