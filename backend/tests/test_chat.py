from __future__ import annotations

from conftest import stream_events


def _conv(client, **body):
    r = client.post("/api/chat/conversations", json=body)
    assert r.status_code == 201
    return r.json()


def test_streaming_chat_persists_history_and_metrics(client):
    conv = _conv(client, model="sim-small:1b")
    events = stream_events(client.post(f"/api/chat/conversations/{conv['id']}/messages",
                                       json={"content": "Hello there, what can you do?"}))
    types = [e["type"] for e in events]
    assert types[0] == "start" and types[-1] == "done"
    assert types.count("token") > 3
    text = "".join(e["content"] for e in events if e["type"] == "token")
    assert "Simulated reply" in text
    done = events[-1]
    assert done["metrics"]["tokens_source"] == "runtime"
    assert done["metrics"]["output_tokens"] > 0 and done["metrics"]["ttft_ms"] is not None

    full = client.get(f"/api/chat/conversations/{conv['id']}").json()
    assert [m["role"] for m in full["messages"]] == ["user", "assistant"]
    assert full["messages"][1]["content"] == text
    assert full["title"].startswith("Hello there")


def test_options_are_clamped_to_resource_limits(client):
    client.patch("/api/settings", json={"max_output_tokens": 16})
    conv = _conv(client)
    events = stream_events(client.post(f"/api/chat/conversations/{conv['id']}/messages",
                                       json={"content": "Write a long story", "options": {"num_predict": 5000}}))
    assert events[-1]["metrics"]["output_tokens"] <= 16


def test_thinking_stream_is_separated(client):
    conv = _conv(client, model="sim-medium:3b")
    events = stream_events(client.post(f"/api/chat/conversations/{conv['id']}/messages",
                                       json={"content": "Think first", "think": True}))
    assert any(e["type"] == "thinking" for e in events)
    msg = client.get(f"/api/chat/conversations/{conv['id']}").json()["messages"][-1]
    assert msg["thinking"] and "Let me" in msg["thinking"]


def test_unknown_model_gives_readable_error(client):
    conv = _conv(client)
    r = client.post(f"/api/chat/conversations/{conv['id']}/messages", json={"content": "hi", "model": "llama9:70b"})
    assert r.status_code == 404
    err = r.json()["error"]
    assert err["code"] == "model_not_available" and "ollama pull llama9:70b" in err["hint"]


def test_embedding_model_cannot_chat(client):
    conv = _conv(client)
    r = client.post(f"/api/chat/conversations/{conv['id']}/messages", json={"content": "hi", "model": "sim-embed:latest"})
    assert r.status_code == 422


def test_cloud_model_blocked_in_offline_mode(client):
    conv = _conv(client)
    r = client.post(f"/api/chat/conversations/{conv['id']}/messages", json={"content": "hi", "model": "sim-cloud:cloud"})
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "offline_mode_blocked"


def test_runtime_down_returns_503(make_app):
    c = make_app(ollama_base_url="http://127.0.0.1:9")
    conv = c.post("/api/chat/conversations", json={}).json()
    r = c.post(f"/api/chat/conversations/{conv['id']}/messages", json={"content": "hi"})
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "runtime_unavailable"


def test_conversation_update_clear_and_delete(client):
    conv = _conv(client)
    stream_events(client.post(f"/api/chat/conversations/{conv['id']}/messages", json={"content": "one"}))
    r = client.patch(f"/api/chat/conversations/{conv['id']}",
                     json={"title": "Renamed", "system_prompt": "Be brief.", "options": {"temperature": 0.2}})
    assert r.json()["title"] == "Renamed" and r.json()["options"] == {"temperature": 0.2}
    assert client.delete(f"/api/chat/conversations/{conv['id']}/messages").status_code == 204
    assert client.get(f"/api/chat/conversations/{conv['id']}").json()["messages"] == []
    assert client.delete(f"/api/chat/conversations/{conv['id']}").status_code == 204
    assert client.get(f"/api/chat/conversations/{conv['id']}").status_code == 404


def test_history_is_sent_to_model(client):
    conv = _conv(client)
    stream_events(client.post(f"/api/chat/conversations/{conv['id']}/messages", json={"content": "first message"}))
    events = stream_events(client.post(f"/api/chat/conversations/{conv['id']}/messages", json={"content": "second"}))
    # prompt token count grows because history (system + 3 messages) is included
    assert events[-1]["metrics"]["prompt_tokens"] > 30
