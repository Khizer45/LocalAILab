"""OpenAI-compatible /v1 API, exercised with the official `openai` Python client."""

from __future__ import annotations

import pytest

openai = pytest.importorskip("openai")


def _client(c, key="not-needed"):
    return openai.OpenAI(base_url="http://testserver/v1", api_key=key, http_client=c, max_retries=0)


def test_models_chat_stream_and_embeddings(client):
    oc = _client(client)
    ids = [m.id for m in oc.models.list().data]
    assert "sim-small:1b" in ids
    r = oc.chat.completions.create(model="sim-medium:3b", messages=[{"role": "user", "content": "What is 17 multiplied by 6? Reply with just the number."}],
                                   temperature=0)
    assert r.choices[0].message.role == "assistant" and r.choices[0].message.content
    assert r.usage.completion_tokens > 0 and r.choices[0].finish_reason == "stop"
    chunks = list(oc.chat.completions.create(model="sim-medium:3b", stream=True,
                                             messages=[{"role": "user", "content": "Say hello"}]))
    text = "".join(ch.choices[0].delta.content or "" for ch in chunks if ch.choices)
    assert text and chunks[-1].choices[0].finish_reason == "stop"
    emb = oc.embeddings.create(model="sim-embed:latest", input=["a", "b"])
    assert len(emb.data) == 2 and len(emb.data[0].embedding) > 10


def test_tool_calls_round_trip(client):
    oc = _client(client)
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Weather for a city",
                                               "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                                              "required": ["city"]}}}]
    r = oc.chat.completions.create(model="sim-medium:3b", tools=tools,
                                   messages=[{"role": "user", "content": "What is the weather in Lahore?"}])
    call = r.choices[0].message.tool_calls[0]
    assert r.choices[0].finish_reason == "tool_calls" and call.function.name == "get_weather"
    follow = oc.chat.completions.create(model="sim-medium:3b", tools=tools, messages=[
        {"role": "user", "content": "What is the weather in Lahore?"},
        {"role": "assistant", "content": None, "tool_calls": [call.model_dump()]},
        {"role": "tool", "tool_call_id": call.id, "content": '{"temp_c": 31}'}])
    assert follow.choices[0].message.content


def test_policies_apply_to_the_compatible_api(make_app, fake_cloud_url):
    c = make_app(ollama_cloud_enabled=True, ollama_api_key="k", ollama_cloud_base_url=fake_cloud_url)
    oc = _client(c)
    with pytest.raises(openai.APIStatusError) as e:
        oc.chat.completions.create(model="cloud::sim-cloud", messages=[{"role": "user", "content": "hi"}])
    assert e.value.status_code in (403, 404)  # blocked in Offline-Only mode (or hidden)

    a = make_app(auth_enabled=True)
    a.post("/api/auth/register", json={"username": "ops1", "password": "password123"})
    session = a.post("/api/auth/login", json={"username": "ops1", "password": "password123"}).json()["token"]
    h = {"Authorization": f"Bearer {session}"}
    read_only = a.post("/api/auth/tokens", headers=h, json={"name": "r", "scopes": ["read"]}).json()["token"]
    chat = a.post("/api/auth/tokens", headers=h, json={"name": "n8n", "scopes": ["chat"]}).json()["token"]
    with pytest.raises(openai.AuthenticationError):
        _client(a, "wrong").models.list()
    with pytest.raises(openai.PermissionDeniedError):
        _client(a, read_only).chat.completions.create(model="sim-small:1b", messages=[{"role": "user", "content": "x"}])
    ok = _client(a, chat).chat.completions.create(model="sim-small:1b", messages=[{"role": "user", "content": "x"}])
    assert ok.choices[0].message.content


def test_document_grounding_extension(client):
    client.post("/api/documents", files={"file": ("vault.txt", b"The vault code for the lab is 4242.")})
    r = client.post("/v1/chat/completions", json={"model": "sim-medium:3b",
                                                  "messages": [{"role": "user", "content": "What is the vault code?"}],
                                                  "local_ai_lab": {"use_documents": True}})
    assert r.status_code == 200
    assert r.json()["local_ai_lab"]["sources"][0]["filename"] == "vault.txt"
