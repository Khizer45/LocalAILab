"""Retrieval-quality evaluation (Recall@k, MRR, nDCG@k, abstention) of the real document search."""

from __future__ import annotations

import json
from pathlib import Path

SAMPLES = Path(__file__).resolve().parents[2] / "samples"


def _upload(c, name: str, data: bytes):
    r = c.post("/api/documents", files={"file": (name, data)})
    assert r.status_code == 201, r.text
    return r.json()


def test_metrics_on_a_known_ranking(client):
    _upload(client, "rack.txt", b"The backup server lives in rack seven of the basement data room.")
    _upload(client, "menu.txt", b"The canteen serves lentil soup on Tuesdays and rice on Fridays.")
    body = {"questions": [
        {"id": "a", "question": "The backup server lives in rack seven of the basement data room.",
         "relevant": [{"filename": "rack.txt", "contains": "rack seven"}]},
        {"id": "b", "question": "The backup server lives in rack seven of the basement data room.",
         "relevant": [{"filename": "menu.txt", "contains": "lentil"}]},
        {"id": "u", "question": "zzzz qqqq", "relevant": []}], "k_values": [1, 2]}
    r = client.post("/api/documents/retrieval-eval", json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    rows = {q["id"]: q for q in out["questions"]}
    assert rows["a"]["first_relevant_rank"] == 1 and rows["a"]["recall@1"] == 1.0 and rows["a"]["ndcg@1"] == 1.0
    assert rows["b"]["first_relevant_rank"] == 2 and rows["b"]["recall@1"] == 0.0 and rows["b"]["recall@2"] == 1.0
    assert abs(rows["b"]["ndcg@2"] - 1 / 1.5849625) < 1e-4
    s = out["summary"]
    assert s["mrr"] == 0.75 and s["recall@1"] == 0.5 and s["answerable"] == 2 and s["unanswerable"] == 1
    assert "false_retrieval_rate" in s and isinstance(rows["u"]["false_retrieval"], bool)


def test_sample_question_set_runs_end_to_end(client):
    for p in (SAMPLES / "documents").iterdir():
        _upload(client, p.name, p.read_bytes())
    qa = json.loads((SAMPLES / "retrieval" / "sample_qa.json").read_text())
    out = client.post("/api/documents/retrieval-eval", json={"questions": qa["questions"]}).json()
    s = out["summary"]
    assert s["answerable"] == 10 and s["unanswerable"] == 2
    for k in (1, 3, 5, 10):
        assert 0 <= s[f"recall@{k}"] <= 1 and 0 <= s[f"ndcg@{k}"] <= 1
    assert s["recall@10"] >= s["recall@1"]


def test_other_users_documents_are_never_searched(make_app):
    c = make_app(auth_enabled=True)

    def login(u):
        c.post("/api/auth/register", json={"username": u, "password": "password123"})
        tok = c.post("/api/auth/login", json={"username": u, "password": "password123"}).json()["token"]
        return {"Authorization": "Bearer " + tok}

    alice, bob = login("alice2"), login("bob2")
    assert c.post("/api/documents", headers=alice, files={"file": ("secret.txt", b"Alice salary is 9000.")}).status_code == 201
    out = c.post("/api/documents/retrieval-eval", headers=bob, json={"questions": [
        {"id": "x", "question": "Alice salary is 9000.", "relevant": [{"contains": "salary"}]}]}).json()
    assert out["questions"][0]["retrieved"] == [] and out["summary"]["mrr"] == 0.0
