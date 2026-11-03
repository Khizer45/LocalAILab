from __future__ import annotations

from conftest import make_pdf, stream_events

POLICY_TXT = (
    "Leave policy of Northwind Clinic.\n\n"
    "Employees receive 24 days of paid annual leave per year. Unused leave can be carried over for three months.\n\n"
    "Sick leave requires a medical certificate after two consecutive days."
)


def _upload(client, name: str, data: bytes, ctype="text/plain"):
    return client.post("/api/documents", files={"file": (name, data, ctype)})


def test_upload_txt_index_and_cited_answer(client):
    r = _upload(client, "leave_policy.txt", POLICY_TXT.encode())
    assert r.status_code == 201, r.text
    doc = r.json()
    assert doc["status"] == "ready" and doc["chunk_count"] >= 1
    assert doc["embedding_backend"] == "ollama" and doc["embedding_model"] == "sim-embed:latest"

    events = stream_events(client.post("/api/documents/ask", json={"question": "How many days of paid annual leave do employees get?"}))
    src = next(e for e in events if e["type"] == "sources")
    assert src["sources"][0]["filename"] == "leave_policy.txt"
    answer = "".join(e["content"] for e in events if e["type"] == "token")
    assert "24 days" in answer and "[S1]" in answer
    assert events[-1]["type"] == "done"


def test_pdf_page_numbers_are_cited(client, tmp_path):
    pdf = make_pdf(tmp_path / "handbook.pdf", [
        "Chapter one introduces the security team and the office locations in Mirpur and Lahore.",
        "Passwords must be rotated every ninety days and must contain at least twelve characters.",
    ])
    r = _upload(client, "handbook.pdf", pdf.read_bytes(), "application/pdf")
    assert r.status_code == 201 and r.json()["page_count"] == 2
    events = stream_events(client.post("/api/documents/ask", json={"question": "How often must passwords be rotated?"}))
    sources = next(e for e in events if e["type"] == "sources")["sources"]
    assert sources[0]["page"] == 2
    detail = client.get(f"/api/documents/{r.json()['id']}").json()
    assert [c["page"] for c in detail["chunks"]] == [1, 2]


def test_answer_absent_from_documents(client):
    _upload(client, "leave_policy.txt", POLICY_TXT.encode())
    events = stream_events(client.post("/api/documents/ask", json={"question": "What is the recipe for biryani with saffron?"}))
    answer = "".join(e["content"] for e in events if e["type"] == "token")
    assert answer == "I could not find this in the indexed documents."
    assert next(e for e in events if e["type"] == "sources")["sources"] == []


def test_chat_with_documents_toggle(client):
    _upload(client, "leave_policy.txt", POLICY_TXT.encode())
    conv = client.post("/api/chat/conversations", json={"use_documents": True}).json()
    events = stream_events(client.post(f"/api/chat/conversations/{conv['id']}/messages",
                                       json={"content": "Is a medical certificate needed for sick leave?"}))
    assert any(e["type"] == "sources" and e["sources"] for e in events)
    msg = client.get(f"/api/chat/conversations/{conv['id']}").json()["messages"][-1]
    assert msg["sources"][0]["filename"] == "leave_policy.txt"


def test_validation_errors(client):
    assert _upload(client, "run.exe", b"MZ....").json()["error"]["code"] == "unsupported_file_type"
    assert _upload(client, "fake.pdf", b"hello not a pdf").json()["error"]["code"] == "invalid_file_content"
    assert _upload(client, "bin.txt", b"abc\x00\x01\x02").json()["error"]["code"] == "invalid_file_content"
    assert _upload(client, "empty.txt", b"").json()["error"]["code"] == "empty_file"
    client.patch("/api/settings", json={"max_upload_mb": 1})
    r = _upload(client, "big.txt", b"a " * (600 * 1024))
    assert r.status_code == 422 and r.json()["error"]["code"] == "file_too_large"


def test_duplicate_upload_rejected(client):
    assert _upload(client, "a.txt", POLICY_TXT.encode()).status_code == 201
    r = _upload(client, "copy.txt", POLICY_TXT.encode())
    assert r.status_code == 409


def test_malformed_and_scanned_pdf_marked_failed(client, tmp_path):
    r = _upload(client, "broken.pdf", b"%PDF-1.7\n this is not really a pdf")
    assert r.status_code == 201 and r.json()["status"] == "failed"
    blank = make_pdf(tmp_path / "blank.pdf", [""])
    r = _upload(client, "scan.pdf", blank.read_bytes(), "application/pdf")
    assert r.json()["status"] == "failed" and "OCR" in r.json()["error"]


def test_delete_removes_file_chunks_and_index(make_app):
    c = make_app()
    settings = c.app.state.ctx.settings
    doc = _upload(c, "leave_policy.txt", POLICY_TXT.encode()).json()
    files = list((settings.uploads_dir / "1").glob("*"))
    assert len(files) == 1
    assert c.delete(f"/api/documents/{doc['id']}").status_code == 204
    assert list((settings.uploads_dir / "1").glob("*")) == []
    with c.app.state.ctx.SessionLocal() as db:
        from app.models.document import Chunk

        assert db.query(Chunk).count() == 0
    events = stream_events(c.post("/api/documents/ask", json={"question": "paid annual leave days"}))
    assert next(e for e in events if e["type"] == "sources")["sources"] == []


def test_hashing_fallback_and_reindex(make_app):
    c = make_app(embedding_model="not-installed-embedder")
    doc = _upload(c, "leave_policy.txt", POLICY_TXT.encode()).json()
    assert doc["embedding_backend"] == "hashing"
    events = stream_events(c.post("/api/documents/ask", json={"question": "paid annual leave days per year"}))
    src = next(e for e in events if e["type"] == "sources")
    assert src["sources"] and any("hashing" in w for w in src["warnings"])
    # switch to the neural embedder → document must be flagged for re-indexing
    c.patch("/api/settings", json={"embedding_model": "sim-embed"})
    listed = c.get("/api/documents").json()["documents"][0]
    assert listed["needs_reindex"] is True
    r = c.post(f"/api/documents/{doc['id']}/reindex")
    assert r.status_code == 200
    assert c.get("/api/documents").json()["documents"][0]["embedding_backend"] == "ollama"


def test_utf16_and_markdown(client):
    r = _upload(client, "notes.md", "# Notes\n\nThe backup server lives in rack seven.".encode("utf-16"))
    assert r.status_code == 201 and r.json()["status"] == "ready"
