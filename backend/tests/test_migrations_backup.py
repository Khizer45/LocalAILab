"""Versioned schema migrations, upgrade from v1.1 databases, backup / verify / restore."""

from __future__ import annotations

import io
import json
import sqlite3
import zipfile

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, inspect, text

from app.core import backup as B
from app.core.errors import ValidationFailed
from app.db.base import Base
from app.db.migrate import alembic_config, current_revision, head_revision, upgrade_database


def _engine(path):
    return create_engine(f"sqlite:///{path.as_posix()}")


def test_models_and_migrations_are_in_sync(tmp_path):
    from app import models  # noqa: F401

    eng = _engine(tmp_path / "fresh.db")
    upgrade_database(eng, tmp_path / "backups")
    with eng.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn, opts={"render_as_batch": True}), Base.metadata)
    assert diff == [], f"run `alembic revision --autogenerate`: {diff}"
    assert current_revision(eng) == head_revision()


def test_v11_database_is_upgraded_in_place_with_a_backup_first(tmp_path):
    from alembic import command

    db = tmp_path / "legacy.db"
    eng = _engine(db)
    with eng.begin() as conn:  # build exactly the v1.1 schema, then drop alembic's bookkeeping = a create_all() DB
        cfg = alembic_config()
        cfg.attributes["connection"] = conn
        command.upgrade(cfg, "0001")
        conn.execute(text("DROP TABLE alembic_version"))
        conn.execute(text("INSERT INTO users (username, is_admin, created_at) VALUES ('local', 1, '2026-01-01')"))
        conn.execute(text("INSERT INTO arena_runs (owner_id, name, dataset_info, config, contenders, status, "
                          "progress_done, progress_total, created_at) VALUES (1, 'old run', '{}', '{}', '[]', "
                          "'completed', 1, 1, '2026-01-01')"))
    assert current_revision(eng) is None
    result = upgrade_database(eng, tmp_path / "backups")
    assert result["from"] is None and result["to"] == head_revision() and result["backup"]
    backup_db = sqlite3.connect(result["backup"])
    assert backup_db.execute("SELECT name FROM arena_runs").fetchone() == ("old run",)
    backup_db.close()
    with eng.connect() as conn:
        row = conn.execute(text("SELECT name, protocol FROM arena_runs")).one()
    assert row == ("old run", "v1")  # existing runs are labelled as the legacy protocol
    cols = {c["name"] for c in inspect(eng).get_columns("arena_datasets")}
    assert {"n_val", "split_sha256", "group_column"} <= cols
    assert upgrade_database(eng, tmp_path / "backups")["backup"] is None  # already current: nothing to do


def test_database_from_a_newer_version_is_refused(tmp_path):
    eng = _engine(tmp_path / "new.db")
    upgrade_database(eng, None)
    with eng.begin() as conn:
        conn.execute(text("UPDATE alembic_version SET version_num = '9999'"))
    with pytest.raises(RuntimeError, match="newer version"):
        upgrade_database(eng, None)


def _seed(client):
    client.post("/api/chat/conversations", json={"title": "keep me"})
    client.post("/api/documents", files={"file": ("note.txt", b"The vault code is 4242.")})


def test_backup_verify_restore_round_trip(make_app, tmp_path):
    c = make_app()
    _seed(c)
    created = c.post("/api/system/backups", json={"label": "test"})
    assert created.status_code == 201, created.text
    info = created.json()
    assert info["ok"] and info["has_database"] and info["files"] >= 2
    settings = c.app.state.ctx.settings
    zip_path = settings.backups_dir / info["name"]
    listed = c.get("/api/system/backups").json()["backups"]
    assert any(b["name"] == info["name"] for b in listed)
    blob = c.get(f"/api/system/backups/{info['name']}/download")
    assert blob.status_code == 200 and zipfile.ZipFile(io.BytesIO(blob.content)).testzip() is None

    # restore into a fresh data folder (the CLI path: app stopped)
    target = tmp_path / "restored"
    target.mkdir()
    (target / "local_ai_lab.db").write_bytes(b"old database")
    res = B.restore_backup(zip_path, target)
    assert (target / "local_ai_lab.db").exists() and res["previous_data_moved_to"]
    con = sqlite3.connect(target / "local_ai_lab.db")
    assert con.execute("SELECT title FROM conversations").fetchone() == ("keep me",)
    con.close()
    assert any((target / "uploads").rglob("*"))
    moved = next((target / "backups").glob("replaced-*"))
    assert (moved / "local_ai_lab.db").read_bytes() == b"old database"  # nothing is deleted by a restore


def test_staged_restore_is_applied_at_next_start(make_app, tmp_path):
    from fastapi.testclient import TestClient

    from app.core.config import Settings
    from app.main import create_app

    c = make_app()
    _seed(c)
    name = c.post("/api/system/backups", json={}).json()["name"]
    c.post("/api/chat/conversations", json={"title": "made after the backup"})
    r = c.post(f"/api/system/backups/{name}/restore")
    assert r.status_code == 200 and r.json()["staged"]
    assert c.get("/api/system/backups").json()["pending_restore"]["source"] == name
    s = c.app.state.ctx.settings
    c.__exit__(None, None, None)
    c.app.state.ctx.engine.dispose()
    with TestClient(create_app(Settings(_env_file=None, **{k: getattr(s, k) for k in (
            "app_env", "data_dir", "ollama_base_url", "doc_processing_mode", "mlflow_enabled", "embedding_model",
            "log_json", "log_level", "default_model")}))) as c2:
        titles = [x["title"] for x in c2.get("/api/chat/conversations").json()["conversations"]]
        assert "keep me" in titles and "made after the backup" not in titles
        assert c2.get("/api/system/backups").json()["pending_restore"] is None


def test_tampered_and_malicious_backups_are_rejected(make_app, tmp_path):
    c = make_app()
    _seed(c)
    name = c.post("/api/system/backups", json={}).json()["name"]
    src = c.app.state.ctx.settings.backups_dir / name
    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(tampered, "w") as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename.startswith("db/"):
                data = data[:-10] + b"X" * 10
            zout.writestr(item, data)
    with pytest.raises(ValidationFailed, match="damaged or was modified"):
        B.verify_backup(tampered)
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as z:
        z.writestr("manifest.json", json.dumps({"format": B.FORMAT, "format_version": 1, "files": [
            {"path": "../../escape.txt", "size": 1, "sha256": "x"}]}))
        z.writestr("../../escape.txt", "x")
    with pytest.raises(ValidationFailed, match="unsafe path"):
        B.verify_backup(evil)
    r = c.post("/api/system/backups/restore-upload", files={"file": ("evil.zip", evil.read_bytes())})
    assert r.status_code == 422


def test_backup_endpoints_require_an_administrator(make_app):
    c = make_app(auth_enabled=True)
    c.post("/api/auth/register", json={"username": "boss", "password": "password123"})  # first user = admin
    c.post("/api/auth/register", json={"username": "worker", "password": "password123"})
    tok = c.post("/api/auth/login", json={"username": "worker", "password": "password123"}).json()["token"]
    h = {"Authorization": f"Bearer {tok}"}
    assert c.get("/api/system/backups", headers=h).status_code == 403
    assert c.post("/api/system/backups", headers=h, json={}).status_code == 403
