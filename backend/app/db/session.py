"""Engine / session factory and the database initialisation procedure."""

from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy import create_engine, event, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

log = logging.getLogger("app.db")

# The schema is versioned with Alembic (app/db/migrations). To change a table:
# edit the model, then `alembic -c alembic.ini revision --autogenerate -m "..."`
# and review the generated file (see docs/DEVELOPER_MANUAL.md, "Database").


def make_engine(url: str) -> Engine:
    is_sqlite = url.startswith("sqlite")
    engine = create_engine(
        url,
        connect_args={"check_same_thread": False, "timeout": 30} if is_sqlite else {},
        pool_pre_ping=True,
    )
    if is_sqlite:

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _):  # pragma: no cover - trivial
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA busy_timeout=30000")
            cur.close()

    return engine


def make_sessionmaker(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db(engine: Engine, SessionLocal: sessionmaker[Session], backup_dir: Path | None = None) -> dict:
    """Migrate the schema to the newest revision and ensure the local user."""
    from app import models  # noqa: F401  (register models)
    from app.db.migrate import upgrade_database
    from app.models.core import Meta, User

    migration = upgrade_database(engine, backup_dir)
    with SessionLocal() as db:
        meta = db.get(Meta, "schema_version")
        if meta is None:
            db.add(Meta(key="schema_version", value=migration["to"]))
        else:
            meta.value = migration["to"]
        if db.scalar(select(User).where(User.username == "local")) is None:
            db.add(User(username="local", password_hash=None, is_admin=True))
        db.commit()
    return migration
