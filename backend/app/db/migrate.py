"""Versioned schema migrations (Alembic) with an automatic pre-migration backup.

At start-up ``upgrade_database`` brings the database to the newest revision:

* empty database                 → all migrations run (fresh install);
* database created by v1.0/v1.1  → no ``alembic_version`` table yet; revision
  0001 creates only the tables that are missing, later revisions add the v1.2
  changes;
* database already at a revision → only newer revisions run.

Before an existing SQLite database is changed, a consistent copy is written to
``data/backups/`` with SQLite's online-backup API, so a failed or unwanted
upgrade can be undone by restoring that file.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import inspect
from sqlalchemy.engine import Engine

log = logging.getLogger("app.db")

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def alembic_config() -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    return cfg


def head_revision() -> str:
    head = ScriptDirectory.from_config(alembic_config()).get_current_head()
    assert head is not None
    return head


def current_revision(engine: Engine) -> str | None:
    with engine.connect() as conn:
        return MigrationContext.configure(conn).get_current_revision()


def known_revisions() -> list[str]:
    return [s.revision for s in ScriptDirectory.from_config(alembic_config()).walk_revisions()]


def _sqlite_path(engine: Engine) -> Path | None:
    if engine.url.get_backend_name() != "sqlite":
        return None
    db = engine.url.database
    if not db or db == ":memory:":
        return None
    return Path(db)


def sqlite_snapshot(src: Path, dest: Path) -> Path:
    """Copy a live SQLite database consistently (works while it is open in WAL mode)."""
    import sqlite3
    from contextlib import closing

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.unlink(missing_ok=True)
    with closing(sqlite3.connect(str(src), timeout=30)) as s, \
            closing(sqlite3.connect(tmp)) as d:
        s.backup(d)
    tmp.replace(dest)
    return dest


def upgrade_database(engine: Engine, backup_dir: Path | None) -> dict:
    """Upgrade to the newest schema. Returns {"from", "to", "backup"}."""
    head = head_revision()
    before = current_revision(engine)
    if before == head:
        return {"from": before, "to": head, "backup": None}
    if before is not None and before not in known_revisions():
        raise RuntimeError(f"The database was created by a newer version of Local AI Lab (schema {before}); "
                           f"this version only knows up to {head}. Install the newer version or restore a backup.")
    has_tables = inspect(engine).has_table("users")
    backup = None
    src = _sqlite_path(engine)
    if has_tables and src is not None and src.exists() and backup_dir is not None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = sqlite_snapshot(src, backup_dir / f"pre-migration-{before or 'legacy'}-to-{head}-{stamp}.db")
        log.info("database backed up before migration", extra={"backup": str(backup)})
    cfg = alembic_config()
    with engine.connect() as conn:
        if engine.url.get_backend_name() == "sqlite":
            # table rebuilds (batch mode) must not trigger ON DELETE CASCADE
            conn.exec_driver_sql("PRAGMA foreign_keys=OFF")
            conn.commit()
        with conn.begin():
            cfg.attributes["connection"] = conn
            command.upgrade(cfg, "head")
        if engine.url.get_backend_name() == "sqlite":
            conn.exec_driver_sql("PRAGMA foreign_keys=ON")
            conn.commit()
    log.info("database migrated", extra={"from": before or "legacy/empty", "to": head})
    return {"from": before, "to": head, "backup": str(backup) if backup else None}
