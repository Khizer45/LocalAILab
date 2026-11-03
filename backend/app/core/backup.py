"""Backup and restore of everything Local AI Lab stores.

A backup is one ZIP file::

    manifest.json            app version, schema revision, creation time, every file with size + SHA-256
    db/local_ai_lab.db       consistent snapshot (SQLite online-backup API – safe while the app runs)
    db/mlflow.db             experiment tracking database (optional)
    files/uploads/...        uploaded documents
    files/arena/datasets/... Model Arena datasets (CSV / extracted images + split files)
    files/arena/models/...   uploaded models
    files/arena/runs/...     trained models kept by comparison runs
    files/ml/...             intent-baseline artifacts
    files/mlartifacts/...    MLflow artifacts (optional)
    files/reports/...

Not included: logs, older backups, caches that are rebuilt automatically (image feature cache, staging) and the
pretrained backbones (downloadable again; add ``include_backbones=True`` to keep them).

Restoring replaces the current data. It can run

* from the command line while the app is stopped (``python -m app.cli backup restore <zip>``), or
* from the UI/API: the backup is verified and *staged*; it is applied the next time the app starts, before the
  database is opened. The data it replaces is moved to ``data/backups/replaced-<time>/`` first, so a restore can
  itself be undone.

Every file is verified against the manifest before anything is replaced; archives with absolute paths, ``..``
components or unexpected entries are rejected.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath

from app import __version__
from app.core.errors import ValidationFailed

log = logging.getLogger("app.backup")

FORMAT = "local-ai-lab-backup"
FORMAT_VERSION = 1
DB_NAME = "local_ai_lab.db"
FILE_DIRS = ("uploads", "arena/datasets", "arena/models", "arena/runs", "ml", "reports")
OPTIONAL_DIRS = {"mlflow": ("mlartifacts",), "backbones": ("arena/backbones",)}
PENDING_DIR = "restore-pending"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _snapshot(src: Path, dest: Path) -> None:
    from app.db.migrate import sqlite_snapshot

    sqlite_snapshot(src, dest)


def create_backup(data_dir: Path, dest_dir: Path, *, schema_revision: str | None = None, include_mlflow: bool = True,
                  include_backbones: bool = False, label: str | None = None) -> Path:
    """Write a backup ZIP to ``dest_dir`` and return its path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_label = "".join(ch for ch in (label or "") if ch.isalnum() or ch in "-_")[:40]
    out = dest_dir / f"local-ai-lab-backup-{stamp}{('-' + safe_label) if safe_label else ''}.zip"
    tmp_dir = dest_dir / f".tmp-{stamp}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    entries: list[tuple[Path, str]] = []
    try:
        db = data_dir / DB_NAME
        if db.exists():
            _snapshot(db, tmp_dir / DB_NAME)
            entries.append((tmp_dir / DB_NAME, f"db/{DB_NAME}"))
        if include_mlflow and (data_dir / "mlflow.db").exists():
            _snapshot(data_dir / "mlflow.db", tmp_dir / "mlflow.db")
            entries.append((tmp_dir / "mlflow.db", "db/mlflow.db"))
        dirs = list(FILE_DIRS) + (list(OPTIONAL_DIRS["mlflow"]) if include_mlflow else []) + \
            (list(OPTIONAL_DIRS["backbones"]) if include_backbones else [])
        for rel in dirs:
            base = data_dir / rel
            if not base.exists():
                continue
            for p in sorted(base.rglob("*")):
                if p.is_file() and not p.is_symlink():
                    entries.append((p, f"files/{p.relative_to(data_dir).as_posix()}"))
        manifest = {"format": FORMAT, "format_version": FORMAT_VERSION, "app_version": __version__,
                    "schema_revision": schema_revision, "created_at": datetime.now().isoformat(timespec="seconds"),
                    "includes": {"mlflow": include_mlflow, "backbones": include_backbones},
                    "files": [{"path": arc, "size": src.stat().st_size, "sha256": _sha256(src)} for src, arc in entries]}
        part = out.with_suffix(".zip.part")
        with zipfile.ZipFile(part, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            zf.writestr("manifest.json", json.dumps(manifest, indent=1))
            for src, arc in entries:
                zf.write(src, arc)
        part.replace(out)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    log.info("backup created", extra={"path": str(out), "files": len(entries)})
    return out


def _safe_name(name: str) -> PurePosixPath:
    p = PurePosixPath(name.replace("\\", "/"))
    if p.is_absolute() or ".." in p.parts or (p.parts and ":" in p.parts[0]):
        raise ValidationFailed("The backup contains an unsafe path and was rejected.", code="unsafe_archive")
    return p


def verify_backup(path: Path) -> dict:
    """Check the archive structure and every file's SHA-256. Raises ValidationFailed when invalid."""
    try:
        zf = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError) as exc:
        raise ValidationFailed("The file is not a valid backup ZIP.") from exc
    with zf:
        names = zf.namelist()
        for n in names:
            _safe_name(n)
        if "manifest.json" not in names:
            raise ValidationFailed("The backup has no manifest.json.")
        manifest = json.loads(zf.read("manifest.json"))
        if manifest.get("format") != FORMAT:
            raise ValidationFailed("This ZIP is not a Local AI Lab backup.")
        if int(manifest.get("format_version", 0)) > FORMAT_VERSION:
            raise ValidationFailed("The backup was made by a newer version of Local AI Lab.")
        listed = {f["path"]: f for f in manifest.get("files", [])}
        extra = set(names) - set(listed) - {"manifest.json"}
        if extra:
            raise ValidationFailed(f"The backup contains files that are not in its manifest: {sorted(extra)[:3]}")
        problems = []
        total = 0
        for arc, meta in listed.items():
            p = _safe_name(arc)
            if p.parts[0] not in ("db", "files"):
                problems.append(f"unexpected location {arc}")
                continue
            if arc not in names:
                problems.append(f"missing {arc}")
                continue
            h = hashlib.sha256()
            size = 0
            with zf.open(arc) as f:
                for chunk in iter(lambda f=f: f.read(1024 * 1024), b""):
                    h.update(chunk)
                    size += len(chunk)
            total += size
            if h.hexdigest() != meta["sha256"] or size != meta["size"]:
                problems.append(f"checksum mismatch {arc}")
        if problems:
            raise ValidationFailed("The backup is damaged or was modified: " + "; ".join(problems[:5]),
                                   code="backup_corrupt")
    return {"ok": True, "app_version": manifest.get("app_version"), "schema_revision": manifest.get("schema_revision"),
            "created_at": manifest.get("created_at"), "files": len(listed), "bytes": total,
            "has_database": f"db/{DB_NAME}" in listed, "includes": manifest.get("includes", {})}


def restore_backup(path: Path, data_dir: Path) -> dict:
    """Replace the data in ``data_dir`` with the backup. The app must not be using ``data_dir``.

    The replaced data is moved to ``data_dir/backups/replaced-<time>/`` (nothing is deleted)."""
    info = verify_backup(path)
    from app.db.migrate import known_revisions

    rev = info.get("schema_revision")
    if rev and rev not in known_revisions():
        raise ValidationFailed(f"The backup's database schema ({rev}) is newer than this version of Local AI Lab "
                               "supports. Install the newer version to restore it.")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    replaced = data_dir / "backups" / f"replaced-{stamp}"
    replaced.mkdir(parents=True, exist_ok=True)
    staging = data_dir / f".restore-{stamp}"
    staging.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(path) as zf:
            for arc in zf.namelist():
                if arc == "manifest.json":
                    continue
                target = (staging / _safe_name(arc)).resolve()
                if staging.resolve() not in target.parents:
                    raise ValidationFailed("The backup contains an unsafe path and was rejected.", code="unsafe_archive")
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(arc) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
        # move the current data out of the way
        moved = []
        for name in (DB_NAME, f"{DB_NAME}-wal", f"{DB_NAME}-shm", "mlflow.db", "mlflow.db-wal", "mlflow.db-shm"):
            if (data_dir / name).exists():
                shutil.move(str(data_dir / name), str(replaced / name))
                moved.append(name)
        for rel in list(FILE_DIRS) + [d for ds in OPTIONAL_DIRS.values() for d in ds]:
            if (staging / "files" / rel).exists() and (data_dir / rel).exists():
                (replaced / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(data_dir / rel), str(replaced / rel))
                moved.append(rel)
        # put the backup in place
        for name in ("local_ai_lab.db", "mlflow.db"):
            if (staging / "db" / name).exists():
                shutil.move(str(staging / "db" / name), str(data_dir / name))
        if (staging / "files").exists():
            for p in sorted((staging / "files").rglob("*")):
                if p.is_file():
                    dest = data_dir / p.relative_to(staging / "files")
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(p), str(dest))
        shutil.rmtree(data_dir / "arena" / "features", ignore_errors=True)  # cache – rebuilt on demand
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    (replaced / "RESTORE_INFO.txt").write_text(
        f"Data replaced by restoring {path.name} on {stamp}.\nMoved here: {', '.join(moved) or 'nothing'}\n"
        "To undo, stop Local AI Lab and move these files back into the data folder.\n", encoding="utf-8")
    log.info("backup restored", extra={"backup": str(path), "replaced_to": str(replaced)})
    return {**info, "restored_from": path.name, "previous_data_moved_to": str(replaced)}


# ----------------------------------------------------------------------------- staged restore (UI/API)
def stage_restore(path: Path, data_dir: Path) -> dict:
    info = verify_backup(path)
    pending = data_dir / PENDING_DIR
    pending.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, pending / "backup.zip")
    (pending / "info.json").write_text(json.dumps({**info, "source": path.name,
                                                   "staged_at": datetime.now().isoformat(timespec="seconds")}),
                                       encoding="utf-8")
    return info


def pending_restore(data_dir: Path) -> dict | None:
    info = data_dir / PENDING_DIR / "info.json"
    return json.loads(info.read_text(encoding="utf-8")) if info.exists() else None


def cancel_pending_restore(data_dir: Path) -> None:
    shutil.rmtree(data_dir / PENDING_DIR, ignore_errors=True)


def apply_pending_restore(data_dir: Path) -> dict | None:
    """Called at start-up before the database is opened."""
    zip_path = data_dir / PENDING_DIR / "backup.zip"
    if not zip_path.exists():
        return None
    try:
        result = restore_backup(zip_path, data_dir)
    except Exception as exc:  # noqa: BLE001 - keep the app usable; the staged file stays for inspection
        log.error("staged restore failed", extra={"error": str(exc)[:300]})
        (data_dir / PENDING_DIR / "FAILED.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        (data_dir / PENDING_DIR / "backup.zip").rename(data_dir / PENDING_DIR / "backup.failed.zip")
        return {"ok": False, "error": str(exc)[:300]}
    cancel_pending_restore(data_dir)
    return {"ok": True, **result}


def list_backups(backups_dir: Path) -> list[dict]:
    if not backups_dir.exists():
        return []
    out = []
    for p in sorted(backups_dir.glob("*.zip"), reverse=True):
        out.append({"name": p.name, "size_bytes": p.stat().st_size,
                    "created_at": datetime.fromtimestamp(p.stat().st_mtime).astimezone().isoformat(timespec="seconds")})
    for p in sorted(backups_dir.glob("pre-migration-*.db"), reverse=True):
        out.append({"name": p.name, "size_bytes": p.stat().st_size, "kind": "pre-migration database copy",
                    "created_at": datetime.fromtimestamp(p.stat().st_mtime).astimezone().isoformat(timespec="seconds")})
    return out


def prune_backups(backups_dir: Path, keep: int) -> list[str]:
    zips = sorted(backups_dir.glob("local-ai-lab-backup-*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    removed = []
    for p in zips[keep:]:
        p.unlink(missing_ok=True)
        removed.append(p.name)
    return removed
