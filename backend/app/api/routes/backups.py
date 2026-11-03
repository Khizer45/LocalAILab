"""Backup and restore (administrators only)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.api.deps import AppContext, get_ctx, require_admin
from app.core import backup as B
from app.core.audit import audit
from app.core.errors import NotFoundError, ValidationFailed
from app.db.migrate import current_revision
from app.models.core import User

router = APIRouter(prefix="/api/system/backups", tags=["backup"])


class BackupRequest(BaseModel):
    label: str | None = Field(None, max_length=40)
    include_mlflow: bool = True
    include_backbones: bool = False


def _file(ctx: AppContext, name: str):
    path = (ctx.settings.backups_dir / name).resolve()
    if ctx.settings.backups_dir.resolve() not in path.parents or not path.is_file():
        raise NotFoundError("Backup not found.")
    return path


@router.get("")
def list_all(ctx: AppContext = Depends(get_ctx), _: User = Depends(require_admin)) -> dict:
    return {"backups": B.list_backups(ctx.settings.backups_dir), "pending_restore": B.pending_restore(ctx.settings.data_dir),
            "backups_dir": str(ctx.settings.backups_dir)}


@router.post("", status_code=201)
def create(body: BackupRequest, request: Request, ctx: AppContext = Depends(get_ctx),
           user: User = Depends(require_admin)) -> dict:
    path = B.create_backup(ctx.settings.data_dir, ctx.settings.backups_dir, schema_revision=current_revision(ctx.engine),
                           include_mlflow=body.include_mlflow, include_backbones=body.include_backbones,
                           label=body.label)
    audit(ctx, user, "backup.create", path.name, request=request)
    return {"name": path.name, "size_bytes": path.stat().st_size, **B.verify_backup(path)}


@router.get("/{name}/download")
def download(name: str, request: Request, ctx: AppContext = Depends(get_ctx), user: User = Depends(require_admin)):
    path = _file(ctx, name)
    audit(ctx, user, "backup.download", name, request=request)
    return FileResponse(path, filename=path.name, media_type="application/zip")


@router.post("/{name}/verify")
def verify(name: str, ctx: AppContext = Depends(get_ctx), _: User = Depends(require_admin)) -> dict:
    path = _file(ctx, name)
    if path.suffix != ".zip":
        raise ValidationFailed("Only backup ZIP files can be verified.")
    return B.verify_backup(path)


@router.delete("/{name}", status_code=204)
def delete(name: str, request: Request, ctx: AppContext = Depends(get_ctx), user: User = Depends(require_admin)) -> None:
    _file(ctx, name).unlink()
    audit(ctx, user, "backup.delete", name, request=request)


@router.post("/{name}/restore")
def restore_existing(name: str, request: Request, ctx: AppContext = Depends(get_ctx),
                     user: User = Depends(require_admin)) -> dict:
    """Verify and stage a backup; it replaces the current data at the next start of Local AI Lab."""
    info = B.stage_restore(_file(ctx, name), ctx.settings.data_dir)
    audit(ctx, user, "backup.restore_staged", name, request=request)
    return {"staged": True, **info, "next_step": "Restart Local AI Lab to apply the restore. The current data will be "
                                                 "moved to data/backups/replaced-<time>/ first."}


@router.post("/restore-upload")
async def restore_upload(request: Request, file: UploadFile = File(...), ctx: AppContext = Depends(get_ctx),
                         user: User = Depends(require_admin)) -> dict:
    dest = ctx.settings.backups_dir / f"uploaded-{(file.filename or 'backup.zip').replace('/', '_')[-80:]}"
    if not dest.name.endswith(".zip"):
        dest = dest.with_suffix(".zip")
    ctx.settings.backups_dir.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)
    try:
        info = await run_in_threadpool(B.stage_restore, dest, ctx.settings.data_dir)
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    audit(ctx, user, "backup.restore_staged", dest.name, request=request)
    return {"staged": True, **info, "next_step": "Restart Local AI Lab to apply the restore."}


@router.delete("/pending-restore/cancel", status_code=204)
def cancel_restore(request: Request, ctx: AppContext = Depends(get_ctx), user: User = Depends(require_admin)) -> None:
    B.cancel_pending_restore(ctx.settings.data_dir)
    audit(ctx, user, "backup.restore_cancelled", None, request=request)
