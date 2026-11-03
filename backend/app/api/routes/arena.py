"""Model Arena API: bring your own dataset and models; choose on validation, report on test."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, PlainTextResponse, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import AppContext, current_user, get_ctx, get_db, require_admin
from app.arena import images as IMG
from app.arena.algorithms import catalog
from app.arena.report import dataset_out, model_out, result_out, run_out, to_csv, to_markdown
from app.arena.runner import ArenaRunRequest
from app.core.errors import AppError, ConflictError, NotFoundError, OfflineModeViolation, ValidationFailed
from app.core.network_guard import OfflineNetworkBlocked, guarded_http_client
from app.models.arena import ArenaDataset, ArenaModel, ArenaResult, ArenaRun
from app.models.core import User

router = APIRouter(prefix="/api/arena", tags=["model-arena"])


def _enabled(ctx: AppContext) -> None:
    if not ctx.settings.arena_enabled:
        raise AppError("The Model Arena is disabled (ARENA_ENABLED=false).", code="arena_disabled", status_code=404)


async def _receive(request: Request, file: UploadFile, ctx: AppContext):
    """Stream an upload to the staging folder with a size limit."""
    _enabled(ctx)
    limit = ctx.settings.arena_max_upload_mb * 1024 * 1024
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > limit + 1024 * 1024:
        raise ValidationFailed(f"File is larger than the {ctx.settings.arena_max_upload_mb} MB limit.",
                               code="file_too_large", hint="Raise ARENA_MAX_UPLOAD_MB in the backend .env file.")
    dest = ctx.arena_store.staging_path(file.filename or "upload")
    size = 0
    try:
        with dest.open("wb") as f:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise ValidationFailed(f"File is larger than the {ctx.settings.arena_max_upload_mb} MB limit.",
                                           code="file_too_large")
                f.write(chunk)
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    if size == 0:
        dest.unlink(missing_ok=True)
        raise ValidationFailed("The uploaded file is empty.")
    return dest


@router.get("/catalog")
def get_catalog(ctx: AppContext = Depends(get_ctx), _: User = Depends(current_user)) -> dict:
    _enabled(ctx)
    return {"algorithms": catalog(), "backbones": IMG.backbone_status(ctx.settings.arena_dir),
            "normalizations": IMG.NORMALIZATION_HELP, "limits": ctx.arena_store.limits}


# ------------------------------------------------------------------ datasets
@router.post("/datasets/preview")
async def preview_dataset(request: Request, file: UploadFile = File(...), ctx: AppContext = Depends(get_ctx),
                          _: User = Depends(current_user)) -> dict:
    path = await _receive(request, file, ctx)
    try:
        return await run_in_threadpool(ctx.arena_store.preview, path, file.filename or "")
    finally:
        path.unlink(missing_ok=True)


@router.post("/datasets", status_code=201)
async def create_dataset(request: Request, file: UploadFile = File(...), name: str | None = Form(None),
                         target: str | None = Form(None), task: str = Form("auto"),
                         text_column: str | None = Form(None), ignore: str | None = Form(None),
                         test_size: float = Form(0.2), val_size: float = Form(0.2), seed: int = Form(42),
                         split_strategy: str = Form("random"), group_column: str | None = Form(None),
                         time_column: str | None = Form(None), user: User = Depends(current_user),
                         ctx: AppContext = Depends(get_ctx)) -> dict:
    path = await _receive(request, file, ctx)
    try:
        row = await run_in_threadpool(
            ctx.arena_store.create_dataset, user.id, path, file.filename or "dataset.csv", name=name, target=target,
            task=task, text_column=text_column, ignore=[c.strip() for c in (ignore or "").split(",") if c.strip()],
            test_size=test_size, seed=seed, val_size=val_size, split_strategy=split_strategy,
            group_column=group_column, time_column=time_column)
    finally:
        path.unlink(missing_ok=True)
    return dataset_out(row)


@router.get("/datasets")
def list_datasets(user: User = Depends(current_user), db: Session = Depends(get_db),
                  ctx: AppContext = Depends(get_ctx)) -> dict:
    _enabled(ctx)
    rows = db.scalars(select(ArenaDataset).where(ArenaDataset.owner_id == user.id)
                      .order_by(ArenaDataset.created_at.desc())).all()
    return {"datasets": [dataset_out(d) for d in rows]}


@router.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: int, user: User = Depends(current_user), db: Session = Depends(get_db),
                ctx: AppContext = Depends(get_ctx)) -> dict:
    _enabled(ctx)
    d = ctx.arena_store.dataset_owned(db, user.id, dataset_id)
    return dataset_out(d) | {"algorithms": catalog(d.kind, d.task),
                             "models": ctx.arena_store.compatible_models(db, user.id, d)}


@router.get("/datasets/{dataset_id}/runs")
def dataset_runs(dataset_id: int, user: User = Depends(current_user), db: Session = Depends(get_db),
                 ctx: AppContext = Depends(get_ctx)) -> dict:
    """Lineage: every comparison run that used this dataset (and which split version)."""
    _enabled(ctx)
    d = ctx.arena_store.dataset_owned(db, user.id, dataset_id)
    runs = db.scalars(select(ArenaRun).where(ArenaRun.owner_id == user.id, ArenaRun.dataset_id == d.id)
                      .order_by(ArenaRun.created_at.desc())).all()
    return {"dataset": {"id": d.id, "sha256": d.sha256, "split_sha256": d.split_sha256},
            "runs": [{"id": r.id, "name": r.name, "status": r.status, "protocol": r.protocol,
                      "split_sha256": (r.dataset_info or {}).get("split_sha256"),
                      "recommendation": ((r.summary or {}).get("recommendation") or {}).get("name"),
                      "created_at": r.created_at.isoformat() + "Z"} for r in runs]}


@router.get("/datasets/{dataset_id}/split.csv")
def export_split(dataset_id: int, user: User = Depends(current_user), db: Session = Depends(get_db),
                 ctx: AppContext = Depends(get_ctx)):
    """Row → split membership, so models trained elsewhere (e.g. Colab) can use exactly the training rows."""
    import csv
    import io

    from app.arena import data as D

    _enabled(ctx)
    d = ctx.arena_store.dataset_owned(db, user.id, dataset_id)
    folder = ctx.arena_store.abs(d.path)
    split = D.load_split(folder)
    member = {}
    for part in ("train", "val", "test"):
        for i in split[part]:
            member[int(i)] = part
    buf = io.StringIO()
    w = csv.writer(buf)
    sources = [it["source"] for it in D.load_manifest(folder)] if d.kind == "image" else None
    source_rows = None
    if d.kind != "image":
        df = D.read_csv((folder / "data.csv").read_bytes(), max_rows=ctx.settings.arena_max_rows,
                        max_columns=ctx.settings.arena_max_columns)
        df, _, _ = D.clean_frame(df, d.target, d.task, group_column=d.group_column, time_column=d.time_column)
        source_rows = df.attrs.get("source_rows")
    w.writerow(["row", "split"] + (["source"] if sources else ["source_row"]))
    for i in range(d.n_rows):
        extra = [sources[i]] if sources else [source_rows[i] if source_rows else ""]
        w.writerow([i, member.get(i, "excluded")] + extra)
    head = (f"# dataset_sha256={d.sha256} split_sha256={split['sha256']} strategy={split.get('strategy', d.split_source)}"
            " row=position after cleaning; source_row=0-based data row in the uploaded file (header not counted)"
            "; source=path inside the uploaded ZIP\n")
    return Response(head + buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="arena_dataset_{d.id}_split.csv"'})


@router.delete("/datasets/{dataset_id}", status_code=204)
def delete_dataset(dataset_id: int, user: User = Depends(current_user), db: Session = Depends(get_db),
                   ctx: AppContext = Depends(get_ctx)) -> None:
    _enabled(ctx)
    busy = db.scalar(select(ArenaRun).where(ArenaRun.dataset_id == dataset_id,
                                            ArenaRun.status.in_(["queued", "running"])))
    if busy:
        raise ConflictError("A comparison using this dataset is still running.")
    ctx.arena_store.delete_dataset(db, user.id, dataset_id)


# ------------------------------------------------------------------ uploaded models
@router.post("/models", status_code=201)
async def upload_model(request: Request, file: UploadFile = File(...), name: str | None = Form(None),
                       kind: str = Form(...), task: str = Form("classification"), role: str = Form("predictor"),
                       labels: str | None = Form(None), feature_columns: str | None = Form(None),
                       size: int | None = Form(None), layout: str | None = Form(None),
                       normalization: str | None = Form(None), color: str | None = Form(None),
                       resize_mode: str | None = Form(None), trusted: bool = Form(False),
                       training_data: str = Form("unknown"), provenance_note: str | None = Form(None),
                       user: User = Depends(current_user), ctx: AppContext = Depends(get_ctx)) -> dict:
    path = await _receive(request, file, ctx)
    pp = {k: v for k, v in {"size": size, "layout": layout, "normalization": normalization, "color": color,
                            "resize_mode": resize_mode}.items() if v not in (None, "", "auto")}
    try:
        row = await run_in_threadpool(
            ctx.arena_store.create_model, user.id, user.is_admin, path, file.filename or "model.onnx", name=name,
            kind=kind, task=task, role=role, labels_text=labels, feature_columns=feature_columns, preprocess=pp,
            trusted=trusted, provenance={"training_data": training_data, "note": provenance_note})
    finally:
        path.unlink(missing_ok=True)
    return model_out(row)


@router.get("/models")
def list_models(user: User = Depends(current_user), db: Session = Depends(get_db),
                ctx: AppContext = Depends(get_ctx)) -> dict:
    _enabled(ctx)
    rows = db.scalars(select(ArenaModel).where(ArenaModel.owner_id == user.id)
                      .order_by(ArenaModel.created_at.desc())).all()
    return {"models": [model_out(m) for m in rows]}


@router.delete("/models/{model_id}", status_code=204)
def delete_model(model_id: int, user: User = Depends(current_user), db: Session = Depends(get_db),
                 ctx: AppContext = Depends(get_ctx)) -> None:
    _enabled(ctx)
    ctx.arena_store.delete_model(db, user.id, model_id)


# ------------------------------------------------------------------ backbones
@router.get("/backbones")
def backbones(ctx: AppContext = Depends(get_ctx), _: User = Depends(current_user)) -> dict:
    _enabled(ctx)
    return {"backbones": IMG.backbone_status(ctx.settings.arena_dir)}


@router.post("/backbones/{backbone_id}/download")
def download_backbone(backbone_id: str, ctx: AppContext = Depends(get_ctx), _: User = Depends(require_admin)) -> dict:
    """Download a pretrained feature extractor. Blocked while Offline-Only mode is on."""
    _enabled(ctx)
    if ctx.settings_service.get().offline_only:
        raise OfflineModeViolation(
            "Offline-Only mode blocks downloads.",
            hint="Run scripts\\download_backbones.bat once (it downloads and verifies the file), or switch Offline-Only "
                 "off for a moment in Settings, or copy the .onnx file into backend\\data\\arena\\backbones.")
    try:
        with guarded_http_client("https://media.githubusercontent.com", timeout=600) as http:
            return IMG.download_backbone(ctx.settings.arena_dir, backbone_id, http)
    except OfflineNetworkBlocked as exc:
        raise OfflineModeViolation(str(exc)) from exc
    except AppError:
        raise
    except Exception as exc:  # noqa: BLE001 - network errors
        raise AppError(f"Download failed: {type(exc).__name__}", code="download_failed", status_code=502,
                       hint="Check the internet connection or use scripts\\download_backbones.bat.") from exc


# ------------------------------------------------------------------ runs
@router.post("/runs", status_code=201)
def create_run(body: ArenaRunRequest, user: User = Depends(current_user), ctx: AppContext = Depends(get_ctx)) -> dict:
    _enabled(ctx)
    return run_out(ctx.arena.submit(user.id, body))


@router.get("/runs")
def list_runs(limit: int = 100, user: User = Depends(current_user), db: Session = Depends(get_db),
              ctx: AppContext = Depends(get_ctx)) -> dict:
    _enabled(ctx)
    runs = db.scalars(select(ArenaRun).where(ArenaRun.owner_id == user.id).order_by(ArenaRun.created_at.desc())
                      .limit(min(limit, 500))).all()
    return {"runs": [run_out(r) for r in runs]}


@router.get("/runs/{run_id}")
def get_run(run_id: int, user: User = Depends(current_user), db: Session = Depends(get_db),
            ctx: AppContext = Depends(get_ctx)) -> dict:
    _enabled(ctx)
    run = ctx.arena.run_owned(db, user.id, run_id)
    results = db.scalars(select(ArenaResult).where(ArenaResult.run_id == run.id).order_by(ArenaResult.id)).all()
    return run_out(run) | {"results": [result_out(r) for r in results]}


@router.post("/runs/{run_id}/cancel")
def cancel_run(run_id: int, user: User = Depends(current_user), ctx: AppContext = Depends(get_ctx)) -> dict:
    ctx.arena.cancel(user.id, run_id)
    return {"status": "cancelling"}


@router.delete("/runs/{run_id}", status_code=204)
def delete_run(run_id: int, user: User = Depends(current_user), db: Session = Depends(get_db),
               ctx: AppContext = Depends(get_ctx)) -> None:
    run = ctx.arena.run_owned(db, user.id, run_id)
    if run.status in ("queued", "running"):
        raise ConflictError("Cancel the run before deleting it.")
    if run.best_model_path:
        import shutil

        shutil.rmtree(ctx.settings.arena_dir / "runs" / str(run.id), ignore_errors=True)
    db.query(ArenaResult).filter(ArenaResult.run_id == run.id).delete()
    db.delete(run)
    db.commit()


@router.get("/runs/{run_id}/export")
def export_run(run_id: int, format: str = "json", user: User = Depends(current_user), db: Session = Depends(get_db),
               ctx: AppContext = Depends(get_ctx)):
    run = ctx.arena.run_owned(db, user.id, run_id)
    results = list(db.scalars(select(ArenaResult).where(ArenaResult.run_id == run.id).order_by(ArenaResult.id)))
    fname = f"model_arena_run_{run.id}"
    if format == "csv":
        return Response(to_csv(run, results), media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="{fname}.csv"'})
    if format in ("md", "markdown"):
        return PlainTextResponse(to_markdown(run, results), media_type="text/markdown",
                                 headers={"Content-Disposition": f'attachment; filename="{fname}.md"'})
    payload = run_out(run) | {"results": [result_out(r, with_predictions=True) for r in results]}
    return Response(json.dumps(payload, indent=2, default=str), media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="{fname}.json"'})


@router.get("/runs/{run_id}/model")
def download_trained_model(run_id: int, user: User = Depends(current_user), db: Session = Depends(get_db),
                           ctx: AppContext = Depends(get_ctx)):
    """The trained scikit-learn bundle of the recommended (or best) built-in model."""
    run = ctx.arena.run_owned(db, user.id, run_id)
    if not run.best_model_path:
        raise NotFoundError("This run has no trained built-in model to download.")
    path = (ctx.settings.arena_dir / run.best_model_path).resolve()
    if ctx.settings.arena_dir.resolve() not in path.parents or not path.exists():
        raise NotFoundError("Model file not found.")
    return FileResponse(path, filename=f"arena_run_{run.id}_{path.stem}.joblib", media_type="application/octet-stream")
