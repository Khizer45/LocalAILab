"""Experiment bundles: export, verify, import and re-run (see app/services/bundles.py)."""

from __future__ import annotations

import json
from datetime import datetime

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agents.report import summarize as agent_summarize
from app.agents.report import to_markdown as agent_md
from app.api.deps import AppContext, current_user, get_ctx, get_db
from app.arena import data as D
from app.arena.report import to_markdown as arena_md
from app.benchmark.aggregate import compare_models
from app.benchmark.aggregate import summarize as bench_summarize
from app.benchmark.aggregate import to_markdown as bench_md
from app.benchmark.runner import BenchmarkRequest
from app.core.errors import NotFoundError, ValidationFailed
from app.models.agents import AgentResult, AgentRun
from app.models.arena import ArenaDataset, ArenaResult, ArenaRun
from app.models.benchmark import BenchmarkResult, BenchmarkRun
from app.models.core import User
from app.services.bundles import export_bundle, read_bundle, verify_bundle

router = APIRouter(prefix="/api/bundles", tags=["bundles"])
MAX_BUNDLE = 500 * 1024 * 1024

MODELS = {"benchmark": (BenchmarkRun, BenchmarkResult), "agent": (AgentRun, AgentResult),
          "arena": (ArenaRun, ArenaResult)}


def _owned(db: Session, kind: str, run_id: int, user: User):
    if kind not in MODELS:
        raise NotFoundError("Unknown bundle kind.")
    run_cls, res_cls = MODELS[kind]
    run = db.get(run_cls, run_id)
    if run is None or run.owner_id != user.id:
        raise NotFoundError("Run not found.")
    results = list(db.scalars(select(res_cls).where(res_cls.run_id == run.id).order_by(res_cls.id)))
    return run, results


@router.get("/{kind}/{run_id}")
def export(kind: str, run_id: int, scope: str = "full", include_dataset: bool = False,
           ctx: AppContext = Depends(get_ctx), db: Session = Depends(get_db), user: User = Depends(current_user)):
    run, results = _owned(db, kind, run_id, user)
    suite = dataset = data_file = None
    if kind == "benchmark":
        suite = run.suite_snapshot
        report = bench_md(run, bench_summarize(results), comparison=compare_models(results))
    elif kind == "agent":
        suite = run.suite_snapshot
        report = agent_md(run, agent_summarize(results))
    else:
        report = arena_md(run, results)
        dataset = {"dataset_info": run.dataset_info, "labels": (run.summary or {}).get("labels")}
        ds = db.get(ArenaDataset, run.dataset_id) if run.dataset_id else None
        if ds is not None:
            folder = ctx.arena_store.abs(ds.path)
            split = D.load_split(folder)
            dataset["split"] = {k: split[k] for k in ("train", "val", "test", "sha256")}
            if scope == "full":
                loaded = ctx.arena_store.load(ds)
                rows = list(split["val"]) + list(split["test"])
                dataset["labels_by_row"] = {str(int(i)): (str(loaded.y[i]) if ds.task == "classification"
                                                          else float(loaded.y[i])) for i in rows}
            if include_dataset and ds.kind != "image" and (folder / "data.csv").exists():
                data_file = (ds.filename or "data.csv", (folder / "data.csv").read_bytes())
    blob = export_bundle(kind, run, results, scope=scope, suite=suite, dataset=dataset, report=report,
                         data_file=data_file)
    name = f"experiment_{kind}_{run.id}_{scope}.zip"
    return Response(blob, media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="{name}"'})


async def _read(file: UploadFile) -> bytes:
    raw = await file.read(MAX_BUNDLE + 1)
    if len(raw) > MAX_BUNDLE:
        raise ValidationFailed("The bundle is larger than 500 MB.")
    return raw


@router.post("/verify")
async def verify(file: UploadFile = File(...), _: User = Depends(current_user)) -> dict:
    return verify_bundle(await _read(file))


def _dt(v):
    if not v:
        return None
    try:
        return datetime.fromisoformat(str(v).replace("Z", ""))
    except ValueError:
        return None


@router.post("/import", status_code=201)
async def import_bundle(file: UploadFile = File(...), ctx: AppContext = Depends(get_ctx),
                        user: User = Depends(current_user)) -> dict:
    """Store a bundle's run (read-only copy) under the current user, so it can be viewed and compared."""
    raw = await _read(file)
    report = verify_bundle(raw)
    if not report["ok"]:
        raise ValidationFailed("The bundle failed verification: " + "; ".join(report["problems"][:3]))
    manifest, files = read_bundle(raw)
    kind = manifest["kind"]
    run_cls, res_cls = MODELS[kind]
    data = json.loads(files["run.json"])
    suite = json.loads(files["suite.json"]) if "suite.json" in files else None
    run_cols = {c.name for c in run_cls.__table__.columns} - {"id", "owner_id", "mlflow_run_id"}
    values = {k: v for k, v in data.items() if k in run_cols}
    for k in ("created_at", "started_at", "finished_at"):
        if k in values:
            values[k] = _dt(values[k])
    values["name"] = f"[imported] {data.get('name', '')}"[:200]
    values["config"] = {**(data.get("config") or {}), "imported": {"from_run": manifest.get("run_id"),
                                                                   "scope": manifest.get("scope"),
                                                                   "app_version": manifest.get("app_version")}}
    if kind in ("benchmark", "agent"):
        values["suite_snapshot"] = suite or {}
    if kind == "arena":
        values["dataset_id"] = None
        values["best_model_path"] = None
    res_cols = {c.name for c in res_cls.__table__.columns} - {"id", "run_id"}
    with ctx.SessionLocal() as db:
        run = run_cls(owner_id=user.id, **values)
        db.add(run)
        db.flush()
        for row in [json.loads(x) for x in files.get("results.jsonl", b"").decode().splitlines() if x.strip()]:
            vals = {k: v for k, v in row.items() if k in res_cols}
            if "created_at" in vals:
                vals["created_at"] = _dt(vals["created_at"])
            db.add(res_cls(run_id=run.id, **vals))
        db.commit()
        return {"kind": kind, "run_id": run.id, "verification": report}


@router.post("/rerun", status_code=201)
async def rerun(file: UploadFile = File(...), models: str | None = Form(None), ctx: AppContext = Depends(get_ctx),
                user: User = Depends(current_user)) -> dict:
    """Start a new run with the bundle's exact suite snapshot and settings (optionally other models)."""
    raw = await _read(file)
    manifest, files = read_bundle(raw)
    kind = manifest["kind"]
    if kind == "arena":
        raise ValidationFailed("Re-run Model Arena comparisons from the Model Arena page with the same dataset – "
                               "the bundle's split indices and settings are in run.json / dataset.json.")
    if manifest.get("scope") != "full" or "suite.json" not in files:
        raise ValidationFailed("Re-running needs a 'full' bundle (it contains the suite snapshot).")
    run = json.loads(files["run.json"])
    suite = json.loads(files["suite.json"])
    model_list = [m.strip() for m in (models or "").split(",") if m.strip()] or list(run.get("models") or [])
    cfg = dict(run.get("config") or {})
    if kind == "benchmark":
        builtin = ctx.suites.builtin().get(suite.get("id"))
        cases_sha = cfg.get("cases_sha256")
        from app.benchmark.runner import suite_snapshot

        if builtin is not None and suite_snapshot(builtin, [c for c in builtin.cases
                                                           if c.id in set(cfg.get("case_ids", []))])["cases_sha256"] == cases_sha:
            suite_id = builtin.id
        else:
            data = {"id": f"{suite.get('id', 'imported')}-imported"[:60], "name": f"{suite.get('name', '')} (imported)",
                    "version": "1.0.0", "system": suite.get("system"),
                    "default_max_tokens": suite.get("default_max_tokens", 200), "cases": suite.get("cases", [])}
            suite_id = ctx.suites.save(user.id, data, note=f"from bundle of run {manifest.get('run_id')}").id
        allowed = set(BenchmarkRequest.model_fields)
        req = BenchmarkRequest(**{k: v for k, v in cfg.items() if k in allowed and k not in ("suite_id", "models",
                                                                                             "categories", "max_cases",
                                                                                             "name", "case_ids")},
                               suite_id=suite_id, models=model_list, name=f"Re-run of {run.get('name', '')}"[:200],
                               case_ids=cfg.get("case_ids"))
        new = ctx.benchmarks.submit(user.id, req)
        return {"kind": kind, "run_id": new.id, "suite_id": suite_id}
    from app.agents.runner import AgentRunRequest

    sid = suite.get("id")
    builtin = ctx.agent_suites.builtin().get(sid)
    if builtin is None or builtin.sha256 != run.get("suite_hash"):
        data = {k: v for k, v in suite.items() if k not in ("sha256", "source")}
        data["id"] = (sid + "-imported")[:60] if builtin is not None else sid
        sid = ctx.agent_suites.save(user.id, data).id
    allowed = set(AgentRunRequest.model_fields)
    req = AgentRunRequest(**{k: v for k, v in cfg.items() if k in allowed and k not in ("suite_id", "models",
                                                                                         "categories", "name")},
                          suite_id=sid, models=model_list, name=f"Re-run of {run.get('name', '')}"[:200])
    new = ctx.agents.submit(user.id, req)
    return {"kind": kind, "run_id": new.id, "suite_id": sid}
