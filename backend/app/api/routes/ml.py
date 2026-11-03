from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import AppContext, current_user, get_ctx, get_db
from app.core.errors import AppError, NotFoundError
from app.ml.intent_baseline import ARTIFACT_NAME, classification_metrics, train_and_evaluate
from app.ml.intent_data import load_split
from app.models.benchmark import BenchmarkResult, BenchmarkRun
from app.models.core import User
from app.models.ml import MLTrainingRun
from app.schemas import PredictRequest

router = APIRouter(prefix="/api/ml", tags=["ml"])


def _enabled(ctx: AppContext) -> None:
    if not ctx.settings.ml_baseline_enabled:
        raise AppError("The ML baseline module is disabled (ML_BASELINE_ENABLED=false).", code="ml_disabled",
                       status_code=404)


def _run_out(r: MLTrainingRun) -> dict:
    return {"id": r.id, "dataset_id": r.dataset_id, "dataset_version": r.dataset_version,
            "dataset_hash": r.dataset_hash, "selected_model": r.selected_model, "metrics": r.metrics,
            "artifact_path": r.artifact_path, "mlflow_run_id": r.mlflow_run_id,
            "created_at": r.created_at.isoformat() + "Z"}


@router.get("/dataset")
def dataset(ctx: AppContext = Depends(get_ctx), _: User = Depends(current_user)) -> dict:
    _enabled(ctx)
    split = load_split(ctx.settings.datasets_dir)
    return split.summary() | {"examples": [{"text": t, "label": y} for t, y in split.train[:12]]}


@router.get("/baseline")
def latest_baseline(ctx: AppContext = Depends(get_ctx), db: Session = Depends(get_db),
                    _: User = Depends(current_user)) -> dict:
    _enabled(ctx)
    runs = db.scalars(select(MLTrainingRun).order_by(MLTrainingRun.id.desc()).limit(20)).all()
    return {"latest": _run_out(runs[0]) if runs else None, "history": [_run_out(r) | {"metrics": {
        "test": {k: r.metrics.get("test", {}).get(k) for k in ("accuracy", "f1_macro")}}} for r in runs]}


@router.post("/baseline/train", status_code=201)
def train(ctx: AppContext = Depends(get_ctx), db: Session = Depends(get_db), _: User = Depends(current_user)) -> dict:
    _enabled(ctx)
    report = train_and_evaluate(ctx.settings.datasets_dir, ctx.settings.ml_dir)
    t = report["test"]
    run_id = ctx.tracker.log_run(
        "local-ai-lab-ml-baseline", f"intent-{report['model_selection']['selected']}",
        params={"dataset_version": report["dataset"]["version"], "dataset_sha256": report["dataset"]["sha256"],
                "selected_model": report["model_selection"]["selected"], "seed": report["dataset"]["seed"],
                **{f"split_{k}": v for k, v in report["dataset"]["sizes"].items()}},
        metrics={"test_accuracy": t["accuracy"], "test_f1_macro": t["f1_macro"],
                 "test_precision_macro": t["precision_macro"], "test_recall_macro": t["recall_macro"],
                 "latency_ms_per_example": t["latency_ms_per_example"],
                 **{f"val_f1_{c['model']}": c["val_f1_macro"] for c in report["model_selection"]["candidates"]}},
        artifacts={"report.json": report}, tags={"task": "intent_classification"},
    )
    row = MLTrainingRun(dataset_id=report["dataset"]["dataset_id"], dataset_version=report["dataset"]["version"],
                        dataset_hash=report["dataset"]["sha256"], selected_model=report["model_selection"]["selected"],
                        metrics=report, artifact_path=report["artifact_path"], mlflow_run_id=run_id)
    db.add(row)
    db.commit()
    db.refresh(row)
    return _run_out(row)


@router.post("/baseline/predict")
def predict(body: PredictRequest, ctx: AppContext = Depends(get_ctx), _: User = Depends(current_user)) -> dict:
    _enabled(ctx)
    out = ctx.baseline.predict(body.text)
    if out is None:
        raise NotFoundError("No trained baseline yet. Train it first on the ML Lab page.", code="not_trained")
    return out


@router.get("/comparison")
def comparison(ctx: AppContext = Depends(get_ctx), db: Session = Depends(get_db),
               user: User = Depends(current_user)) -> dict:
    """Baseline vs. LLM zero-shot classification on the same held-out test split."""
    _enabled(ctx)
    split = load_split(ctx.settings.datasets_dir)
    latest = db.scalars(select(MLTrainingRun).order_by(MLTrainingRun.id.desc()).limit(1)).first()
    rows = []
    if latest:
        t = latest.metrics["test"]
        rows.append({"approach": "Traditional ML (scikit-learn)", "model": latest.selected_model, "source": f"training run {latest.id}",
                     **{k: t[k] for k in ("n", "accuracy", "precision_macro", "recall_macro", "f1_macro")},
                     "latency_ms_per_example": t.get("latency_ms_per_example"), "unparseable_predictions": 0})
    runs = db.scalars(select(BenchmarkRun).where(BenchmarkRun.owner_id == user.id,
                                                 BenchmarkRun.suite_id == "intent-classification",
                                                 BenchmarkRun.status == "completed")
                      .order_by(BenchmarkRun.id.desc()).limit(10)).all()
    gold = {f"intent-test-{i:03d}": y for i, (_, y) in enumerate(split.test)}
    for run in runs:
        results = db.scalars(select(BenchmarkResult).where(BenchmarkResult.run_id == run.id,
                                                           BenchmarkResult.repeat_index == 0)).all()
        by_model: dict[str, list[BenchmarkResult]] = {}
        for r in results:
            by_model.setdefault(r.model, []).append(r)
        for model, rs in by_model.items():
            y_true, y_pred, lat = [], [], []
            for r in rs:
                if r.case_id not in gold:
                    continue
                y_true.append(gold[r.case_id])
                y_pred.append((r.eval_details or {}).get("details", {}).get("predicted"))
                if r.total_ms:
                    lat.append(r.total_ms)
            if not y_true:
                continue
            m = classification_metrics(y_true, y_pred, split.labels)
            rows.append({"approach": "LLM zero-shot (deep learning)", "model": model, "source": f"benchmark run {run.id}",
                         **{k: m[k] for k in ("n", "accuracy", "precision_macro", "recall_macro", "f1_macro")},
                         "latency_ms_per_example": round(sum(lat) / len(lat), 1) if lat else None,
                         "unparseable_predictions": m["unparseable_predictions"],
                         "confusion_matrix": m["confusion_matrix"]})
    return {"labels": split.labels, "test_size": len(split.test), "rows": rows,
            "artifact": str(ctx.settings.ml_dir / ARTIFACT_NAME),
            "note": "Classification metrics on the identical held-out test split. LLM latency includes model "
                    "inference through Ollama; baseline latency is in-process CPU prediction."}
