from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import AppContext, current_user, get_ctx, get_db
from app.benchmark.aggregate import compare_models, result_dict, run_dict, summarize, to_csv, to_markdown
from app.benchmark.evaluators import EVALUATOR_VERSIONS, evaluate_spec, spec_version, validate_spec
from app.benchmark.runner import BenchmarkRequest
from app.core.errors import ConflictError, ValidationFailed
from app.models.benchmark import BenchmarkResult, BenchmarkRun
from app.models.core import User

router = APIRouter(prefix="/api/benchmarks", tags=["benchmarks"])


class SuiteSave(BaseModel):
    suite: dict
    note: str | None = Field(None, max_length=300)


class EvaluatorTry(BaseModel):
    response: str = Field(max_length=20000)
    evaluator: dict


@router.get("/suites")
def list_suites(ctx: AppContext = Depends(get_ctx), user: User = Depends(current_user)) -> dict:
    return {"suites": [s.summary() for s in ctx.suites.all(user.id).values()]}


@router.get("/suites/{suite_id}")
def get_suite(suite_id: str, version: str | None = None, ctx: AppContext = Depends(get_ctx),
              user: User = Depends(current_user)) -> dict:
    s = ctx.suites.get(suite_id, user.id, version)
    return s.summary() | {"system": s.system, "default_max_tokens": s.default_max_tokens,
                          "cases": [c.model_dump() for c in s.cases]}


@router.post("/suites", status_code=201)
def save_suite(body: SuiteSave, ctx: AppContext = Depends(get_ctx), user: User = Depends(current_user)) -> dict:
    """Create a user suite or save a new version of it. Every version is kept."""
    return ctx.suites.save(user.id, body.suite, body.note).summary()


@router.get("/suites/{suite_id}/versions")
def suite_versions(suite_id: str, ctx: AppContext = Depends(get_ctx), user: User = Depends(current_user)) -> dict:
    return {"suite_id": suite_id, "versions": ctx.suites.versions(user.id, suite_id)}


@router.delete("/suites/{suite_id}", status_code=204)
def delete_suite(suite_id: str, ctx: AppContext = Depends(get_ctx), user: User = Depends(current_user)) -> None:
    ctx.suites.delete(user.id, suite_id)


@router.post("/evaluators/try")
def try_evaluator(body: EvaluatorTry, _: User = Depends(current_user)) -> dict:
    """Check an evaluator spec against a sample response before saving it in a suite."""
    try:
        validate_spec(body.evaluator)
        result = evaluate_spec(body.response, body.evaluator)
    except ValueError as exc:
        raise ValidationFailed(f"Invalid evaluator: {exc}") from exc
    return {**result.to_dict(), "evaluator_version": spec_version(body.evaluator)}


@router.get("/evaluators")
def list_evaluators(_: User = Depends(current_user)) -> dict:
    return {"evaluators": [{"type": k, "version": v} for k, v in EVALUATOR_VERSIONS.items()]}


@router.get("/runs")
def list_runs(limit: int = 100, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    runs = db.scalars(select(BenchmarkRun).where(BenchmarkRun.owner_id == user.id)
                      .order_by(BenchmarkRun.created_at.desc()).limit(min(limit, 500))).all()
    out = []
    for run in runs:
        results = db.scalars(select(BenchmarkResult).where(BenchmarkResult.run_id == run.id)).all()
        out.append(run_dict(run) | {"summary": summarize(results)})
    return {"runs": out}


@router.get("/overview")
def overview(user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    """Dashboard numbers: totals, best score, fastest model and score trend."""
    runs = db.scalars(select(BenchmarkRun).where(BenchmarkRun.owner_id == user.id, BenchmarkRun.status == "completed")
                      .order_by(BenchmarkRun.finished_at)).all()
    trend, best, fastest, total_s = [], None, None, 0.0
    for run in runs:
        results = db.scalars(select(BenchmarkResult).where(BenchmarkResult.run_id == run.id)).all()
        if run.started_at and run.finished_at:
            total_s += (run.finished_at - run.started_at).total_seconds()
        for s in summarize(results):
            point = {"run_id": run.id, "suite_id": run.suite_id, "model": s["model"],
                     "finished_at": run.finished_at.isoformat() if run.finished_at else None,
                     "quality_score": s["quality_score"], "tokens_per_s": s["tokens_per_s_mean"]}
            trend.append(point)
            if s["quality_score"] is not None and (best is None or s["quality_score"] > best["quality_score"]):
                best = point
            if s["tokens_per_s_mean"] is not None and (fastest is None or s["tokens_per_s_mean"] > fastest["tokens_per_s"]):
                fastest = point
    total_runs = db.query(BenchmarkRun).filter(BenchmarkRun.owner_id == user.id).count()
    return {"total_runs": total_runs, "completed_runs": len(runs), "best": best, "fastest": fastest,
            "total_bench_time_s": round(total_s, 1), "trend": trend[-200:]}


@router.post("/runs", status_code=201)
def create_run(body: BenchmarkRequest, user: User = Depends(current_user), ctx: AppContext = Depends(get_ctx)) -> dict:
    return run_dict(ctx.benchmarks.submit(user.id, body))


@router.get("/runs/{run_id}")
def get_run(run_id: int, include_responses: bool = True, user: User = Depends(current_user),
            db: Session = Depends(get_db), ctx: AppContext = Depends(get_ctx)) -> dict:
    run = ctx.benchmarks.run_owned(db, user.id, run_id)
    results = db.scalars(select(BenchmarkResult).where(BenchmarkResult.run_id == run.id)
                         .order_by(BenchmarkResult.id)).all()
    return run_dict(run) | {"summary": summarize(results), "comparison": compare_models(results),
                            "results": [result_dict(r, include_responses) for r in results]}


@router.post("/runs/{run_id}/cancel")
def cancel_run(run_id: int, user: User = Depends(current_user), ctx: AppContext = Depends(get_ctx)) -> dict:
    ctx.benchmarks.cancel(user.id, run_id)
    return {"status": "cancelling"}


@router.delete("/runs/{run_id}", status_code=204)
def delete_run(run_id: int, user: User = Depends(current_user), db: Session = Depends(get_db),
               ctx: AppContext = Depends(get_ctx)) -> None:
    run = ctx.benchmarks.run_owned(db, user.id, run_id)
    if run.status in ("queued", "running"):
        raise ConflictError("Cancel the run before deleting it.")
    db.query(BenchmarkResult).filter(BenchmarkResult.run_id == run.id).delete()
    db.delete(run)
    db.commit()


@router.get("/runs/{run_id}/export")
def export_run(run_id: int, format: str = "json", user: User = Depends(current_user), db: Session = Depends(get_db),
               ctx: AppContext = Depends(get_ctx)):
    run = ctx.benchmarks.run_owned(db, user.id, run_id)
    results = db.scalars(select(BenchmarkResult).where(BenchmarkResult.run_id == run.id)
                         .order_by(BenchmarkResult.id)).all()
    summary = summarize(results)
    fname = f"benchmark_run_{run.id}"
    if format == "csv":
        return Response(to_csv(results), media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="{fname}.csv"'})
    if format in ("md", "markdown"):
        return PlainTextResponse(to_markdown(run, summary, comparison=compare_models(results)),
                                 media_type="text/markdown",
                                 headers={"Content-Disposition": f'attachment; filename="{fname}.md"'})
    import json

    payload = run_dict(run, with_snapshot=True) | {"summary": summary, "comparison": compare_models(results),
                               "results": [result_dict(r) for r in results]}
    return Response(json.dumps(payload, indent=2, default=str), media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="{fname}.json"'})
