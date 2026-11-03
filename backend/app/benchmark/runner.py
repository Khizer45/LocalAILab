"""Benchmark execution.

Runs are executed one at a time by a single background worker thread so that
latency and memory measurements are not distorted by concurrent runs. Each
(model × test case × repeat) produces one stored :class:`BenchmarkResult`.

Reproducibility (v1.2): the exact cases, prompts, references and evaluator
specs are frozen into ``suite_snapshot`` when the run is submitted and the run
executes from that snapshot (editing a suite file later cannot change it).
Every result stores its evaluator version and execution order; the run stores
whether it was produced by the Ollama simulator and an integrity record
(expected vs stored results).

Order effects: by default models run one after another in the order given.
``model_order="shuffled"`` / ``case_order="shuffled"`` randomise the order with
the run's seed, and ``interleave=True`` alternates models case by case. On a
computer that cannot keep all models in memory, interleaving makes the runtime
reload models constantly, so load time dominates latency – use it only when
there is enough RAM/VRAM for every model.
"""

from __future__ import annotations

import hashlib
import json
import logging
import queue
import random
import threading
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app import __version__
from app.benchmark.aggregate import summarize
from app.benchmark.datasets import BenchmarkCase, BenchmarkSuite, SuiteRepository
from app.benchmark.evaluators import EVALUATOR_VERSIONS, evaluate_spec, spec_version
from app.core.config import Settings
from app.core.errors import AppError, ConflictError, InferenceTimeout, NotFoundError, ValidationFailed
from app.core.jobs import JobLock
from app.db.base import utcnow
from app.ml.tracking import Tracker
from app.models.benchmark import BenchmarkResult, BenchmarkRun
from app.services.inference import GenerationResult, clamp_options, stream_generate, strip_think
from app.services.model_registry import ModelRegistry, ResolvedModel
from app.services.settings_service import SettingsService
from app.services.system_monitor import ResourceSampler, snapshot

log = logging.getLogger("app.benchmark")


class BenchmarkRequest(BaseModel):
    name: str | None = Field(None, max_length=200)
    suite_id: str
    models: list[str] = Field(min_length=1, max_length=8)
    categories: list[str] | None = None
    max_cases: int | None = Field(None, ge=1, le=1000)
    repeats: int = Field(1, ge=1, le=10)
    warmup: bool = True
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    seed: int | None = 42
    num_ctx: int | None = Field(2048, ge=256, le=262144)
    max_tokens: int | None = Field(None, ge=1, le=8192)
    timeout_s: int | None = Field(None, ge=5, le=3600)
    think: bool | None = None
    case_ids: list[str] | None = Field(None, max_length=5000)  # exact cases (e.g. re-running a bundle)
    model_order: Literal["as_given", "shuffled"] = "as_given"
    case_order: Literal["suite", "shuffled"] = "suite"
    interleave: bool = False


@dataclass
class _Job:
    run_id: int
    cancel: threading.Event


class BenchmarkManager:
    def __init__(self, settings: Settings, SessionLocal: sessionmaker[Session], settings_service: SettingsService,
                 registry: ModelRegistry, suites: SuiteRepository, tracker: Tracker, job_lock: JobLock | None = None):
        self.settings = settings
        self.SessionLocal = SessionLocal
        self.settings_service = settings_service
        self.registry = registry
        self.suites = suites
        self.tracker = tracker
        self.job_lock = job_lock or JobLock()
        self._queue: queue.Queue[_Job] = queue.Queue()
        self._jobs: dict[int, _Job] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        with self.SessionLocal() as db:
            for run in db.scalars(select(BenchmarkRun).where(BenchmarkRun.status.in_(["queued", "running"]))):
                run.status, run.error = "failed", "Interrupted: the server stopped before this run finished."
                run.finished_at = utcnow()
            db.commit()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="benchmark-worker")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        for job in self._jobs.values():
            job.cancel.set()

    # ------------------------------------------------------------ API
    def select_cases(self, suite: BenchmarkSuite, req: BenchmarkRequest) -> list[BenchmarkCase]:
        cases = suite.cases
        if req.case_ids:
            wanted = set(req.case_ids)
            cases = [c for c in cases if c.id in wanted]
            missing = wanted - {c.id for c in cases}
            if missing:
                raise ValidationFailed(f"Cases not in the suite: {', '.join(sorted(missing)[:5])}.")
        if req.categories:
            cases = [c for c in cases if c.category in set(req.categories)]
        if req.max_cases:
            cases = cases[: req.max_cases]
        if not cases:
            raise ValidationFailed("No test cases match the selected categories.")
        return cases

    def submit(self, owner_id: int, req: BenchmarkRequest) -> BenchmarkRun:
        suite = self.suites.get(req.suite_id, owner_id)
        cases = self.select_cases(suite, req)
        models = list(dict.fromkeys(m.strip() for m in req.models if m.strip()))
        for m in models:  # validate up-front so the user gets a clear error now
            self.registry.resolve(m)
        rs = self.settings_service.get()
        config = req.model_dump()
        config["timeout_s"] = req.timeout_s or rs.benchmark_timeout_s
        config["case_ids"] = [c.id for c in cases]
        config["evaluator_versions"] = dict(EVALUATOR_VERSIONS)
        snapshot = suite_snapshot(suite, cases)
        config["cases_sha256"] = snapshot["cases_sha256"]
        with self.SessionLocal() as db:
            run = BenchmarkRun(
                owner_id=owner_id, name=req.name or f"{suite.name} ({', '.join(models)})"[:200],
                suite_id=suite.id, suite_version=suite.version, suite_hash=suite.sha256, models=models, config=config,
                status="queued", progress_total=len(models) * len(cases) * req.repeats, suite_snapshot=snapshot,
                simulated=self._is_simulated(models),
            )
            db.add(run)
            db.commit()
            db.refresh(run)
        job = _Job(run.id, threading.Event())
        self._jobs[run.id] = job
        self._queue.put(job)
        log.info("benchmark queued", extra={"run_id": run.id, "models": len(models), "cases": len(cases)})
        return run

    def cancel(self, owner_id: int, run_id: int) -> None:
        with self.SessionLocal() as db:
            run = db.get(BenchmarkRun, run_id)
            if run is None or run.owner_id != owner_id:
                raise NotFoundError("Benchmark run not found.")
            if run.status not in ("queued", "running"):
                raise ConflictError(f"Run is already {run.status}.")
            if run.status == "queued":
                run.status, run.finished_at = "cancelled", utcnow()
                db.commit()
        job = self._jobs.get(run_id)
        if job:
            job.cancel.set()

    def is_busy(self) -> bool:
        return any(not j.cancel.is_set() for j in self._jobs.values())

    # ------------------------------------------------------------ worker
    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if self.job_lock.acquire(f"benchmark run {job.run_id}", self._stop):
                    try:
                        self._execute(job)
                    finally:
                        self.job_lock.release()
            except Exception:  # noqa: BLE001
                log.exception("benchmark worker crashed", extra={"run_id": job.run_id})
                with self.SessionLocal() as db:
                    run = db.get(BenchmarkRun, job.run_id)
                    if run and run.status == "running":
                        run.status, run.error, run.finished_at = "failed", "Internal error – see server logs.", utcnow()
                        db.commit()
            finally:
                self._jobs.pop(job.run_id, None)

    def _execute(self, job: _Job) -> None:
        with self.SessionLocal() as db:
            run = db.get(BenchmarkRun, job.run_id)
            if run is None or run.status != "queued":
                return
            run.status, run.started_at = "running", utcnow()
            env = snapshot(self.settings.data_dir)
            env["app_version"] = __version__
            try:
                env["ollama_version"] = self.registry.local.version()
            except AppError:
                env["ollama_version"] = None
            run.environment = env
            db.commit()
            config = dict(run.config)
            models = list(run.models)
            run_snapshot = dict(run.suite_snapshot) if run.suite_snapshot else None
        snap = run_snapshot or suite_snapshot(self.suites.get(config["suite_id"]),
                                              [c for c in self.suites.get(config["suite_id"]).cases
                                               if c.id in set(config["case_ids"])])
        suite = BenchmarkSuite(id=snap["id"], name=snap["name"], version=snap["version"], system=snap.get("system"),
                               default_max_tokens=snap["default_max_tokens"],
                               cases=[BenchmarkCase(**c) for c in snap["cases"]], sha256=snap.get("sha256", ""))
        cases = suite.cases
        rs = self.settings_service.get()
        model_info: dict[str, dict] = {}
        resolved_models: dict[str, ResolvedModel | None] = {}
        cancelled = False
        plan = execution_plan(models, [c.id for c in cases], int(config["repeats"]), config)
        by_id = {c.id: c for c in cases}

        for order_index, (model_ref, case_id, rep) in enumerate(plan):
            if job.cancel.is_set():
                cancelled = True
                break
            case = by_id[case_id]
            if model_ref not in resolved_models:
                try:
                    resolved = self.registry.resolve(model_ref)
                    info = {"provider": resolved.provider, "digest": resolved.digest,
                            "parameter_size": resolved.parameter_size, "quantization": resolved.quantization,
                            "first_order_index": order_index}
                    if config.get("warmup"):
                        info["warmup"] = self._warmup(resolved, config)
                    resolved_models[model_ref] = resolved
                except AppError as exc:
                    info = {"error": exc.message}
                    resolved_models[model_ref] = None
                model_info[model_ref] = info
                self._save_model_info(job.run_id, model_info)
            resolved = resolved_models[model_ref]
            if resolved is None:
                self._store(job.run_id, BenchmarkResult(
                    run_id=job.run_id, model=model_ref, case_id=case.id, category=case.category, repeat_index=rep,
                    status="error", error=model_info[model_ref]["error"], score=0.0, passed=False,
                    order_index=order_index, evaluator_version=spec_version(case.evaluator)))
                continue
            self._set_current(job.run_id, f"{model_ref} on {case.id}" + (f" (repeat {rep + 1})" if config["repeats"] > 1 else ""))
            result = self._run_case(job.run_id, resolved, suite, case, rep, config, rs.max_context_tokens,
                                    rs.max_output_tokens)
            result.order_index = order_index
            self._store(job.run_id, result)

        with self.SessionLocal() as db:
            run = db.get(BenchmarkRun, job.run_id)
            run.model_info = model_info
            run.current_item = "Saving results and experiment tracking…"
            run.finished_at = utcnow()
            results = list(db.scalars(select(BenchmarkResult).where(BenchmarkResult.run_id == run.id)))
            expected = len(plan)
            keys = [(r.model, r.case_id, r.repeat_index) for r in results]
            run.integrity = {"expected_results": expected, "stored_results": len(results),
                             "duplicates": len(keys) - len(set(keys)),
                             "missing": max(0, expected - len(set(keys))), "complete": len(set(keys)) == expected
                             and len(keys) == expected}
            if cancelled:
                run.status = "cancelled"
            elif not run.integrity["complete"]:
                run.status = "failed"
                run.error = (f"Integrity check failed: {len(results)} of {expected} results stored "
                             f"({run.integrity['duplicates']} duplicates). The run is not reported as complete.")
            else:
                run.status = "completed"
            run.mlflow_run_id = self._track(run, summarize(results))
            run.current_item = None
            db.commit()  # status becomes visible only once everything is stored
        log.info("benchmark finished", extra={"run_id": job.run_id, "status": "cancelled" if cancelled else "completed"})

    def _is_simulated(self, models: list[str]) -> bool:
        try:
            if "simulator" in (self.registry.local.version() or ""):
                return True
        except AppError:
            pass
        return any(m.split("::")[-1].startswith("sim-") for m in models)

    def _warmup(self, model: ResolvedModel, config: dict) -> dict:
        res = GenerationResult()
        try:
            for _ in stream_generate(model, [{"role": "user", "content": "Reply with OK."}],
                                     options={"num_predict": 4, "temperature": 0, "num_ctx": config.get("num_ctx") or 2048},
                                     timeout_s=float(config["timeout_s"]) * 2, result=res):
                pass
            return {"load_ms": res.metrics()["load_ms"], "total_ms": res.metrics()["total_ms"]}
        except AppError as exc:
            return {"error": exc.message}

    def _run_case(self, run_id: int, model: ResolvedModel, suite: BenchmarkSuite, case: BenchmarkCase, rep: int,
                  config: dict, max_ctx: int, max_out: int) -> BenchmarkResult:
        messages = []
        system = case.system or suite.system
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": case.prompt})
        options = clamp_options(
            {"temperature": config["temperature"], "seed": config.get("seed"), "num_ctx": config.get("num_ctx"),
             "num_predict": case.max_tokens or config.get("max_tokens") or suite.default_max_tokens},
            max_ctx=max_ctx, max_out=max(max_out, case.max_tokens or 0),
        )
        res = GenerationResult()
        status, error = "ok", None
        with ResourceSampler() as sampler:
            try:
                for _ in stream_generate(model, messages, options=options, timeout_s=float(config["timeout_s"]),
                                         think=config.get("think"), result=res):
                    pass
            except InferenceTimeout as exc:
                status, error = "timeout", exc.message
            except AppError as exc:
                status, error = "error", exc.message
        response = strip_think(res.content)
        if status == "ok":
            ev = evaluate_spec(response, case.evaluator)
            score, passed, details = ev.score, ev.passed, ev.to_dict()
        else:
            score, passed, details = 0.0, False, None
        model_mem = None
        if model.provider == "local":
            for m in self.registry.loaded_models():
                if m.get("name") == model.name or m.get("model") == model.name:
                    model_mem = round((m.get("size") or 0) / (1024 * 1024), 1) or None
        s = sampler.summary()
        mt = res.metrics()
        return BenchmarkResult(
            run_id=run_id, model=model.ref, case_id=case.id, category=case.category, repeat_index=rep, status=status,
            response=response[:20000], thinking_chars=len(res.thinking), score=score, passed=passed,
            eval_details=details, ttft_ms=mt["ttft_ms"], ttft_answer_ms=mt["ttft_answer_ms"], total_ms=mt["total_ms"],
            load_ms=mt["load_ms"], evaluator_version=spec_version(case.evaluator),
            prompt_tokens=mt["prompt_tokens"], output_tokens=mt["output_tokens"], tokens_per_s=mt["tokens_per_s"],
            tokens_source=mt["tokens_source"] if status == "ok" else "none",
            peak_ram_used_mb=s["peak_ram_used_mb"], avg_cpu_pct=s["avg_cpu_pct"], runtime_rss_mb=s["runtime_rss_mb"],
            model_mem_mb=model_mem, error=error,
        )

    # ------------------------------------------------------------ persistence helpers
    def _store(self, run_id: int, result: BenchmarkResult) -> None:
        with self.SessionLocal() as db:
            db.add(result)
            run = db.get(BenchmarkRun, run_id)
            run.progress_done = (run.progress_done or 0) + 1
            db.commit()

    def _set_current(self, run_id: int, text: str) -> None:
        with self.SessionLocal() as db:
            run = db.get(BenchmarkRun, run_id)
            run.current_item = text[:300]
            db.commit()

    def _save_model_info(self, run_id: int, info: dict) -> None:
        with self.SessionLocal() as db:
            run = db.get(BenchmarkRun, run_id)
            run.model_info = dict(info)
            db.commit()

    def _track(self, run: BenchmarkRun, summary: list[dict]) -> str | None:
        metrics: dict[str, float] = {}
        for s in summary:
            safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in s["model"])
            for key in ("quality_score", "pass_rate", "ttft_ms_p50", "latency_ms_p50", "latency_ms_p95",
                        "tokens_per_s_mean", "peak_ram_used_mb", "errors", "timeouts"):
                if s.get(key) is not None:
                    metrics[f"{safe}.{key}"] = s[key]
        params = {"suite_id": run.suite_id, "suite_version": run.suite_version, "suite_sha256": run.suite_hash,
                  "models": ",".join(run.models), **{k: v for k, v in run.config.items() if k != "case_ids"},
                  "case_count": len(run.config.get("case_ids", []))}
        return self.tracker.log_run(
            "local-ai-lab-benchmarks", f"run-{run.id}", params, metrics,
            {"summary.json": summary, "environment.json": run.environment, "model_info.json": run.model_info},
            tags={"status": run.status, "app_version": __version__},
        )

    def run_owned(self, db: Session, owner_id: int, run_id: int) -> BenchmarkRun:
        run = db.get(BenchmarkRun, run_id)
        if run is None or run.owner_id != owner_id:
            raise NotFoundError("Benchmark run not found.")
        return run


# ----------------------------------------------------------------------------- helpers
def suite_snapshot(suite: BenchmarkSuite, cases: list[BenchmarkCase]) -> dict:
    """The exact suite content a run uses, with a hash of the selected cases."""
    case_dicts = [c.model_dump() for c in cases]
    canon = json.dumps({"system": suite.system, "default_max_tokens": suite.default_max_tokens, "cases": case_dicts},
                       sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return {"id": suite.id, "name": suite.name, "version": suite.version, "sha256": suite.sha256,
            "system": suite.system, "default_max_tokens": suite.default_max_tokens, "cases": case_dicts,
            "cases_sha256": hashlib.sha256(canon.encode("utf-8")).hexdigest()}


def execution_plan(models: list[str], case_ids: list[str], repeats: int, config: dict) -> list[tuple[str, str, int]]:
    """(model, case, repeat) in execution order according to the run's order options."""
    rng = random.Random(config.get("seed") if config.get("seed") is not None else 0)  # noqa: S311 - not security
    models = list(models)
    if config.get("model_order") == "shuffled":
        rng.shuffle(models)
    order = list(case_ids)
    if config.get("case_order") == "shuffled":
        rng.shuffle(order)
    if config.get("interleave"):
        return [(m, c, r) for c in order for r in range(repeats) for m in models]
    return [(m, c, r) for m in models for c in order for r in range(repeats)]
