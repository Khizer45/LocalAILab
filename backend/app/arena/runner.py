"""Model Arena runs: train / evaluate every contender on the same test split.

Contenders
* ``baseline:dummy``                 – majority class / mean (always included)
* ``builtin:<algorithm>``            – trained here on the training split
* ``builtin:<algorithm>@<extractor>`` – image datasets: classifier on top of a
  feature extractor (``raw_pixels``, ``backbone:mobilenetv2``, ``model:<id>``)
* ``uploaded:<id>``                  – the user's own trained model (ONNX / joblib)
* ``llm:<model>``                    – zero-shot classification by a local LLM (text)

Protocol v2 (see ``app.arena.stats``): built-ins are fitted on the training
split (tuning = cross-validation inside it); every contender predicts the
validation and the test split; the recommendation is chosen on **validation**
only (constraints, equivalence margin, Holm-adjusted paired bootstrap
comparisons); the chosen model's **test** score is then reported as the final,
unbiased estimate. Test scores of the other models are shown for information.

For each contender the runner also records cross-validation on the training
split (built-ins), training time, batch and single-example latency (measured on
validation inputs), throughput, model size, peak memory, raw per-example
predictions for both splits and the library versions.
"""

from __future__ import annotations

import json
import logging
import pickle
import queue
import re
import shutil
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import psutil
from pydantic import BaseModel, Field
from sklearn.base import clone
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app import __version__
from app.arena import images as IMG
from app.arena.algorithms import ALGORITHMS, build_pipeline, param_space
from app.arena.service import ArenaStore, LoadedDataset, provenance_caveats, take
from app.arena.stats import (
    METRICS,
    PRIMARY,
    VERDICT_TEXT,
    Policy,
    bootstrap_cis,
    classification_metrics,
    compare,
    make_resampling,
    regression_metrics,
)
from app.arena.uploaded import check_compatible
from app.benchmark.evaluators import label as parse_label
from app.core.config import Settings
from app.core.errors import AppError, ConflictError, NotFoundError, ValidationFailed
from app.core.jobs import JobLock
from app.db.base import utcnow
from app.ml.tracking import Tracker
from app.models.arena import ArenaDataset, ArenaModel, ArenaResult, ArenaRun
from app.services.inference import GenerationResult, stream_generate, strip_think
from app.services.model_registry import ModelRegistry
from app.services.settings_service import SettingsService
from app.services.system_monitor import snapshot

log = logging.getLogger("app.arena")

LLM_PROMPT = ("Classify the text into exactly one of these labels: {labels}.\n"
              "Text: \"{text}\"\n"
              "Answer with the label only.")
MAX_CONTENDERS = 40
SINGLE_SAMPLES = 25


class ArenaRunRequest(BaseModel):
    dataset_id: int
    name: str | None = Field(None, max_length=200)
    algorithms: list[str] = Field(default_factory=list)
    uploaded_models: list[int] = Field(default_factory=list)
    llm_models: list[str] = Field(default_factory=list)
    image_features: list[str] = Field(default_factory=list)
    include_baseline: bool = True
    cv_folds: int = Field(5, ge=0, le=10)
    tune: bool = False
    tune_iterations: int = Field(10, ge=2, le=50)
    llm_max_examples: int = Field(100, ge=5, le=5000)
    seed: int = 42
    # --- recommendation policy (protocol v2) ---------------------------------------------------
    primary_metric: str | None = None  # default: f1_macro (classification) / r2 (regression)
    objective: Literal["efficiency", "quality"] = "efficiency"
    equivalence_margin: float | None = Field(None, ge=0, le=0.5)
    min_quality: float | None = None
    max_latency_ms: float | None = Field(None, gt=0)
    max_size_mb: float | None = Field(None, gt=0)
    max_memory_mb: float | None = Field(None, gt=0)
    bootstrap_resamples: int = Field(2000, ge=200, le=10000)
    bootstrap: Literal["auto", "simple", "stratified", "cluster"] = "auto"
    add_validation_split: bool = True  # v1.1 datasets: carve a validation split from the training rows


@dataclass
class _Job:
    run_id: int
    cancel: threading.Event


class _Cancelled(Exception):
    pass


class _PeakRSS:
    """Peak resident memory of this process while a contender trains/predicts."""

    def __init__(self) -> None:
        self.proc = psutil.Process()
        self.base = self.peak = self.proc.memory_info().rss
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(0.1):
            try:
                self.peak = max(self.peak, self.proc.memory_info().rss)
            except psutil.Error:
                return

    def __enter__(self) -> _PeakRSS:
        self._t.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._t.join(timeout=1)
        self.peak = max(self.peak, self.proc.memory_info().rss)

    def summary(self) -> dict:
        mb = 1024 * 1024
        return {"peak_process_rss_mb": round(self.peak / mb, 1), "memory_increase_mb": round((self.peak - self.base) / mb, 1)}


def extractor_label(key: str, store_models: dict[int, ArenaModel] | None = None) -> str:
    if key == "raw_pixels":
        return "raw pixels"
    if key.startswith("backbone:"):
        return IMG.BACKBONES.get(key.split(":", 1)[1], {}).get("name", key).split(" (")[0] + " features"
    if key.startswith("model:") and store_models:
        m = store_models.get(int(key.split(":", 1)[1]))
        return f"{m.name} features" if m else key
    return key


class ArenaManager:
    def __init__(self, settings: Settings, SessionLocal: sessionmaker[Session], store: ArenaStore,
                 registry: ModelRegistry, settings_service: SettingsService, tracker: Tracker, job_lock: JobLock):
        self.settings = settings
        self.SessionLocal = SessionLocal
        self.store = store
        self.registry = registry
        self.settings_service = settings_service
        self.tracker = tracker
        self.job_lock = job_lock
        self._queue: queue.Queue[_Job] = queue.Queue()
        self._jobs: dict[int, _Job] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        with self.SessionLocal() as db:
            for run in db.scalars(select(ArenaRun).where(ArenaRun.status.in_(["queued", "running"]))):
                run.status, run.error, run.finished_at = "failed", "Interrupted: the server stopped first.", utcnow()
            db.commit()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="arena-worker")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        for job in self._jobs.values():
            job.cancel.set()

    # ------------------------------------------------------------ submit / cancel
    def plan(self, db: Session, owner_id: int, req: ArenaRunRequest) -> tuple[ArenaDataset, list[dict]]:
        ds = self.store.dataset_owned(db, owner_id, req.dataset_id)
        if req.primary_metric is not None and req.primary_metric not in METRICS[ds.task]:
            raise ValidationFailed(f"'{req.primary_metric}' is not a {ds.task} metric.",
                                   hint="Choose one of: " + ", ".join(METRICS[ds.task]))
        if req.bootstrap == "cluster" and not ds.group_column:
            raise ValidationFailed("Cluster bootstrap needs a dataset with a group column (group split).")
        contenders: list[dict] = []
        if req.include_baseline:
            contenders.append({"key": "baseline:dummy", "source": "baseline", "algorithm": "dummy",
                               "name": ALGORITHMS["dummy"].name})
        extractors: list[str] = []
        fx_models: dict[int, ArenaModel] = {}
        if ds.kind == "image":
            extractors = list(dict.fromkeys(req.image_features or ["raw_pixels"]))
            for key in extractors:
                if key == "raw_pixels":
                    continue
                if key.startswith("backbone:"):
                    bid = key.split(":", 1)[1]
                    if bid not in IMG.BACKBONES:
                        raise ValidationFailed(f"Unknown backbone '{bid}'.")
                    if not IMG.backbone_path(self.settings.arena_dir, bid).exists():
                        raise ValidationFailed(f"{IMG.BACKBONES[bid]['name']} is not downloaded yet.",
                                               hint="Download it under Model Arena → Models → Feature extractors, or "
                                                    "run scripts\\download_backbones.bat.")
                elif key.startswith("model:"):
                    m = self.store.model_owned(db, owner_id, int(key.split(":", 1)[1]))
                    if m.role != "feature_extractor":
                        raise ValidationFailed(f"'{m.name}' was not uploaded as a feature extractor.")
                    fx_models[m.id] = m
                else:
                    raise ValidationFailed(f"Unknown feature extractor '{key}'.")
        elif req.image_features:
            raise ValidationFailed("Feature extractors only apply to image datasets.")
        for algo_id in dict.fromkeys(req.algorithms):
            algo = ALGORITHMS.get(algo_id)
            if algo is None:
                raise ValidationFailed(f"Unknown algorithm '{algo_id}'.")
            if algo_id == "dummy":
                continue
            if not algo.supports(ds.kind, ds.task):
                raise ValidationFailed(f"{algo.name} does not support {ds.kind} {ds.task}.")
            if not algo.available():
                raise ValidationFailed(f"{algo.name} needs the '{algo.requires}' package.",
                                       hint=f"Install it: pip install {algo.requires}")
            if ds.kind == "image":
                for ex in extractors:
                    contenders.append({"key": f"builtin:{algo_id}@{ex}", "source": "builtin", "algorithm": algo_id,
                                       "extractor": ex,
                                       "name": f"{algo.name} · {extractor_label(ex, fx_models)}"})
            else:
                contenders.append({"key": f"builtin:{algo_id}", "source": "builtin", "algorithm": algo_id,
                                   "name": algo.name})
        for mid in dict.fromkeys(req.uploaded_models):
            m = self.store.model_owned(db, owner_id, mid)
            if m.role == "feature_extractor":
                raise ValidationFailed(f"'{m.name}' is a feature extractor – pick it under image features instead.")
            reason = check_compatible(m.kind, m.task, m.role, ds.kind, ds.task)
            if reason:
                raise ValidationFailed(f"'{m.name}' cannot be used with this dataset: {reason}.")
            contenders.append({"key": f"uploaded:{m.id}", "source": "uploaded", "model_id": m.id,
                               "name": f"{m.name} (your {m.format.upper()} model)",
                               "provenance": m.provenance or {"training_data": "unknown"}})
        if req.llm_models:
            if not (ds.kind == "text" and ds.task == "classification"):
                raise ValidationFailed("Local LLMs can only be compared on text classification datasets.")
            for ref in dict.fromkeys(req.llm_models):
                self.registry.resolve(ref)
                contenders.append({"key": f"llm:{ref}", "source": "llm", "model": ref, "name": f"{ref} (LLM zero-shot)"})
        real = [c for c in contenders if c["source"] != "baseline"]
        if not real:
            raise ValidationFailed("Pick at least one algorithm, uploaded model or LLM to compare.")
        if len(contenders) > MAX_CONTENDERS:
            raise ValidationFailed(f"Too many contenders ({len(contenders)}); the limit is {MAX_CONTENDERS}.")
        return ds, contenders

    def submit(self, owner_id: int, req: ArenaRunRequest) -> ArenaRun:
        with self.SessionLocal() as db:
            ds, contenders = self.plan(db, owner_id, req)
            if req.add_validation_split and not ds.n_val:
                self.store.ensure_validation(ds)
                db.refresh(ds)
            info = {"id": ds.id, "name": ds.name, "kind": ds.kind, "task": ds.task, "labels": ds.labels,
                    "n_rows": ds.n_rows, "n_train": ds.n_train, "n_val": ds.n_val, "n_test": ds.n_test,
                    "sha256": ds.sha256, "seed": ds.seed, "test_size": ds.test_size, "val_size": ds.val_size,
                    "split_source": ds.split_source, "split_sha256": ds.split_sha256,
                    "group_column": ds.group_column, "time_column": ds.time_column,
                    "filename": ds.filename, "target": ds.target}
            extract_steps = len({c.get("extractor") for c in contenders if c.get("extractor")})
            run = ArenaRun(owner_id=owner_id, dataset_id=ds.id, dataset_info=info, config=req.model_dump(),
                           contenders=contenders, status="queued", protocol="v2",
                           name=(req.name or f"{ds.name}: {len(contenders)} models")[:200],
                           progress_total=len(contenders) + extract_steps)
            db.add(run)
            db.commit()
            db.refresh(run)
        job = _Job(run.id, threading.Event())
        self._jobs[run.id] = job
        self._queue.put(job)
        return run

    def cancel(self, owner_id: int, run_id: int) -> None:
        with self.SessionLocal() as db:
            run = self.run_owned(db, owner_id, run_id)
            if run.status not in ("queued", "running"):
                raise ConflictError(f"Run is already {run.status}.")
            if run.status == "queued":
                run.status, run.finished_at = "cancelled", utcnow()
                db.commit()
        job = self._jobs.get(run_id)
        if job:
            job.cancel.set()

    def run_owned(self, db: Session, owner_id: int, run_id: int) -> ArenaRun:
        run = db.get(ArenaRun, run_id)
        if run is None or run.owner_id != owner_id:
            raise NotFoundError("Arena run not found.")
        return run

    # ------------------------------------------------------------ worker
    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if self.job_lock.acquire(f"arena run {job.run_id}", self._stop):
                    try:
                        self._execute(job)
                    finally:
                        self.job_lock.release()
            except Exception as exc:  # noqa: BLE001
                log.exception("arena worker crashed", extra={"run_id": job.run_id})
                self._fail(job.run_id, f"Internal error ({type(exc).__name__}) – see server logs.")
            finally:
                self._jobs.pop(job.run_id, None)

    def _fail(self, run_id: int, message: str) -> None:
        with self.SessionLocal() as db:
            run = db.get(ArenaRun, run_id)
            if run and run.status in ("queued", "running"):
                run.status, run.error, run.finished_at, run.current_item = "failed", message, utcnow(), None
                db.commit()

    def _update(self, run_id: int, **fields) -> None:
        with self.SessionLocal() as db:
            run = db.get(ArenaRun, run_id)
            for k, v in fields.items():
                setattr(run, k, v)
            db.commit()

    def _progress(self, run_id: int) -> None:
        with self.SessionLocal() as db:
            run = db.get(ArenaRun, run_id)
            run.progress_done = (run.progress_done or 0) + 1
            db.commit()

    def _execute(self, job: _Job) -> None:
        with self.SessionLocal() as db:
            run = db.get(ArenaRun, job.run_id)
            if run is None or run.status != "queued":
                return
            if run.dataset_id is None:
                run.status, run.error, run.finished_at = "failed", "The dataset was deleted.", utcnow()
                db.commit()
                return
            ds = db.get(ArenaDataset, run.dataset_id)
            run.status, run.started_at = "running", utcnow()
            env = snapshot(self.settings.data_dir)
            env["app_version"] = __version__
            env["libraries"] = _library_versions()
            run.environment = env
            run.current_item = "Loading the dataset…"
            db.commit()
            config, contenders, owner_id = dict(run.config), list(run.contenders), run.owner_id
            db.expunge(ds)
            models = {m.id: m for m in db.scalars(select(ArenaModel).where(ArenaModel.owner_id == owner_id)).all()}
            for m in models.values():
                db.expunge(m)

        data = self.store.load(ds)
        seed = int(config.get("seed", 42))
        labels = data.labels
        index = {lab: i for i, lab in enumerate(labels)}
        if data.task == "classification":
            y_enc = np.array([index[v] for v in data.y])
        else:
            y_enc = data.y.astype(float)
        has_val = len(data.val_idx) > 0
        run_dir = self.settings.arena_dir / "runs" / str(job.run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        features: dict[str, dict] = {}
        evaluated: list[dict] = []
        cancelled = False
        n_boot = int(config.get("bootstrap_resamples") or 2000)

        def resampling(rows: np.ndarray):
            groups = data.groups[rows] if data.groups is not None else None
            return make_resampling(data.task, y_enc[rows], groups, n_boot, seed, config.get("bootstrap") or "auto")

        rs_split = {"test": resampling(data.test_idx), "val": resampling(data.val_idx) if has_val else None}

        for c in contenders:
            if job.cancel.is_set():
                cancelled = True
                break
            self._update(job.run_id, current_item=f"{c['name']}")
            t0 = time.perf_counter()
            try:
                if c["source"] in ("baseline", "builtin"):
                    feat = None
                    if c.get("extractor"):
                        feat = features.get(c["extractor"])
                        if feat is None:
                            try:
                                feat = self._extract(job, ds, data, c["extractor"], models)
                            except AppError as exc:
                                feat = {"error": exc}
                            features[c["extractor"]] = feat
                            self._progress(job.run_id)
                        if "error" in feat:
                            raise feat["error"]
                    elif data.kind == "image":  # the majority-class baseline ignores the inputs
                        feat = {"X": np.zeros((len(data.y), 1), dtype=np.float32), "label": "none", "dim": 1,
                                "ms_per_image": 0.0, "ms_single_p50": 0.0}
                    out = self._eval_builtin(c, data, feat, config, seed, run_dir)
                elif c["source"] == "uploaded":
                    out = self._eval_uploaded(c, data, models.get(c["model_id"]))
                else:
                    out = self._eval_llm(job, c, data, config, seed)
            except _Cancelled:
                cancelled = True
                break
            except AppError as exc:
                out = {"status": "error", "error": exc.message + (f" {exc.hint}" if exc.hint else "")}
            except Exception as exc:  # noqa: BLE001
                log.exception("arena contender failed", extra={"run_id": job.run_id, "contender": c["key"]})
                out = {"status": "error", "error": f"{type(exc).__name__}: {str(exc)[:400]}"}
            out.setdefault("details", {})["wall_s"] = round(time.perf_counter() - t0, 2)
            if out["status"] == "ok":
                caveats = []
                if c["source"] == "uploaded":
                    caveats = provenance_caveats(c.get("provenance"), ds.sha256, data.split_sha256, has_val)
                    out["details"]["provenance"] = c.get("provenance")
                out["details"]["caveats"] = caveats
                perf = out.pop("perf")
                per_split = {}
                for part in ("test", "val"):
                    block = out.pop(part, None)
                    if block is None:
                        continue
                    rows = data.test_idx if part == "test" else data.val_idx
                    pred = np.asarray(block["pred"])
                    idx = np.asarray(block["idx"], dtype=int)
                    yt = y_enc[rows][idx]
                    rs = rs_split[part]
                    sub_rs = _subset_resampling(rs, idx)
                    if data.task == "classification":
                        m = classification_metrics(yt, pred, labels, block.get("scores"))
                        m.update(bootstrap_cis("classification", yt, pred, len(labels), sub_rs))
                        if part == "test":
                            out["details"]["errors_sample"] = _error_examples(data, rows, idx, yt, pred, labels)
                    else:
                        m = regression_metrics(yt, pred)
                        m.update(bootstrap_cis("regression", yt, pred, 0, sub_rs))
                        if part == "test":
                            out["details"]["errors_sample"] = _regression_examples(data, rows, idx, yt, pred)
                    per_split[part] = {"metrics": m, "pred": pred, "idx": idx}
                out["metrics"] = {**per_split["test"]["metrics"], **perf}
                out["predictions"] = per_split["test"]["pred"].tolist()
                out["evaluated_idx"] = per_split["test"]["idx"].tolist()
                if "val" in per_split:
                    out["val_metrics"] = per_split["val"]["metrics"]
                    out["val_predictions"] = per_split["val"]["pred"].tolist()
                    out["val_idx"] = per_split["val"]["idx"].tolist()
                evaluated.append({"key": c["key"], "name": c["name"], "source": c["source"], "splits": per_split,
                                  "latency_ms": perf.get("latency_ms_single_p50") or perf.get("latency_ms_per_example"),
                                  "size_bytes": perf.get("model_size_bytes"),
                                  "memory_mb": perf.get("peak_process_rss_mb"), "caveats": caveats})
            self._store_result(job.run_id, c, out)
            self._progress(job.run_id)

        self._update(job.run_id, current_item="Comparing models and saving results…")
        summary = self._summarize(data, config, evaluated, y_enc, rs_split)
        best_path = self._keep_best_model(run_dir, summary, evaluated)
        with self.SessionLocal() as db:
            run = db.get(ArenaRun, job.run_id)
            run.summary = summary
            run.best_model_path = best_path
            run.finished_at = utcnow()
            run.status = "cancelled" if cancelled else "completed"
            run.mlflow_run_id = self._track(run, evaluated, summary)
            run.current_item = None
            db.commit()
        log.info("arena run finished", extra={"run_id": job.run_id, "status": "cancelled" if cancelled else "completed"})

    def _summarize(self, data: LoadedDataset, config: dict, evaluated: list[dict], y_enc: np.ndarray,
                   rs_split: dict) -> dict:
        """Select on validation, then report the selected model's test score (protocol v2)."""
        task = data.task
        metric = config.get("primary_metric") or PRIMARY[task][0]
        label, higher = METRICS[task][metric]
        policy = Policy(primary_metric=metric, margin=config.get("equivalence_margin"),
                        objective=config.get("objective") or "efficiency", min_quality=config.get("min_quality"),
                        max_latency_ms=config.get("max_latency_ms"), max_size_mb=config.get("max_size_mb"),
                        max_memory_mb=config.get("max_memory_mb"))
        k = len(data.labels)
        has_val = len(data.val_idx) > 0
        sel_part = "val" if has_val else "test"
        sel_rows = data.val_idx if has_val else data.test_idx

        def cands(part: str) -> list[dict]:
            out = []
            for e in evaluated:
                block = e["splits"].get(part)
                if block is None:
                    continue
                caveats = list(e["caveats"])
                excluded = part == "val" and any("validation score is optimistic" in c for c in caveats)
                if excluded:
                    continue
                out.append({"key": e["key"], "name": e["name"], "source": e["source"], "pred": block["pred"],
                            "idx": block["idx"], "metrics": block["metrics"], "latency_ms": e["latency_ms"],
                            "size_bytes": e["size_bytes"], "memory_mb": e["memory_mb"], "caveats": caveats})
            return out

        selection = compare(task, cands(sel_part), y_enc[sel_rows], k, policy, rs_split[sel_part],
                            split="validation" if has_val else "test")
        test_view = compare(task, cands("test"), y_enc[data.test_idx], k, Policy(primary_metric=metric,
                            margin=selection.get("margin"), objective="quality"), rs_split["test"], split="test")
        rec = selection.get("recommendation")
        final = None
        if rec:
            e = next(x for x in evaluated if x["key"] == rec["key"])
            tm = e["splits"]["test"]["metrics"]
            final = {"key": rec["key"], "name": rec["name"], "split": "test", "metric": metric, "label": label,
                     "value": tm.get(metric), "ci": tm.get(f"{metric}_ci"), "n": tm.get("n"),
                     "unbiased": has_val,
                     "note": ("Chosen on the validation split; the test split was not used for the choice, so this "
                              "is an unbiased estimate for new data like the test rows." if has_val else
                              "Exploratory: there is no validation split, so the model was chosen on the same test "
                              "rows – this score is optimistic.")}
            if e["caveats"]:
                final["note"] += " " + " ".join(e["caveats"])
        test_leader = (test_view.get("ranking") or [None])
        test_leader = next((key for key in test_leader if not key.startswith("baseline:")), None)
        warnings_: list[str] = list(selection.get("warnings", []))
        if rec and test_leader and test_leader != rec["key"] and has_val:
            warnings_.append("A different model has the highest test score. That is expected noise between splits; "
                             "choosing it because of its test score would be biased (winner's bias). Use a "
                             "confirmation run on new data if the difference matters.")
        if not has_val:
            warnings_.append("No validation split: the model was selected on the test split (exploratory result).")
        limitations = [
            "Intervals cover uncertainty from the finite evaluation rows only – not from re-training with other "
            "seeds or from a different train/test split.",
            "A single split shows how models compare on this dataset; it does not show that the winner generalises "
            "to other datasets or to data that changes over time.",
            "Equivalence means the difference is smaller than the margin (" + _margin_text(selection) + "); "
            "choose a margin that matters for your use before reading the result.",
        ]
        return {
            "protocol": "v2", "task": task, "kind": data.kind, "labels": data.labels,
            "primary_metric": metric, "primary_label": label, "higher_is_better": higher,
            "split": {"n_train": int(len(data.train_idx)), "n_val": int(len(data.val_idx)),
                      "n_test": int(len(data.test_idx)), "sha256": data.split_sha256, "selection_split": sel_part},
            "test_size": int(len(data.test_idx)),
            "selection": selection, "ranking": selection.get("ranking", []), "recommendation": rec,
            "final": final, "test_view": {"ranking": test_view.get("ranking", []), "leader": test_leader,
                                          "comparisons": test_view.get("comparisons", {})},
            "pareto": selection.get("pareto", []), "warnings": warnings_, "limitations": limitations,
            "verdict_text": VERDICT_TEXT,
        }

    # ------------------------------------------------------------ contenders
    def _extract(self, job: _Job, ds: ArenaDataset, data: LoadedDataset, key: str, models: dict[int, ArenaModel]) -> dict:
        """Compute (or load cached) image features for every image of the dataset."""
        folder = self.settings.arena_dir / "features" / str(ds.id)
        folder.mkdir(parents=True, exist_ok=True)
        if key == "raw_pixels":
            fingerprint = "raw32-v1"
        elif key.startswith("backbone:"):
            fingerprint = IMG.BACKBONES[key.split(":", 1)[1]]["sha256"][:12]
        else:
            m = models[int(key.split(":", 1)[1])]
            fingerprint = m.sha256[:12]
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", key)
        npy, meta_p = folder / f"{safe}__{fingerprint}.npy", folder / f"{safe}__{fingerprint}.json"
        if npy.exists() and meta_p.exists():
            return {"X": np.load(npy), **json.loads(meta_p.read_text())}
        paths: list[Path] = data.X  # type: ignore[assignment]
        n = len(paths)
        if key == "raw_pixels":
            def fn(part):
                return IMG.raw_pixel_features(part)
        else:
            if key.startswith("backbone:"):
                bid = key.split(":", 1)[1]
                sess, feat = IMG.feature_session(IMG.backbone_path(self.settings.arena_dir, bid))
                pp = IMG.ImagePreprocess(**IMG.BACKBONES[bid]["preprocess"])
            else:
                m = models[int(key.split(":", 1)[1])]
                if self.settings.arena_sandbox:  # uploaded extractor: run it isolated, in one go
                    from app.arena.sandbox import run_isolated

                    self._update(job.run_id, current_item=f"Extracting {extractor_label(key, models)} "
                                                          f"({n} images, isolated worker)")
                    res = run_isolated("extract", {"path": str(self.store.abs(m.path)),
                                                   "preprocess": m.config["preprocess"],
                                                   "image_paths": [str(p) for p in paths]},
                                       timeout_s=self.settings.arena_sandbox_timeout_s,
                                       memory_mb=self.settings.arena_sandbox_memory_mb)
                    X = res["X"]
                    meta = {"ms_per_image": res["ms_per_image"], "ms_single_p50": res["ms_single_p50"],
                            "dim": int(X.shape[1]), "label": extractor_label(key, models), "isolated": True}
                    np.save(npy, X)
                    meta_p.write_text(json.dumps(meta))
                    return {"X": X, **meta}
                sess, feat = IMG.feature_session(self.store.abs(m.path))
                pp = IMG.ImagePreprocess(**m.config["preprocess"])

            def fn(part):
                x = IMG.preprocess(part, pp)
                return sess.run_batched(x, output=feat).reshape(len(part), -1)
        chunks = []
        step = 32
        t0 = time.perf_counter()
        for i in range(0, n, step):
            if job.cancel.is_set():
                raise _Cancelled()
            if i % (step * 8) == 0:
                self._update(job.run_id, current_item=f"Extracting {extractor_label(key, models)} ({i}/{n} images)")
            chunks.append(fn(paths[i:i + step]))
        batch_ms = (time.perf_counter() - t0) * 1000 / max(1, n)
        singles = []
        for p in paths[: min(10, n)]:
            t1 = time.perf_counter()
            fn([p])
            singles.append((time.perf_counter() - t1) * 1000)
        X = np.concatenate(chunks).astype(np.float32)
        meta = {"ms_per_image": round(batch_ms, 3), "ms_single_p50": round(float(np.median(singles)), 3),
                "dim": int(X.shape[1]), "label": extractor_label(key, models)}
        np.save(npy, X)
        meta_p.write_text(json.dumps(meta))
        return {"X": X, **meta}

    def _eval_builtin(self, c: dict, data: LoadedDataset, feat: dict | None, config: dict, seed: int,
                      run_dir: Path) -> dict:
        algo = ALGORITHMS[c["algorithm"]]
        kind = "image" if data.kind == "image" else data.kind
        Xsrc = feat["X"] if feat is not None else data.X
        types, features = None, None
        if data.kind == "tabular":
            features = list(data.X.columns)  # type: ignore[union-attr]
            types = data.types
        pipe = build_pipeline(algo, data.task, kind, seed=seed, types=types, features=features)
        Xtr, Xte = take(Xsrc, data.train_idx), take(Xsrc, data.test_idx)
        has_val = len(data.val_idx) > 0
        Xva = take(Xsrc, data.val_idx) if has_val else None
        ytr = data.y[data.train_idx]
        encoded = data.task == "classification" and algo.id in ("xgboost",)
        index = {lab: i for i, lab in enumerate(data.labels)}
        if encoded:
            ytr = np.array([index[v] for v in ytr])
        details: dict = {"algorithm": algo.id, "family": algo.family}
        if feat is not None and c.get("extractor"):
            details["feature_extractor"] = {"key": c["extractor"], "label": feat["label"],
                                            "dim": feat["dim"], "ms_per_image": feat["ms_per_image"]}
        scoring = "f1_macro" if data.task == "classification" else "r2"
        with _PeakRSS() as mem, warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            folds = int(config.get("cv_folds") or 0)
            if data.task == "classification":
                min_class = int(min(np.unique(ytr, return_counts=True)[1]))
                folds = min(folds, min_class)
            if folds >= 2 and algo.id != "dummy":
                splitter = (StratifiedKFold(folds, shuffle=True, random_state=seed) if data.task == "classification"
                            else KFold(folds, shuffle=True, random_state=seed))
                t0 = time.perf_counter()
                try:
                    cv = cross_val_score(clone(pipe), Xtr, ytr, cv=splitter, scoring=scoring, n_jobs=1)
                    details["cv"] = {"folds": folds, "metric": scoring, "mean": round(float(np.mean(cv)), 4),
                                     "std": round(float(np.std(cv)), 4), "scores": [round(float(v), 4) for v in cv],
                                     "seconds": round(time.perf_counter() - t0, 2)}
                except ValueError as exc:
                    details["cv"] = {"error": str(exc)[:200]}
            elif folds < 2 and int(config.get("cv_folds") or 0) >= 2:
                details["cv"] = {"error": "Skipped: a class has fewer training examples than folds."}
            if config.get("tune") and algo.grid:
                space = param_space(algo, data.task)
                n_iter = min(int(config.get("tune_iterations", 10)), int(np.prod([len(v) for v in space.values()])))
                search = RandomizedSearchCV(pipe, space, n_iter=n_iter, cv=3, scoring=scoring, random_state=seed,
                                            n_jobs=1, error_score=np.nan)
                t0 = time.perf_counter()
                search.fit(Xtr, ytr)
                best_params = {k.split("__")[-1]: _jsonable(v) for k, v in search.best_params_.items()}
                details["tuning"] = {"iterations": n_iter, "cv": 3, "best_params": best_params,
                                     "best_cv_score": round(float(search.best_score_), 4),
                                     "seconds": round(time.perf_counter() - t0, 2)}
                pipe = clone(search.best_estimator_)
            t0 = time.perf_counter()
            pipe.fit(Xtr, ytr)  # training rows only – validation/test never reach fit()
            fit_ms = (time.perf_counter() - t0) * 1000
            details["fit_rows"] = int(len(data.train_idx))
            raw_val = pipe.predict(Xva) if has_val else None
            t0 = time.perf_counter()
            raw_pred = pipe.predict(Xte)
            batch_ms = (time.perf_counter() - t0) * 1000
            _single_threaded(pipe)  # one request at a time, as in a deployed service
            timing_X = Xva if has_val else Xte
            singles = []
            for i in range(min(SINGLE_SAMPLES, len(data.val_idx) if has_val else len(data.test_idx))):
                one = take(timing_X, [i])
                t1 = time.perf_counter()
                pipe.predict(one)
                singles.append((time.perf_counter() - t1) * 1000)
            scores = val_scores = None
            if data.task == "classification":
                scores = _scores(pipe, Xte, data.labels, encoded)
                val_scores = _scores(pipe, Xva, data.labels, encoded) if has_val else None
        if caught:
            details["warnings"] = sorted({f"{w.category.__name__}: {str(w.message)[:160]}" for w in caught})[:5]
        n = len(data.test_idx)

        def decode(raw):
            if data.task == "classification":
                return np.asarray(raw, dtype=int) if encoded else np.array([index.get(str(v), -1) for v in raw])
            return np.asarray(raw, dtype=float)

        pred = decode(raw_pred)
        extra_ms = feat["ms_per_image"] if feat else 0.0
        extra_single = feat["ms_single_p50"] if feat else 0.0
        model_bytes = len(pickle.dumps(pipe, protocol=pickle.HIGHEST_PROTOCOL))
        perf = {"fit_ms": round(fit_ms, 1), "latency_ms_per_example": round(batch_ms / max(1, n) + extra_ms, 4),
                "latency_ms_single_p50": round(float(np.median(singles)) + extra_single, 3) if singles else None,
                "latency_ms_single_p95": round(float(np.percentile(singles, 95)) + extra_single, 3) if singles else None,
                "throughput_per_s": round(n / ((batch_ms + extra_ms * n) / 1000), 1) if batch_ms > 0 else None,
                "model_size_bytes": model_bytes, **mem.summary()}
        if feat is not None and c.get("extractor"):
            perf["feature_ms_per_example"] = extra_ms
        import joblib

        bundle = {"pipeline": pipe, "task": data.task, "kind": data.kind, "labels": data.labels,
                  "label_encoded": encoded, "features": features, "feature_extractor": details.get("feature_extractor"),
                  "created_by": f"Local AI Lab {__version__}"}
        joblib.dump(bundle, run_dir / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', c['key'])}.joblib", compress=3)
        out = {"status": "ok", "test": {"pred": pred, "idx": np.arange(n), "scores": scores}, "perf": perf,
               "details": details}
        if has_val:
            out["val"] = {"pred": decode(raw_val), "idx": np.arange(len(data.val_idx)), "scores": val_scores}
        return out

    def _eval_uploaded(self, c: dict, data: LoadedDataset, m: ArenaModel | None) -> dict:
        if m is None:
            raise ValidationFailed("The uploaded model was deleted.")
        from app.arena import sandbox_tasks

        path = self.store.abs(m.path)
        Xsrc = data.X_full if data.kind == "tabular" else data.X
        n = len(data.test_idx)
        has_val = len(data.val_idx) > 0
        kwargs = {"fmt": m.format, "path": str(path), "kind": m.kind, "task": m.task, "config": m.config,
                  "labels": data.labels,
                  "columns": list(data.X_full.columns) if hasattr(data.X_full, "columns") else None,
                  "X_val": take(Xsrc, data.val_idx) if has_val else None, "X_test": take(Xsrc, data.test_idx),
                  "singles": SINGLE_SAMPLES if data.kind != "image" else 10}
        if self.settings.arena_sandbox:
            from app.arena.sandbox import run_isolated

            res = run_isolated("evaluate", kwargs, timeout_s=self.settings.arena_sandbox_timeout_s,
                               memory_mb=self.settings.arena_sandbox_memory_mb)
        else:
            res = sandbox_tasks.evaluate(**kwargs)
        p, pv = res["test"], res["val"]
        batch_ms, singles = res["batch_ms"], res["singles"]
        details = {"format": m.format, "model_id": m.id, "sha256": m.sha256, "inspection": m.inspection,
                   "isolated": bool(self.settings.arena_sandbox),
                   "notes": sorted(set(p["notes"] + (pv["notes"] if pv else [])))}
        if m.kind == "image":
            details["preprocess"] = m.config.get("preprocess")
        perf = {"fit_ms": None, "latency_ms_per_example": round(batch_ms / max(1, n), 4),
                "latency_ms_single_p50": round(float(np.median(singles)), 3) if singles else None,
                "latency_ms_single_p95": round(float(np.percentile(singles, 95)), 3) if singles else None,
                "throughput_per_s": round(n / (batch_ms / 1000), 1) if batch_ms > 0 else None,
                "model_size_bytes": m.size_bytes, "parameters": (m.inspection or {}).get("parameters"),
                "peak_process_rss_mb": res.get("peak_rss_mb")}
        out = {"status": "ok", "test": {"pred": p["values"], "idx": np.arange(n), "scores": p["scores"]}, "perf": perf,
               "details": details}
        if pv is not None:
            out["val"] = {"pred": pv["values"], "idx": np.arange(len(data.val_idx)), "scores": pv["scores"]}
        return out

    def _eval_llm(self, job: _Job, c: dict, data: LoadedDataset, config: dict, seed: int) -> dict:
        model = self.registry.resolve(c["model"])
        rs = self.settings_service.get()
        limit = int(config.get("llm_max_examples", 100))
        labels = data.labels
        spec = {"labels": labels, "answer": ""}
        index = {lab: i for i, lab in enumerate(labels)}
        opts = {"temperature": 0, "seed": seed, "num_predict": 24, "num_ctx": 2048}
        warm = GenerationResult()
        try:
            for _ in stream_generate(model, [{"role": "user", "content": "Reply with OK."}],
                                     options={"num_predict": 4, "temperature": 0}, timeout_s=rs.benchmark_timeout_s * 2,
                                     result=warm):
                pass
        except AppError:
            pass
        lat: list[float] = []
        tps: list[float] = []
        errors = 0
        attempted = 0
        out_parts: dict[str, dict] = {}
        notes = []
        parts = [("val", data.val_idx), ("test", data.test_idx)] if len(data.val_idx) else [("test", data.test_idx)]
        for part, rows in parts:
            n = len(rows)
            positions = _stratified_positions(data.y[rows], limit, seed)
            if len(positions) < n:
                notes.append(f"{part}: evaluated on a stratified sample of {len(positions)} of {n} examples "
                             "(LLM max examples setting).")
            texts = take(data.X, rows)
            pred = []
            for k, pos in enumerate(positions):
                if job.cancel.is_set():
                    raise _Cancelled()
                if k % 10 == 0:
                    self._update(job.run_id, current_item=f"{c['name']}: {part} {k}/{len(positions)} examples")
                text = str(texts[pos])[:2000].replace('"', "'")
                res = GenerationResult()
                attempted += 1
                try:
                    for _ in stream_generate(model, [{"role": "user", "content": LLM_PROMPT.format(
                            labels=", ".join(labels), text=text)}], options=opts, timeout_s=rs.benchmark_timeout_s,
                            result=res):
                        pass
                    ev = parse_label(strip_think(res.content), spec)
                    pred.append(index.get(ev.details.get("predicted"), -1))
                    if res.total_ms:
                        lat.append(res.total_ms)
                    if res.tokens_per_s:
                        tps.append(res.tokens_per_s)
                except AppError as exc:
                    errors += 1
                    pred.append(-1)
                    if errors >= 5 and errors == attempted:
                        raise AppError(f"The model failed on the first {errors} examples: {exc.message}",
                                       code="llm_failed") from exc
            out_parts[part] = {"pred": np.array(pred, dtype=int), "idx": positions, "scores": None}
        perf = {"fit_ms": None, "latency_ms_per_example": round(float(np.mean(lat)), 1) if lat else None,
                "latency_ms_single_p50": round(float(np.median(lat)), 1) if lat else None,
                "latency_ms_single_p95": round(float(np.percentile(lat, 95)), 1) if lat else None,
                "latency_samples": len(lat),
                "throughput_per_s": round(1000 / float(np.mean(lat)), 2) if lat else None,
                "tokens_per_s_mean": round(float(np.mean(tps)), 2) if tps else None, "model_size_bytes": None}
        details = {"model": c["model"], "provider": model.provider, "parameter_size": model.parameter_size,
                   "quantization": model.quantization, "errors": errors, "prompt": LLM_PROMPT,
                   "evaluated": int(len(out_parts["test"]["idx"])), "of": int(len(data.test_idx)), "notes": notes}
        return {"status": "ok", **out_parts, "perf": perf, "details": details}

    # ------------------------------------------------------------ persistence
    def _store_result(self, run_id: int, c: dict, out: dict) -> None:
        with self.SessionLocal() as db:
            db.add(ArenaResult(run_id=run_id, key=c["key"], source=c["source"], name=c["name"][:300],
                               status=out["status"], metrics=out.get("metrics"), val_metrics=out.get("val_metrics"),
                               details=out.get("details"), predictions=out.get("predictions"),
                               evaluated_idx=out.get("evaluated_idx"), val_predictions=out.get("val_predictions"),
                               val_idx=out.get("val_idx"), error=out.get("error")))
            db.commit()

    def _keep_best_model(self, run_dir: Path, summary: dict, results: list[dict]) -> str | None:
        """Keep the trained file of the recommended (or best validation) built-in model; delete the others."""
        files = {p.stem: p for p in run_dir.glob("*.joblib")}
        ranking = summary.get("ranking") or []
        rec = (summary.get("recommendation") or {}).get("key")
        order = ([rec] if rec else []) + ranking
        keep = None
        for key in order:
            stem = re.sub(r"[^A-Za-z0-9_.-]", "_", key)
            if key.startswith("builtin:") and stem in files:
                keep = files[stem]
                break
        for p in files.values():
            if p != keep:
                p.unlink(missing_ok=True)
        if keep is None:
            shutil.rmtree(run_dir, ignore_errors=True)
            return None
        return str(keep.relative_to(self.settings.arena_dir).as_posix())

    def _track(self, run: ArenaRun, results: list[dict], summary: dict) -> str | None:
        metric = summary.get("primary_metric") or PRIMARY[run.dataset_info["task"]][0]
        metrics: dict[str, float] = {}
        for r in results:
            safe = re.sub(r"[^A-Za-z0-9_.-]", "_", r["key"])[:120]
            for part, block in r["splits"].items():
                for k in (metric, "accuracy", "rmse", "mae"):
                    v = block["metrics"].get(k)
                    if isinstance(v, (int, float)):
                        metrics[f"{safe}.{part}.{k}"] = v
            if isinstance(r.get("latency_ms"), (int, float)):
                metrics[f"{safe}.latency_ms"] = r["latency_ms"]
        params = {"dataset": run.dataset_info.get("name"), "dataset_sha256": run.dataset_info.get("sha256"),
                  "split_sha256": run.dataset_info.get("split_sha256"), "protocol": run.protocol,
                  "kind": run.dataset_info.get("kind"), "task": run.dataset_info.get("task"),
                  "n_train": run.dataset_info.get("n_train"), "n_val": run.dataset_info.get("n_val"),
                  "n_test": run.dataset_info.get("n_test"),
                  "contenders": ",".join(c["key"] for c in run.contenders),
                  **{k: v for k, v in run.config.items() if k not in ("algorithms", "uploaded_models", "llm_models")}}
        return self.tracker.log_run("local-ai-lab-model-arena", f"arena-{run.id}", params, metrics,
                                    {"summary.json": summary, "environment.json": run.environment},
                                    tags={"status": run.status, "app_version": __version__})


# ----------------------------------------------------------------------------- helpers
def _scores(pipe, Xte, labels: list[str], encoded: bool) -> np.ndarray | None:
    try:
        if hasattr(pipe, "predict_proba"):
            proba = np.asarray(pipe.predict_proba(Xte))
            classes = [str(c) for c in pipe.classes_] if not encoded else [labels[int(c)] for c in pipe.classes_]
            if classes == labels:
                return proba
            out = np.zeros((proba.shape[0], len(labels)))
            for j, cl in enumerate(classes):
                if cl in labels:
                    out[:, labels.index(cl)] = proba[:, j]
            return out
        if hasattr(pipe, "decision_function") and len(labels) == 2:
            return np.asarray(pipe.decision_function(Xte)).ravel()
    except Exception:  # noqa: BLE001 - scores are optional
        return None
    return None


def _single_threaded(pipe) -> None:
    params = {k: 1 for k, v in pipe.get_params().items() if k.endswith("n_jobs") and v not in (None, 1)}
    if params:
        pipe.set_params(**params)


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, tuple):
        return list(v)
    return v


def _error_examples(data: LoadedDataset, rows: np.ndarray, idx: np.ndarray, yt: np.ndarray, pred: np.ndarray,
                    labels: list[str], limit: int = 12) -> list[dict]:
    out = []
    for pos, t, p in zip(idx, yt, pred, strict=True):
        if t == p:
            continue
        row = int(rows[pos])
        ex = {"row": row, "expected": labels[int(t)], "predicted": labels[int(p)] if p >= 0 else None}
        if data.kind == "text":
            ex["text"] = str(data.X[row])[:220]
        elif data.kind == "image":
            ex["image"] = (data.sources or [])[row] if data.sources else str(data.X[row])
        else:
            ex["values"] = {k: (None if v != v else (round(v, 4) if isinstance(v, float) else str(v)[:40]))
                            for k, v in list(data.X.iloc[row].items())[:6]}  # type: ignore[union-attr]
        out.append(ex)
        if len(out) >= limit:
            break
    return out


def _regression_examples(data: LoadedDataset, rows: np.ndarray, idx: np.ndarray, yt: np.ndarray, pred: np.ndarray,
                         limit: int = 10) -> list[dict]:
    err = np.abs(pred - yt)
    order = np.argsort(-err)[:limit]
    return [{"row": int(rows[idx[i]]), "expected": round(float(yt[i]), 4),
             "predicted": round(float(pred[i]), 4), "abs_error": round(float(err[i]), 4)} for i in order]


def _stratified_positions(y: np.ndarray, limit: int, seed: int) -> np.ndarray:
    n = len(y)
    positions = np.arange(n)
    if n <= limit:
        return positions
    from sklearn.model_selection import train_test_split

    try:
        positions, _ = train_test_split(positions, train_size=limit, random_state=seed, stratify=y)
    except ValueError:
        positions, _ = train_test_split(positions, train_size=limit, random_state=seed)
    return np.sort(positions)


def _subset_resampling(rs, idx: np.ndarray):
    """The split's resampling scheme restricted to the evaluated positions (LLMs may use a sample)."""
    from app.arena.stats import Resampling

    if rs is None:
        return None
    return Resampling(rs.method, rs.n_boot, rs.seed, strata=rs.strata[idx] if rs.strata is not None else None,
                      groups=rs.groups[idx] if rs.groups is not None else None)


def _margin_text(selection: dict) -> str:
    m = selection.get("margin")
    if m is None:
        return "not set"
    return f"±{m:.3g} {selection.get('primary_label', '')} ({selection.get('margin_rule', '')})"


def _library_versions() -> dict:
    out = {}
    for mod in ("sklearn", "numpy", "pandas", "onnxruntime", "xgboost", "lightgbm"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001, S112 - optional library
            continue
    return out
