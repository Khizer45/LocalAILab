"""Storage of Model Arena datasets and uploaded models (files + database rows)."""

from __future__ import annotations

import hashlib
import logging
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.arena import data as D
from app.arena.images import sha256_file
from app.arena.uploaded import check_compatible, detect_format, inspect_model, parse_labels
from app.core.config import Settings
from app.core.errors import ForbiddenError, NotFoundError, ValidationFailed
from app.core.security import sanitize_filename
from app.models.arena import ArenaDataset, ArenaModel

log = logging.getLogger("app.arena")


@dataclass
class LoadedDataset:
    kind: str
    task: str
    labels: list[str]
    X: object  # DataFrame (tabular), np.ndarray[str] (text) or list[Path] (image) – features used by built-ins
    X_full: object  # all non-target columns (tabular) – handed to uploaded models
    y: np.ndarray  # strings (classification) or floats (regression)
    train_idx: np.ndarray
    test_idx: np.ndarray
    sources: list[str] | None = None  # image source paths (for error examples)
    types: dict[str, str] | None = None  # column types of the feature columns (tabular)
    val_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    groups: np.ndarray | None = None  # group id per row (group split) – bootstrap resamples whole groups
    split_sha256: str | None = None


def take(X, idx):
    if isinstance(X, pd.DataFrame):
        return X.iloc[idx]
    if isinstance(X, np.ndarray):
        return X[idx]
    return [X[i] for i in idx]


TRAINING_DATA = {
    "unknown": "Training data unknown – it may overlap with the validation/test rows.",
    "train_split": "Declared: trained only on the training split exported from this app.",
    "external": "Declared: trained on separate data that does not contain these rows.",
}


def normalize_provenance(raw: dict | None, inspection: dict) -> dict:
    raw = raw or {}
    meta = ((inspection or {}).get("metadata") or {}).get("local_ai_lab") or {}
    training = raw.get("training_data") if raw.get("training_data") in TRAINING_DATA else "unknown"
    if training == "unknown" and meta.get("trained_on") in ("train", "train+val") and meta.get("split_sha256"):
        training = "train_split"
    return {"training_data": training, "note": str(raw.get("note") or "")[:500] or None,
            "dataset_sha256": meta.get("dataset_sha256"), "split_sha256": meta.get("split_sha256"),
            "trained_on": meta.get("trained_on"), "source": "metadata" if meta else "user"}


def provenance_caveats(prov: dict | None, dataset_sha256: str | None, split_sha256: str | None,
                       evaluated_on_val: bool) -> list[str]:
    """Plain-language caveats for an uploaded model on a given dataset/split."""
    prov = prov or {"training_data": "unknown"}
    kind = prov.get("training_data", "unknown")
    if kind == "train_split":
        if prov.get("split_sha256") and split_sha256 and prov["split_sha256"] != split_sha256:
            return ["The model's metadata names a different split than this dataset's – it may have been trained on "
                    "rows that are in the validation/test split here."]
        if prov.get("trained_on") == "train+val" and evaluated_on_val:
            return ["Trained on train+validation (declared): its validation score is optimistic, so it is compared "
                    "using test only for information."]
        if prov.get("split_sha256") and prov["split_sha256"] == split_sha256:
            return []
        return ["Declared as trained on this dataset's training split (not verifiable)."]
    if kind == "external":
        return ["Declared as trained on separate data (not verifiable)."]
    return ["Training data unknown: this model may have seen validation/test rows, so its scores may be optimistic."]


class ArenaStore:
    def __init__(self, settings: Settings, SessionLocal: sessionmaker[Session]):
        self.settings = settings
        self.SessionLocal = SessionLocal
        self.root = settings.arena_dir
        for sub in ("datasets", "models", "features", "runs", "staging", "backbones"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ helpers
    def staging_path(self, filename: str) -> Path:
        suffix = Path(sanitize_filename(filename)).suffix.lower()[:10]
        return self.root / "staging" / f"{uuid.uuid4().hex}{suffix}"

    def abs(self, rel: str) -> Path:
        return (self.root / rel).resolve()

    @property
    def limits(self) -> dict:
        s = self.settings
        return {"max_upload_mb": s.arena_max_upload_mb, "max_rows": s.arena_max_rows,
                "max_columns": s.arena_max_columns, "max_images": s.arena_max_images}

    # ------------------------------------------------------------ datasets
    def preview(self, upload: Path, filename: str) -> dict:
        ext = Path(filename).suffix.lower()
        if ext == ".zip":
            return D.preview_zip(upload, max_images=self.settings.arena_max_images)
        if ext in (".csv", ".tsv", ".txt"):
            return D.preview_csv(upload.read_bytes(), max_rows=self.settings.arena_max_rows,
                                 max_columns=self.settings.arena_max_columns)
        raise ValidationFailed("Upload a .csv file (tabular or text data) or a .zip of image folders.")

    def create_dataset(self, owner_id: int, upload: Path, filename: str, *, name: str | None, target: str | None,
                       task: str, text_column: str | None, ignore: list[str], test_size: float, seed: int,
                       val_size: float = 0.2, split_strategy: str = "random", group_column: str | None = None,
                       time_column: str | None = None) -> ArenaDataset:
        if not 0.05 <= test_size <= 0.5:
            raise ValidationFailed("Test size must be between 5% and 50%.")
        if not 0 <= val_size <= 0.4 or test_size + val_size > 0.7:
            raise ValidationFailed("Validation size must be between 0% and 40% (test + validation at most 70%).")
        ext = Path(filename).suffix.lower()
        size = upload.stat().st_size
        digest = sha256_file(upload)
        with self.SessionLocal() as db:
            row = ArenaDataset(owner_id=owner_id, name=(name or Path(sanitize_filename(filename)).stem)[:200],
                               kind="pending", task="classification", filename=sanitize_filename(filename),
                               size_bytes=size, sha256=digest, path="", test_size=test_size, seed=seed,
                               val_size=val_size)
            db.add(row)
            db.commit()
            db.refresh(row)
            rel = f"datasets/{row.id}"
            folder = self.abs(rel)
            folder.mkdir(parents=True, exist_ok=True)
            try:
                if ext == ".zip":
                    if split_strategy not in ("random", "provided"):
                        raise ValidationFailed("Image datasets support the random split or train/val/test folders.")
                    prep = D.extract_images(upload, folder, test_size=test_size, val_size=val_size, seed=seed,
                                            max_images=self.settings.arena_max_images)
                elif ext in (".csv", ".tsv", ".txt"):
                    raw = upload.read_bytes()
                    if not target:
                        target = D.suggest_target(D.read_csv(raw, max_rows=self.settings.arena_max_rows,
                                                             max_columns=self.settings.arena_max_columns))
                    _, prep = D.prepare_csv(raw, D.CsvConfig(target=target, task=task, text_column=text_column or None,
                                                             ignore=ignore, test_size=test_size, val_size=val_size,
                                                             seed=seed, split_strategy=split_strategy,
                                                             group_column=group_column or None,
                                                             time_column=time_column or None),
                                            max_rows=self.settings.arena_max_rows,
                                            max_columns=self.settings.arena_max_columns)
                    shutil.copyfile(upload, folder / "data.csv")
                else:
                    raise ValidationFailed("Upload a .csv file or a .zip of image folders.")
                split_digest = D.save_split(folder, prep.train_idx, prep.test_idx, prep.val_idx,
                                            strategy=prep.split_strategy, group_column=prep.group_column,
                                            time_column=prep.time_column, seed=seed)
            except Exception:
                db.delete(row)
                db.commit()
                shutil.rmtree(folder, ignore_errors=True)
                raise
            row.kind, row.task, row.target, row.text_column = prep.kind, prep.task, prep.target, prep.text_column
            row.columns = {"features": prep.features, "ignored": prep.ignored, "types": prep.types}
            row.labels = prep.labels
            row.n_rows, row.n_train, row.n_test = prep.n_rows, len(prep.train_idx), len(prep.test_idx)
            row.n_val, row.split_sha256 = len(prep.val_idx), split_digest
            row.group_column, row.time_column = prep.group_column, prep.time_column
            row.split_source, row.profile, row.warnings, row.path = prep.split_source, prep.profile, prep.warnings, rel
            db.commit()
            db.refresh(row)
            log.info("arena dataset created", extra={"dataset_id": row.id, "kind": row.kind, "rows": row.n_rows})
            return row

    def dataset_owned(self, db: Session, owner_id: int, dataset_id: int) -> ArenaDataset:
        row = db.get(ArenaDataset, dataset_id)
        if row is None or row.owner_id != owner_id:
            raise NotFoundError("Dataset not found.")
        return row

    def delete_dataset(self, db: Session, owner_id: int, dataset_id: int) -> None:
        row = self.dataset_owned(db, owner_id, dataset_id)
        folder = self.abs(row.path) if row.path else None
        db.delete(row)
        db.commit()
        if folder:
            shutil.rmtree(folder, ignore_errors=True)
        shutil.rmtree(self.root / "features" / str(dataset_id), ignore_errors=True)

    def ensure_validation(self, row: ArenaDataset, val_size: float = 0.2) -> bool:
        """Give a v1.1 dataset (train/test only) a validation split taken from its training rows.

        The test split is not touched, so earlier test results stay comparable. Returns True if changed.
        """
        folder = self.abs(row.path)
        split = D.load_split(folder)
        if split["val"] or row.n_train < 10:
            return False
        y = self._labels_for_split(row, folder)
        frac = min(0.5, val_size / (1 - (row.test_size or 0.2)))
        rest, val = D._random_part(np.asarray(split["train"], dtype=int), y, row.task, frac, int(row.seed) + 1)
        digest = D.save_split(folder, sorted(rest.tolist()), split["test"], sorted(val.tolist()),
                              strategy=split.get("strategy", row.split_source), seed=row.seed,
                              upgraded_from="v1.1 train/test split")
        with self.SessionLocal() as db:
            r = db.get(ArenaDataset, row.id)
            r.n_train, r.n_val, r.val_size, r.split_sha256 = len(rest), len(val), val_size, digest
            r.warnings = list(r.warnings or []) + [
                f"Validation split added in v1.2: {len(val)} rows taken from the former training rows "
                "(the test split is unchanged)."]
            db.commit()
        row.n_train, row.n_val, row.split_sha256 = len(rest), len(val), digest
        log.info("arena dataset upgraded with a validation split", extra={"dataset_id": row.id})
        return True

    def _labels_for_split(self, row: ArenaDataset, folder: Path) -> np.ndarray:
        if row.kind == "image":
            return np.array([it["label"] for it in D.load_manifest(folder)])
        df = D.read_csv((folder / "data.csv").read_bytes(), max_rows=self.settings.arena_max_rows,
                        max_columns=self.settings.arena_max_columns)
        df, _, _ = D.clean_frame(df, row.target, row.task, group_column=row.group_column, time_column=row.time_column)
        return df[row.target].to_numpy()

    def load(self, row: ArenaDataset) -> LoadedDataset:
        folder = self.abs(row.path)
        split = D.load_split(folder)
        tr, te = np.asarray(split["train"], dtype=int), np.asarray(split["test"], dtype=int)
        va = np.asarray(split["val"], dtype=int)
        if row.kind == "image":
            items = D.load_manifest(folder)
            paths = [folder / it["file"] for it in items]
            y = np.array([it["label"] for it in items])
            return LoadedDataset("image", "classification", list(row.labels), paths, paths, y, tr, te,
                                 sources=[it["source"] for it in items], val_idx=va, split_sha256=split["sha256"])
        types = dict(row.columns.get("types", {}))
        features = list(row.columns.get("features", []))
        raw = (folder / "data.csv").read_bytes()
        df = D.read_csv(raw, max_rows=self.settings.arena_max_rows, max_columns=self.settings.arena_max_columns)
        df, _, _ = D.clean_frame(df, row.target, row.task, group_column=row.group_column, time_column=row.time_column)
        if len(df) != row.n_rows:
            raise ValidationFailed("The stored dataset no longer matches its split. Delete and upload it again.")
        y = df[row.target].to_numpy(dtype=float) if row.task == "regression" else df[row.target].astype(str).to_numpy()
        others = [c for c in df.columns if c != row.target]
        X_full = D.coerce_features(df, others, {c: types.get(c, D.column_type(df[c])) for c in others})
        if row.kind == "text":
            X = df[row.text_column].fillna("").astype(str).to_numpy(dtype=object)
        else:
            X = D.coerce_features(df, features, types)
        groups = df[row.group_column].map(D.label_str).to_numpy() if row.group_column else None
        return LoadedDataset(row.kind, row.task, list(row.labels), X, X_full, y, tr, te,
                             types={c: types.get(c, "categorical") for c in features}, val_idx=va, groups=groups,
                             split_sha256=split["sha256"])

    # ------------------------------------------------------------ uploaded models
    def create_model(self, owner_id: int, is_admin: bool, upload: Path, filename: str, *, name: str | None, kind: str,
                     task: str, role: str, labels_text: str | None, feature_columns: str | None,
                     preprocess: dict, trusted: bool, provenance: dict | None = None) -> ArenaModel:
        fmt = detect_format(filename)
        if fmt == "joblib" and trusted and not is_admin:
            raise ForbiddenError("Only administrators can upload pickle/joblib models (they can execute code).")
        role = "feature_extractor" if role == "feature_extractor" else "predictor"
        if role == "feature_extractor":
            task = "features"
        config: dict = {"labels": parse_labels(labels_text), "preprocess": preprocess or {}}
        if feature_columns and feature_columns.strip():
            config["feature_columns"] = [c.strip() for c in feature_columns.split(",") if c.strip()]
        if self.settings.arena_sandbox:
            from app.arena.sandbox import run_isolated

            res = run_isolated("inspect", {"path": str(upload), "fmt": fmt, "kind": kind, "task": task, "role": role,
                                           "config": config, "trusted": trusted},
                               timeout_s=min(600, self.settings.arena_sandbox_timeout_s),
                               memory_mb=self.settings.arena_sandbox_memory_mb)
            inspection, config = res["inspection"], res["config"]
            inspection["isolated"] = True
        else:
            inspection, config = inspect_model(upload, fmt, kind, task, role, config, trusted)
        prov = normalize_provenance(provenance, inspection)
        if inspection.get("classes") and not config.get("labels"):
            config["labels"] = None  # joblib classes_ are used automatically at prediction time
        digest = sha256_file(upload)
        with self.SessionLocal() as db:
            row = ArenaModel(owner_id=owner_id, name=(name or Path(sanitize_filename(filename)).stem)[:200],
                             filename=sanitize_filename(filename), format=fmt, kind=kind, task=task, role=role,
                             size_bytes=upload.stat().st_size, sha256=digest, config=config, inspection=inspection,
                             trusted=bool(trusted and fmt == "joblib"), provenance=prov, path="")
            db.add(row)
            db.commit()
            db.refresh(row)
            rel = f"models/{row.id}/{hashlib.sha256(row.filename.encode()).hexdigest()[:8]}{Path(row.filename).suffix.lower()}"
            dest = self.abs(rel)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(upload, dest)
            row.path = rel
            db.commit()
            db.refresh(row)
            return row

    def model_owned(self, db: Session, owner_id: int, model_id: int) -> ArenaModel:
        row = db.get(ArenaModel, model_id)
        if row is None or row.owner_id != owner_id:
            raise NotFoundError("Model not found.")
        return row

    def delete_model(self, db: Session, owner_id: int, model_id: int) -> None:
        row = self.model_owned(db, owner_id, model_id)
        folder = self.abs(row.path).parent if row.path else None
        db.delete(row)
        db.commit()
        if folder:
            shutil.rmtree(folder, ignore_errors=True)

    def compatible_models(self, db: Session, owner_id: int, ds: ArenaDataset) -> list[dict]:
        rows = db.scalars(select(ArenaModel).where(ArenaModel.owner_id == owner_id)).all()
        return [{"id": m.id, "name": m.name, "role": m.role,
                 "reason": check_compatible(m.kind, m.task, m.role, ds.kind, ds.task)} for m in rows]
