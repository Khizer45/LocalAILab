"""Bring-your-own-model support: inspect uploaded files and turn them into predictors.

Formats
* **ONNX** (recommended) – any framework can export to it (PyTorch
  ``torch.onnx.export``, Keras/TensorFlow ``tf2onnx``, scikit-learn
  ``skl2onnx``). No code inside the file is executed. Image classifiers
  (MobileNetV2, EfficientNet, your own CNN…), tabular models with one float
  input or one input per column, and text pipelines with a string input.
* **joblib / pickle** – scikit-learn pipelines. Loading a pickle can run
  arbitrary code, so it is accepted only from an administrator who ticks
  "I trust this file".

Label mapping for classifiers (first match wins): labels typed by the user →
string labels produced by the model → ``classes_`` of a scikit-learn model →
the dataset's class names in alphabetical order when the output size matches
(the order used by Keras ``image_dataset_from_directory`` and torchvision
``ImageFolder``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from app.arena.data import label_str
from app.arena.images import (
    ImagePreprocess,
    OnnxSession,
    count_parameters,
    feature_session,
    infer_image_preprocess,
    preprocess,
)
from app.core.errors import ValidationFailed

FORMATS = {".onnx": "onnx", ".joblib": "joblib", ".pkl": "joblib", ".pickle": "joblib"}


def detect_format(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext in FORMATS:
        return FORMATS[ext]
    if ext in (".h5", ".keras", ".pt", ".pth", ".pb", ".tflite", ".safetensors", ".bin"):
        raise ValidationFailed(f"'{ext}' files cannot be run directly – convert the model to ONNX first.",
                               code="convert_to_onnx",
                               hint="Keras: python -m tf2onnx.convert --saved-model <dir> --output model.onnx · "
                                    "PyTorch: torch.onnx.export(model, example_input, 'model.onnx'). "
                                    "See the User manual, 'Model Arena → Converting models'.")
    raise ValidationFailed("Upload an .onnx model (recommended) or a scikit-learn .joblib/.pkl file.")


def parse_labels(text: str | None) -> list[str] | None:
    if not text or not text.strip():
        return None
    sep = "\n" if "\n" in text.strip() else ","
    labels = [label_str(x) for x in text.split(sep) if x.strip()]
    return labels or None


# ----------------------------------------------------------------------------- ONNX metadata
META_PREFIX = "local_ai_lab."
_JSON_KEYS = ("labels", "preprocess", "feature_columns", "metrics")


def onnx_metadata(path: Path) -> dict:
    """``metadata_props`` written by the Local AI Lab Colab notebooks (or any exporter).

    Recognised keys (all optional, prefix ``local_ai_lab.``): ``kind``, ``task``, ``labels`` (JSON list in output
    order), ``preprocess`` (JSON), ``feature_columns`` (JSON list), ``dataset_sha256``, ``split_sha256``,
    ``trained_on`` (``train`` | ``train+val``), ``framework``, ``license``, ``notebook``. They are *claims made by
    whoever exported the file* – useful defaults, not proof.
    """
    import json

    import onnx

    try:
        m = onnx.load(str(path), load_external_data=False)
    except Exception:  # noqa: BLE001 - the session check reports unreadable files
        return {}
    props = {p.key: p.value for p in m.metadata_props}
    ours: dict = {}
    for k, v in props.items():
        if not k.startswith(META_PREFIX):
            continue
        key = k[len(META_PREFIX):][:60]
        if key in _JSON_KEYS:
            try:
                ours[key] = json.loads(v)
            except ValueError:
                continue
        else:
            ours[key] = str(v)[:300]
    return {"producer": f"{m.producer_name} {m.producer_version}".strip() or None,
            "opset": [int(o.version) for o in m.opset_import][:4], "local_ai_lab": ours or None}


def apply_metadata_defaults(config: dict, meta: dict) -> tuple[dict, list[str]]:
    """Fill labels / preprocessing / feature columns the user left empty from the file's metadata."""
    ours = (meta or {}).get("local_ai_lab") or {}
    cfg, used = dict(config), []
    labels = ours.get("labels")
    if not cfg.get("labels") and isinstance(labels, list) and labels and len(labels) <= 1000:
        cfg["labels"] = [label_str(x) for x in labels]
        used.append("labels")
    pp = ours.get("preprocess")
    if isinstance(pp, dict) and not _pp_cfg(cfg):
        allowed = {"size", "layout", "normalization", "color", "resize_mode", "channels"}
        cfg["preprocess"] = {k: v for k, v in pp.items() if k in allowed}
        used.append("preprocess")
    cols = ours.get("feature_columns")
    if not cfg.get("feature_columns") and isinstance(cols, list) and cols:
        cfg["feature_columns"] = [str(c) for c in cols][:500]
        used.append("feature_columns")
    return cfg, used


# ----------------------------------------------------------------------------- inspection
def inspect_model(path: Path, fmt: str, kind: str, task: str, role: str, config: dict, trusted: bool) -> tuple[dict, dict]:
    """Validate an uploaded model; return (inspection, normalised config)."""
    if kind not in ("tabular", "text", "image"):
        raise ValidationFailed("Data type must be tabular, text or image.")
    if role == "feature_extractor":
        if fmt != "onnx" or kind != "image":
            raise ValidationFailed("Only ONNX image models can be used as feature extractors.")
        sess, feat = feature_session(path)
        pp = infer_image_preprocess(sess, ImagePreprocess(**_pp_cfg(config)))
        dummy = np.zeros((1, pp.channels, pp.size, pp.size) if pp.layout == "nchw" else (1, pp.size, pp.size, pp.channels),
                         dtype=np.float32)
        dim = int(np.prod(sess.run({sess.inputs[0].name: dummy}, [feat])[0].shape[1:]))
        return ({"format": "onnx", **sess.describe(), "feature_tensor": feat, "feature_dim": dim,
                 "parameters": count_parameters(path)}, {**config, "preprocess": pp.to_dict()})
    if task not in ("classification", "regression"):
        raise ValidationFailed("Task must be classification or regression.")
    if fmt == "onnx":
        sess = OnnxSession(path)
        meta = onnx_metadata(path)
        cfg, used = apply_metadata_defaults(config, meta)
        info = {"format": "onnx", **sess.describe(), "parameters": count_parameters(path), "metadata": meta,
                "defaults_from_metadata": used}
        ours = meta.get("local_ai_lab") or {}
        for key, expected in (("kind", kind), ("task", task)):
            if ours.get(key) and ours[key] != expected:
                raise ValidationFailed(f"The file's metadata says it is a {ours[key]} model, but '{expected}' was "
                                       "selected.")
        if kind == "image":
            cfg["preprocess"] = infer_image_preprocess(sess, ImagePreprocess(**_pp_cfg(cfg))).to_dict()
        elif kind == "text":
            if "string" not in sess.inputs[0].type:
                raise ValidationFailed("Text ONNX models must take raw strings as input (e.g. a scikit-learn pipeline "
                                       "exported with skl2onnx).",
                                       hint="Models that need a tokenizer (BERT etc.) are not supported yet.")
        return info, cfg
    # joblib
    if not trusted:
        raise ValidationFailed("Pickle/joblib files can execute code when loaded. Tick 'I trust this file' "
                               "(administrators only) or export the model to ONNX.", code="untrusted_pickle")
    if kind == "image":
        raise ValidationFailed("Image models must be uploaded as ONNX.")
    import joblib

    try:
        model = joblib.load(path)
    except Exception as exc:  # noqa: BLE001
        raise ValidationFailed(f"Could not load the file: {type(exc).__name__}: {str(exc)[:200]}",
                               hint="It must be saved with joblib.dump() by a compatible scikit-learn version.") from exc
    if not hasattr(model, "predict"):
        raise ValidationFailed("The object in the file has no predict() method.")
    info = {"format": "joblib", "class": type(model).__name__,
            "classes": [label_str(c) for c in getattr(model, "classes_", [])][:200] or None,
            "feature_names": [str(c) for c in getattr(model, "feature_names_in_", [])][:500] or None}
    return info, dict(config)


def _pp_cfg(config: dict) -> dict:
    pp = config.get("preprocess") or {}
    keys = ("size", "layout", "normalization", "color", "resize_mode", "channels")
    return {k: pp[k] for k in keys if pp.get(k) not in (None, "", "auto")}


def check_compatible(model_kind: str, model_task: str, role: str, ds_kind: str, ds_task: str) -> str | None:
    if role == "feature_extractor":
        return None if ds_kind == "image" else "feature extractors only apply to image datasets"
    if model_kind != ds_kind:
        return f"model is for {model_kind} data, dataset is {ds_kind}"
    if model_task != ds_task:
        return f"model does {model_task}, dataset needs {ds_task}"
    return None


# ----------------------------------------------------------------------------- predictors
@dataclass
class Prediction:
    values: np.ndarray  # class indices (classification, -1 = unmapped) or floats (regression)
    scores: np.ndarray | None = None
    notes: list[str] | None = None


class LabelMapper:
    def __init__(self, dataset_labels: list[str], user_labels: list[str] | None, model_classes: list[str] | None):
        self.dataset_labels = dataset_labels
        self.index = {lab: i for i, lab in enumerate(dataset_labels)}
        self.user_labels = user_labels
        self.model_classes = model_classes
        self.notes: list[str] = []

    def from_names(self, names) -> np.ndarray:
        out = np.array([self.index.get(label_str(n), -1) for n in names], dtype=int)
        unknown = int((out < 0).sum())
        if unknown:
            self.notes.append(f"{unknown} predictions used labels that are not in the dataset (counted as wrong).")
        return out

    def from_indices(self, idx: np.ndarray, n_outputs: int) -> np.ndarray:
        names = self.user_labels or self.model_classes
        if names:
            if len(names) != n_outputs:
                raise ValidationFailed(f"The model has {n_outputs} outputs but {len(names)} labels were given.")
            return self.from_names([names[i] for i in idx])
        if n_outputs != len(self.dataset_labels):
            raise ValidationFailed(
                f"The model has {n_outputs} output classes but the dataset has {len(self.dataset_labels)}.",
                hint="Type the model's class names (in output order) in the Labels field when uploading it, or use "
                     "the model as a feature extractor if it was trained on other classes (e.g. ImageNet).")
        self.notes.append("Output order assumed to be the dataset's class names sorted alphabetically.")
        return np.asarray(idx, dtype=int)


def _classify_scores(out: np.ndarray, mapper: LabelMapper, n_classes: int) -> Prediction:
    out = np.asarray(out, dtype=np.float32)
    if out.ndim == 1:
        out = out.reshape(-1, 1)
    out = out.reshape(out.shape[0], -1)
    if out.shape[1] == 1 and n_classes == 2:  # sigmoid / logit single output (Keras binary models)
        thr = 0.5 if (out.min() >= 0 and out.max() <= 1) else 0.0
        idx = (out[:, 0] > thr).astype(int)
        mapper.notes.append(f"Single-output binary model: class 1 when output > {thr}.")
        values = mapper.from_indices(idx, 2)
        p1 = out[:, 0] if thr == 0.5 else 1 / (1 + np.exp(-out[:, 0]))
        return Prediction(values, np.stack([1 - p1, p1], axis=1))
    idx = out.argmax(axis=1)
    values = mapper.from_indices(idx, out.shape[1])
    scores = out if out.shape[1] == n_classes and mapper.user_labels is None else None
    if scores is not None and not np.allclose(scores.sum(axis=1), 1.0, atol=1e-3):
        e = np.exp(scores - scores.max(axis=1, keepdims=True))
        scores = e / e.sum(axis=1, keepdims=True)
    return Prediction(values, scores)


class OnnxPredictor:
    def __init__(self, path: Path, kind: str, task: str, config: dict, dataset_labels: list[str],
                 feature_columns: list[str] | None = None):
        self.sess = OnnxSession(path)
        self.kind, self.task, self.config = kind, task, config
        self.labels = dataset_labels
        self.columns = feature_columns or []
        self.mapper = LabelMapper(dataset_labels, config.get("labels"), None)
        self.pp = ImagePreprocess(**config["preprocess"]) if kind == "image" else None

    # inputs ----------------------------------------------------------------
    def _feeds(self, X) -> dict:
        ins = self.sess.inputs
        if self.kind == "text":
            arr = np.array([str(t) for t in X], dtype=object)
            shape = ins[0].shape
            return {ins[0].name: arr.reshape(-1, 1) if len(shape) == 2 else arr}
        # tabular
        names = [i.name for i in ins]
        if len(ins) > 1 or (names[0] in X.columns and len(X.columns) > 1 and len(ins) == len(X.columns)):
            feeds = {}
            for i in ins:
                if i.name not in X.columns:
                    raise ValidationFailed(f"The model expects an input named '{i.name}', which is not a dataset "
                                           "column.")
                col = X[i.name]
                if "string" in i.type:
                    feeds[i.name] = col.fillna("").astype(str).to_numpy(dtype=object).reshape(-1, 1)
                elif "int" in i.type:
                    feeds[i.name] = pd.to_numeric(col, errors="coerce").fillna(0).to_numpy(np.int64).reshape(-1, 1)
                else:
                    feeds[i.name] = pd.to_numeric(col, errors="coerce").to_numpy(np.float32).reshape(-1, 1)
            return feeds
        cols = self.config.get("feature_columns") or list(X.columns)
        missing = [c for c in cols if c not in X.columns]
        if missing:
            raise ValidationFailed(f"Feature columns not in the dataset: {', '.join(missing[:5])}.")
        try:
            mat = X[cols].apply(pd.to_numeric, errors="raise").to_numpy(np.float32)
        except (ValueError, TypeError) as exc:
            raise ValidationFailed("This ONNX model takes one numeric input, but the dataset has non-numeric "
                                   "columns.", hint="Export the model with one input per column (skl2onnx does this "
                                                    "for pipelines with ColumnTransformer).") from exc
        width = ins[0].shape[-1] if ins[0].shape else None
        if isinstance(width, int) and width != mat.shape[1]:
            raise ValidationFailed(f"The model expects {width} features; the dataset provides {mat.shape[1]}.",
                                   hint="Set 'feature columns' (comma-separated, in training order) when uploading.")
        return {ins[0].name: np.nan_to_num(mat)}

    def _run(self, X) -> list:
        """Run the model; images (a list of paths) are loaded and preprocessed in chunks."""
        if self.kind == "image" or self.sess.fixed_batch:
            step = 1 if self.sess.fixed_batch else 32
            outs: list[list] = []
            for i in range(0, len(X), step):
                part = X[i:i + step]
                feeds = {self.sess.inputs[0].name: preprocess(part, self.pp)} if self.kind == "image" else self._feeds(part)
                outs.append(self.sess.run(feeds))
            merged = []
            for j in range(len(outs[0])):
                if isinstance(outs[0][j], np.ndarray):
                    merged.append(np.concatenate([np.atleast_1d(np.asarray(o[j])) for o in outs], axis=0))
                else:
                    merged.append(sum((list(o[j]) for o in outs), []))
            return merged
        return self.sess.run(self._feeds(X))

    def predict(self, X) -> Prediction:
        outs = self._run(X)
        if self.task == "regression":
            vals = np.asarray(outs[0], dtype=float).reshape(len(X), -1)[:, 0]
            return Prediction(vals)
        # prefer an explicit label output (skl2onnx: "label"), then a score matrix
        for o, meta in zip(outs, self.sess.outputs, strict=False):
            if ("string" in meta.type or "int64" in meta.type) and isinstance(o, np.ndarray) and o.size == len(X):
                flat = o.reshape(-1)
                if "string" in meta.type:
                    return Prediction(self.mapper.from_names(flat), None, self.mapper.notes)
                classes = self.config.get("labels")
                if classes:
                    return Prediction(self.mapper.from_names([classes[int(i)] for i in flat]), None, self.mapper.notes)
                return Prediction(self.mapper.from_indices(flat.astype(int), len(self.labels)), None,
                                  self.mapper.notes)
        for o in outs:
            if isinstance(o, np.ndarray) and o.dtype.kind == "f":
                p = _classify_scores(o, self.mapper, len(self.labels))
                p.notes = self.mapper.notes
                return p
        raise ValidationFailed("Could not interpret the model's outputs as class labels or scores.")


class JoblibPredictor:
    def __init__(self, path: Path, kind: str, task: str, config: dict, dataset_labels: list[str]):
        import joblib

        self.model = joblib.load(path)
        self.kind, self.task, self.labels = kind, task, dataset_labels
        classes = [label_str(c) for c in getattr(self.model, "classes_", [])] or None
        self.mapper = LabelMapper(dataset_labels, config.get("labels"), classes)
        self.config = config

    def _input(self, X):
        if self.kind == "text":
            return list(X)
        names = getattr(self.model, "feature_names_in_", None)
        if names is not None:
            missing = [c for c in names if c not in X.columns]
            if missing:
                raise ValidationFailed(f"The model was trained with columns that are not in this dataset: "
                                       f"{', '.join(map(str, missing[:5]))}.")
            return X[list(names)]
        return X

    def predict(self, X) -> Prediction:
        raw = self.model.predict(self._input(X))
        if self.task == "regression":
            return Prediction(np.asarray(raw, dtype=float).ravel())
        scores = None
        if hasattr(self.model, "predict_proba") and self.mapper.model_classes \
                and self.mapper.model_classes == self.labels:
            try:
                scores = np.asarray(self.model.predict_proba(self._input(X)))
            except Exception:  # noqa: BLE001
                scores = None
        return Prediction(self.mapper.from_names(np.asarray(raw).ravel()), scores, self.mapper.notes)


def load_predictor(fmt: str, path: Path, kind: str, task: str, config: dict, dataset_labels: list[str],
                   feature_columns: list[str] | None = None):
    if fmt == "onnx":
        return OnnxPredictor(path, kind, task, config, dataset_labels, feature_columns)
    return JoblibPredictor(path, kind, task, config, dataset_labels)
