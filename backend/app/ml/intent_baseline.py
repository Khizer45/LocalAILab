"""Traditional ML baseline: TF-IDF features + linear classifiers (scikit-learn).

Procedure (reproducible, seed 42):
1. Load the versioned intent dataset and its stratified train/val/test split.
2. Train three candidate pipelines on *train* and compare macro-F1 on *val*:
   Logistic Regression, Linear SVM, Multinomial Naive Bayes.
3. Refit the best candidate on train+val and evaluate once on the held-out
   *test* split: accuracy, macro precision/recall/F1, per-class report and
   confusion matrix, plus measured CPU inference latency.
4. Save the fitted pipeline (joblib) and metrics; log to MLflow if enabled.

These are *classification* metrics and are reported separately from LLM
response-quality scores.
"""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from app.ml.intent_data import SEED, IntentSplit, load_split

ARTIFACT_NAME = "intent_baseline_v1.joblib"


def _features() -> FeatureUnion:
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=1)),
    ])


def candidates() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline([("features", _features()),
                                         ("clf", LogisticRegression(C=10.0, max_iter=2000, random_state=SEED))]),
        "linear_svm": Pipeline([("features", _features()), ("clf", LinearSVC(C=1.0, random_state=SEED))]),
        "naive_bayes": Pipeline([("features", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
                                 ("clf", MultinomialNB(alpha=0.3))]),
    }


def classification_metrics(y_true: list[str], y_pred: list[str | None], labels: list[str]) -> dict:
    """Shared metric computation (also used for LLM zero-shot predictions)."""
    y_pred_s = [p if p is not None else "__none__" for p in y_pred]
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred_s, labels=labels, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred_s, labels=labels, output_dict=True, zero_division=0)
    return {
        "n": len(y_true),
        "accuracy": round(float(accuracy_score(y_true, y_pred_s)), 4),
        "precision_macro": round(float(p), 4),
        "recall_macro": round(float(r), 4),
        "f1_macro": round(float(f), 4),
        "unparseable_predictions": sum(1 for x in y_pred if x is None),
        "per_class": {lab: {k: round(float(v), 4) for k, v in report[lab].items()} for lab in labels},
        "confusion_matrix": {"labels": labels,
                             "matrix": confusion_matrix(y_true, y_pred_s, labels=labels).tolist()},
    }


def train_and_evaluate(datasets_dir: Path, out_dir: Path) -> dict:
    split: IntentSplit = load_split(datasets_dir)
    x_tr, y_tr = [t for t, _ in split.train], [y for _, y in split.train]
    x_va, y_va = [t for t, _ in split.val], [y for _, y in split.val]
    x_te, y_te = [t for t, _ in split.test], [y for _, y in split.test]

    selection = []
    for name, pipe in candidates().items():
        t0 = time.perf_counter()
        pipe.fit(x_tr, y_tr)
        fit_ms = (time.perf_counter() - t0) * 1000
        pred = pipe.predict(x_va)
        selection.append({"model": name, "val_f1_macro": round(float(f1_score(y_va, pred, average="macro")), 4),
                          "val_accuracy": round(float(accuracy_score(y_va, pred)), 4), "fit_ms": round(fit_ms, 1)})
    best = max(selection, key=lambda s: (s["val_f1_macro"], s["val_accuracy"]))["model"]

    final = candidates()[best]
    final.fit(x_tr + x_va, y_tr + y_va)
    t0 = time.perf_counter()
    pred_test = list(final.predict(x_te))
    latency_ms = (time.perf_counter() - t0) * 1000 / max(1, len(x_te))
    test_metrics = classification_metrics(y_te, pred_test, split.labels)
    test_metrics["latency_ms_per_example"] = round(latency_ms, 3)

    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = out_dir / ARTIFACT_NAME
    joblib.dump({"pipeline": final, "labels": split.labels, "model": best, "dataset_sha256": split.sha256}, artifact)

    errors = [{"text": t, "expected": y, "predicted": p} for t, y, p in zip(x_te, y_te, pred_test, strict=True) if y != p]
    return {
        "dataset": split.summary(),
        "model_selection": {"metric": "val_f1_macro", "candidates": selection, "selected": best},
        "test": test_metrics,
        "test_errors": errors,
        "artifact_path": str(artifact),
        "notes": "Final model refit on train+validation, evaluated once on the held-out test split.",
    }


class IntentBaseline:
    def __init__(self, artifact_path: Path):
        self.artifact_path = artifact_path
        self._bundle = None
        self._mtime = 0.0

    def _load(self):
        if not self.artifact_path.exists():
            return None
        mtime = self.artifact_path.stat().st_mtime
        if self._bundle is None or mtime != self._mtime:
            self._bundle = joblib.load(self.artifact_path)
            self._mtime = mtime
        return self._bundle

    def predict(self, text: str) -> dict | None:
        bundle = self._load()
        if bundle is None:
            return None
        pipe = bundle["pipeline"]
        t0 = time.perf_counter()
        label = str(pipe.predict([text])[0])
        scores = None
        clf = pipe.named_steps["clf"]
        if hasattr(clf, "predict_proba"):
            proba = pipe.predict_proba([text])[0]
            scores = {str(c): round(float(p), 4) for c, p in zip(clf.classes_, proba, strict=True)}
        elif hasattr(clf, "decision_function"):
            d = np.atleast_1d(pipe.decision_function([text])[0])
            scores = {str(c): round(float(v), 4) for c, v in zip(clf.classes_, d, strict=True)}
        return {"label": label, "scores": scores, "model": bundle["model"],
                "latency_ms": round((time.perf_counter() - t0) * 1000, 3)}
