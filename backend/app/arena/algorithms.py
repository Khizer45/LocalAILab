"""Built-in algorithm catalog for the Model Arena.

Each algorithm can be trained on the user's own dataset. Preprocessing is
chosen by data kind so every algorithm sees the same information:

* **tabular** – median imputation + standard scaling for numbers, most-frequent
  imputation + one-hot encoding (top 30 values) for categories, TF-IDF for
  short free-text columns.
* **text** – word (1–2 gram) + character (3–5 gram) TF-IDF; algorithms that need
  dense input get TF-IDF → truncated SVD (300 dimensions).
* **image** – features from a chosen extractor (raw pixels or a pretrained
  ONNX backbone), then standard scaling.

Regression targets are standardised (TransformedTargetRegressor) so SVR, KNN and
MLP behave well regardless of the target's units.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

KINDS = ("tabular", "text", "image")


class SafeSVD(BaseEstimator, TransformerMixin):
    """TruncatedSVD that never asks for more components than the data allows."""

    def __init__(self, n_components: int = 300, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        k = max(1, min(self.n_components, X.shape[1] - 1, X.shape[0] - 1))
        self.svd_ = TruncatedSVD(n_components=k, random_state=self.random_state).fit(X)
        return self

    def transform(self, X):
        return self.svd_.transform(X)


def _to_text(X):
    """Column → 1-D array of strings (used inside ColumnTransformer for text columns)."""
    arr = np.asarray(X, dtype=object).reshape(-1)
    return np.array(["" if v is None or v != v else str(v) for v in arr], dtype=object)


@dataclass(frozen=True)
class Algorithm:
    id: str
    name: str
    family: str
    tasks: tuple[str, ...]
    kinds: tuple[str, ...]
    speed: str  # fast | medium | slow
    description: str
    make: Callable[[str, str, int], object]  # (task, kind, seed) -> estimator
    grid: dict = field(default_factory=dict)  # tuning space (estimator parameter -> values)
    needs_dense: bool = False
    requires: str | None = None  # optional python package
    default: bool = False

    def available(self) -> bool:
        return self.requires is None or importlib.util.find_spec(self.requires) is not None

    def supports(self, kind: str, task: str) -> bool:
        return kind in self.kinds and task in self.tasks

    def summary(self) -> dict:
        return {"id": self.id, "name": self.name, "family": self.family, "tasks": list(self.tasks),
                "kinds": list(self.kinds), "speed": self.speed, "description": self.description,
                "tunable": bool(self.grid), "available": self.available(), "requires": self.requires,
                "default": self.default}


def _xgb(task: str, kind: str, seed: int):
    import xgboost as xgb

    cls = xgb.XGBClassifier if task == "classification" else xgb.XGBRegressor
    return cls(n_estimators=300, learning_rate=0.1, max_depth=6, tree_method="hist", random_state=seed, n_jobs=-1)


def _lgbm(task: str, kind: str, seed: int):
    import lightgbm as lgb

    cls = lgb.LGBMClassifier if task == "classification" else lgb.LGBMRegressor
    return cls(n_estimators=300, random_state=seed, verbose=-1)


def _c(task, clf, reg):
    return clf if task == "classification" else reg


ALL_TASKS = ("classification", "regression")
ALGORITHMS: dict[str, Algorithm] = {a.id: a for a in [
    Algorithm("dummy", "Baseline (majority class / mean)", "baseline", ALL_TASKS, KINDS, "fast",
              "Always predicts the most frequent class (or the mean). Any useful model must beat it.",
              lambda t, k, s: _c(t, DummyClassifier(strategy="most_frequent"), DummyRegressor(strategy="mean")),
              default=True),
    Algorithm("logistic_regression", "Logistic Regression", "linear", ("classification",), KINDS, "fast",
              "Linear model with calibrated probabilities. Strong, interpretable baseline.",
              lambda t, k, s: LogisticRegression(C=10.0 if k == "text" else 1.0, max_iter=3000),
              grid={"C": [0.01, 0.1, 1.0, 10.0, 100.0]}, default=True),
    Algorithm("linear_svm", "Linear SVM", "linear", ("classification",), KINDS, "fast",
              "Maximum-margin linear classifier; excellent for text.",
              lambda t, k, s: LinearSVC(C=1.0, random_state=s), grid={"C": [0.01, 0.1, 1.0, 10.0]}),
    Algorithm("linear_regression", "Linear Regression", "linear", ("regression",), ("tabular", "image"), "fast",
              "Ordinary least squares.", lambda t, k, s: LinearRegression(), default=True),
    Algorithm("ridge", "Ridge Regression", "linear", ("regression",), KINDS, "fast",
              "Linear regression with L2 regularisation.", lambda t, k, s: Ridge(alpha=1.0),
              grid={"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}, default=True),
    Algorithm("svm_rbf", "SVM (RBF kernel)", "kernel", ALL_TASKS, KINDS, "medium",
              "Non-linear support vector machine. Slow above ~20k rows.",
              lambda t, k, s: _c(t, SVC(C=1.0, gamma="scale", random_state=s), SVR(C=1.0)),
              grid={"C": [0.1, 1.0, 10.0, 100.0], "gamma": ["scale", 0.01, 0.001]}, needs_dense=True),
    Algorithm("knn", "k-Nearest Neighbours", "instance", ALL_TASKS, KINDS, "fast",
              "Predicts from the most similar training examples.",
              lambda t, k, s: _c(t, KNeighborsClassifier(n_neighbors=5, **({"metric": "cosine", "algorithm": "brute"}
                                                                               if k == "text" else {})),
                                 KNeighborsRegressor(n_neighbors=5, **({"metric": "cosine", "algorithm": "brute"}
                                                                       if k == "text" else {}))),
              grid={"n_neighbors": [3, 5, 7, 11, 15, 25], "weights": ["uniform", "distance"]}),
    Algorithm("naive_bayes", "Naive Bayes", "probabilistic", ("classification",), KINDS, "fast",
              "Multinomial NB for text, Gaussian NB otherwise. Very fast.",
              lambda t, k, s: MultinomialNB(alpha=0.3) if k == "text" else GaussianNB(),
              grid={}),
    Algorithm("decision_tree", "Decision Tree", "tree", ALL_TASKS, KINDS, "fast",
              "A single interpretable tree.",
              lambda t, k, s: _c(t, DecisionTreeClassifier(random_state=s), DecisionTreeRegressor(random_state=s)),
              grid={"max_depth": [None, 4, 8, 16, 32], "min_samples_leaf": [1, 2, 5, 10]}),
    Algorithm("random_forest", "Random Forest", "ensemble", ALL_TASKS, KINDS, "medium",
              "Bagged decision trees; robust default for tabular data.",
              lambda t, k, s: _c(t, RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=s),
                                 RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=s)),
              grid={"n_estimators": [100, 200, 400], "max_depth": [None, 10, 20],
                    "min_samples_leaf": [1, 2, 5], "max_features": ["sqrt", 0.5, 1.0]}, default=True),
    Algorithm("extra_trees", "Extra Trees", "ensemble", ALL_TASKS, KINDS, "medium",
              "Extremely randomised trees; often as good as a random forest, faster to train.",
              lambda t, k, s: _c(t, ExtraTreesClassifier(n_estimators=200, n_jobs=-1, random_state=s),
                                 ExtraTreesRegressor(n_estimators=200, n_jobs=-1, random_state=s)),
              grid={"n_estimators": [100, 200, 400], "max_depth": [None, 10, 20], "min_samples_leaf": [1, 2, 5]}),
    Algorithm("gradient_boosting", "Gradient Boosting (histogram)", "boosting", ALL_TASKS, KINDS, "medium",
              "scikit-learn's LightGBM-style boosting; usually the strongest on tabular data.",
              lambda t, k, s: _c(t, HistGradientBoostingClassifier(random_state=s),
                                 HistGradientBoostingRegressor(random_state=s)),
              grid={"learning_rate": [0.03, 0.1, 0.3], "max_leaf_nodes": [15, 31, 63], "l2_regularization": [0, 0.1, 1.0]},
              needs_dense=True, default=True),
    Algorithm("mlp", "Neural network (MLP)", "neural", ALL_TASKS, KINDS, "medium",
              "Small fully-connected neural network (one hidden layer of 128 units).",
              lambda t, k, s: _c(t, MLPClassifier(hidden_layer_sizes=(128,), max_iter=300, early_stopping=True,
                                                  random_state=s),
                                 MLPRegressor(hidden_layer_sizes=(128,), max_iter=500, early_stopping=True,
                                              random_state=s)),
              grid={"hidden_layer_sizes": [(64,), (128,), (256,), (128, 64)], "alpha": [1e-5, 1e-4, 1e-3, 1e-2]}),
    Algorithm("xgboost", "XGBoost", "boosting", ALL_TASKS, KINDS, "medium",
              "Popular gradient-boosting library (optional: pip install xgboost).", _xgb,
              grid={"learning_rate": [0.03, 0.1, 0.3], "max_depth": [3, 6, 9], "n_estimators": [200, 400]},
              needs_dense=True, requires="xgboost"),
    Algorithm("lightgbm", "LightGBM", "boosting", ALL_TASKS, KINDS, "medium",
              "Fast gradient boosting from Microsoft (optional: pip install lightgbm).", _lgbm,
              grid={"learning_rate": [0.03, 0.1, 0.3], "num_leaves": [15, 31, 63], "n_estimators": [200, 400]},
              needs_dense=True, requires="lightgbm"),
]}


def catalog(kind: str | None = None, task: str | None = None) -> list[dict]:
    return [a.summary() for a in ALGORITHMS.values()
            if (kind is None or kind in a.kinds) and (task is None or task in a.tasks)]


# ----------------------------------------------------------------------------- pipelines
def _tabular_pre(types: dict[str, str], features: list[str]) -> ColumnTransformer:
    num = [c for c in features if types.get(c) == "numeric"]
    cat = [c for c in features if types.get(c) in ("categorical", "boolean")]
    txt = [c for c in features if types.get(c) == "text"]
    parts = []
    if num:
        parts.append(("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                                       ("scale", StandardScaler())]), num))
    if cat:
        parts.append(("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                       ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=30,
                                                                sparse_output=False))]), cat))
    for c in txt:
        parts.append((f"txt_{c}", Pipeline([("to_text", FunctionTransformer(_to_text)),
                                            ("tfidf", TfidfVectorizer(max_features=300, sublinear_tf=True))]), [c]))
    return ColumnTransformer(parts, remainder="drop", sparse_threshold=0.0)


def _text_features(algo: Algorithm, seed: int):
    word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1, max_features=50_000)
    if algo.id == "naive_bayes":
        return word
    if algo.needs_dense:
        return Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=20_000)),
                         ("svd", SafeSVD(300, seed))])
    if algo.id == "mlp":
        return TfidfVectorizer(sublinear_tf=True, max_features=10_000)
    return FeatureUnion([("word", word), ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                                                  sublinear_tf=True, max_features=50_000))])


def build_pipeline(algo: Algorithm, task: str, kind: str, *, seed: int = 42, types: dict[str, str] | None = None,
                   features: list[str] | None = None) -> Pipeline:
    if kind == "tabular":
        pre = _tabular_pre(types or {}, features or [])
    elif kind == "text":
        pre = _text_features(algo, seed)
    elif kind == "image":
        pre = StandardScaler()
    else:
        raise ValueError(f"unknown kind {kind}")
    est = algo.make(task, kind, seed)
    if task == "regression" and algo.id != "dummy":
        est = TransformedTargetRegressor(regressor=est, transformer=StandardScaler())
    return Pipeline([("pre", pre), ("model", est)])


def param_space(algo: Algorithm, task: str) -> dict:
    prefix = "model__regressor__" if task == "regression" and algo.id != "dummy" else "model__"
    return {prefix + k: v for k, v in algo.grid.items()}
