"""Leakage and data-quality checks for Model Arena datasets.

All checks are heuristics that *flag* problems for a human to inspect; a clean
report does not prove a dataset is leakage-free.

* Exact and near-duplicate examples across train / validation / test
  - text: identical after normalisation (case, whitespace, punctuation) and
    near-identical by character n-gram TF-IDF cosine similarity (≥ 0.9);
  - tabular: identical after rounding numbers to 4 significant digits and
    normalising strings;
  - images: identical files (SHA-1) and perceptual near-duplicates (64-bit
    difference hash, Hamming distance ≤ 4).
* Group overlap: the same group id in two splits.
* Target-leakage probe: a single column that predicts the target almost
  perfectly on its own (≥ 0.98 balanced accuracy / R² in 3-fold CV on the
  training rows) is suspicious – it may be derived from the outcome or
  recorded after it.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

import numpy as np
import pandas as pd

TEXT_NEAR_THRESHOLD = 0.9
DHASH_MAX_DISTANCE = 4
PROBE_THRESHOLD = 0.98
MAX_TEXT_CHECK = 3000
PAIRS = (("train", "test"), ("train", "val"), ("val", "test"))


# ----------------------------------------------------------------------------- normalisation
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = _PUNCT.sub(" ", s)
    return _WS.sub(" ", s).strip()


def _hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8"), usedforsecurity=False).hexdigest()


def tabular_keys(frame: pd.DataFrame) -> list[str]:
    def norm(v) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "∅"
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            return f"{float(v):.4g}"
        return normalize_text(v)

    return [_hash("|".join(norm(v) for v in row)) for row in frame.itertuples(index=False, name=None)]


# ----------------------------------------------------------------------------- overlap counting
def _split_map(splits: dict[str, list[int]]) -> dict[str, np.ndarray]:
    return {k: np.asarray(v, dtype=int) for k, v in splits.items() if v is not None and len(v)}


def key_overlap(keys: list[str], splits: dict[str, list[int]]) -> dict[str, int]:
    """Rows of the later split whose key also occurs in the earlier split."""
    s = _split_map(splits)
    out = {}
    for a, b in PAIRS:
        if a in s and b in s:
            seen = {keys[i] for i in s[a]}
            out[f"{a}-{b}"] = int(sum(keys[i] in seen for i in s[b]))
    return out


def text_near_duplicates(texts: list[str], splits: dict[str, list[int]], seed: int = 0) -> dict:
    """Exact-after-normalisation and TF-IDF near-duplicates between splits."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    norm = [normalize_text(t) for t in texts]
    exact = key_overlap([_hash(t) for t in norm], splits)
    s = _split_map(splits)
    near: dict[str, int] = {}
    checked: dict[str, int] = {}
    examples: list[dict] = []
    rng = np.random.default_rng(seed)
    for a, b in PAIRS:
        if a not in s or b not in s:
            continue
        ref, probe = s[a], s[b]
        if len(probe) > MAX_TEXT_CHECK:
            probe = np.sort(rng.choice(probe, MAX_TEXT_CHECK, replace=False))
        if len(ref) > 50_000:
            ref = np.sort(rng.choice(ref, 50_000, replace=False))
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=200_000,
                              sublinear_tf=True)
        try:
            R = vec.fit_transform([norm[i] for i in ref])
            P = vec.transform([norm[i] for i in probe])
        except ValueError:  # empty vocabulary
            continue
        hits = 0
        for start in range(0, P.shape[0], 256):
            sims = (P[start:start + 256] @ R.T).toarray()
            best = sims.max(axis=1)
            arg = sims.argmax(axis=1)
            for j in np.flatnonzero(best >= TEXT_NEAR_THRESHOLD):
                hits += 1
                if len(examples) < 6:
                    examples.append({"pair": f"{a}-{b}", "row": int(probe[start + j]), "similar_to": int(ref[arg[j]]),
                                     "similarity": round(float(best[j]), 3)})
        near[f"{a}-{b}"] = hits
        checked[f"{a}-{b}"] = int(len(probe))
    return {"method": "text", "exact_normalized": exact, "near": near, "checked": checked,
            "threshold": TEXT_NEAR_THRESHOLD, "examples": examples}


def dhash(path, size: int = 8) -> str | None:
    """64-bit difference hash (hex) of an image, robust to resizing and re-encoding."""
    from PIL import Image

    try:
        with Image.open(path) as im:
            g = im.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
            px = np.asarray(g, dtype=np.int16)
    except Exception:  # noqa: BLE001 - unreadable images are reported elsewhere
        return None
    bits = (px[:, 1:] > px[:, :-1]).ravel()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _popcount64(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.uint64)
    return _POP8[x.view(np.uint8)].reshape(*x.shape, 8).sum(axis=-1, dtype=np.int32)


def image_near_duplicates(hashes: list[str | None], splits: dict[str, list[int]]) -> dict:
    s = _split_map(splits)
    vals = np.array([int(h, 16) if h else 0 for h in hashes], dtype=np.uint64)
    valid = np.array([h is not None for h in hashes])
    near: dict[str, int] = {}
    examples: list[dict] = []
    for a, b in PAIRS:
        if a not in s or b not in s:
            continue
        ref = s[a][valid[s[a]]]
        probe = s[b][valid[s[b]]]
        if not len(ref) or not len(probe):
            continue
        hits = 0
        for start in range(0, len(probe), 128):
            chunk = probe[start:start + 128]
            dist = _popcount64(vals[chunk][:, None] ^ vals[ref][None, :])
            best = dist.min(axis=1)
            arg = dist.argmin(axis=1)
            for j in np.flatnonzero(best <= DHASH_MAX_DISTANCE):
                hits += 1
                if len(examples) < 6:
                    examples.append({"pair": f"{a}-{b}", "row": int(chunk[j]), "similar_to": int(ref[arg[j]]),
                                     "hamming": int(best[j])})
        near[f"{a}-{b}"] = hits
    return {"method": "dhash", "near": near, "max_distance": DHASH_MAX_DISTANCE, "examples": examples}


def group_overlap(groups: np.ndarray | None, splits: dict[str, list[int]]) -> dict[str, int] | None:
    if groups is None:
        return None
    s = _split_map(splits)
    out = {}
    for a, b in PAIRS:
        if a in s and b in s:
            out[f"{a}-{b}"] = len(set(groups[s[a]].tolist()) & set(groups[s[b]].tolist()))
    return out


# ----------------------------------------------------------------------------- target-leakage probe
def single_feature_probe(X: pd.DataFrame, y: np.ndarray, task: str, types: dict[str, str], train_idx: list[int],
                         seed: int = 42, max_rows: int = 5000, max_features: int = 100) -> dict:
    """3-fold CV score of a depth-4 decision tree on each column alone (training rows only)."""
    from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    rng = np.random.default_rng(seed)
    rows = np.asarray(train_idx, dtype=int)
    if len(rows) > max_rows:
        rows = np.sort(rng.choice(rows, max_rows, replace=False))
    yy = y[rows]
    if task == "classification":
        _, counts = np.unique(yy, return_counts=True)
        folds = int(min(3, counts.min())) if len(counts) > 1 else 0
        if folds < 2:
            return {"checked": [], "suspicious": [], "note": "too few rows per class for the probe"}
        cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
        model, scoring = DecisionTreeClassifier(max_depth=4, random_state=seed), "balanced_accuracy"
    else:
        cv = KFold(3, shuffle=True, random_state=seed)
        model, scoring = DecisionTreeRegressor(max_depth=4, random_state=seed), "r2"
    scores = []
    for col in list(X.columns)[:max_features]:
        typ = types.get(col, "categorical")
        if typ == "text":
            continue
        s = X[col].iloc[rows]
        if typ == "numeric":
            v = pd.to_numeric(s, errors="coerce").astype(float)
            feat = v.fillna(v.median() if v.notna().any() else 0.0).to_numpy().reshape(-1, 1)
        else:
            top = s.astype(str).value_counts().index[:50]
            mapping = {c: i for i, c in enumerate(top)}
            feat = s.astype(str).map(lambda c, m=mapping: m.get(c, -1)).to_numpy(dtype=float).reshape(-1, 1)
        if np.unique(feat).size < 2:
            continue
        try:
            score = float(np.mean(cross_val_score(model, feat, yy, cv=cv, scoring=scoring)))
        except ValueError:
            continue
        scores.append({"column": col, "score": round(score, 4)})
    scores.sort(key=lambda r: -r["score"])
    suspicious = [r for r in scores if r["score"] >= PROBE_THRESHOLD]
    return {"metric": scoring, "threshold": PROBE_THRESHOLD, "rows_used": int(len(rows)), "top": scores[:5],
            "suspicious": suspicious}


# ----------------------------------------------------------------------------- warnings
def overlap_warnings(report: dict, n_by_split: dict[str, int], what: str = "rows") -> list[str]:
    out = []
    for kind, label in (("exact", "identical"), ("exact_normalized", "identical after normalisation"),
                        ("near", "near-duplicates")):
        for pair, count in (report.get(kind) or {}).items():
            if count:
                a, b = pair.split("-")
                denom = (report.get("checked") or {}).get(pair) or n_by_split.get(b) or 1
                out.append(f"Leakage check: {count} {b} {what} ({count / denom:.0%} of those checked) are "
                           f"{label} to {a} {what}.")
    return out
