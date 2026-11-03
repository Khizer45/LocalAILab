"""Dataset ingestion for the Model Arena.

Supported uploads
* **CSV** – tabular data (numeric / categorical / short text columns) or a text
  dataset (one long-text column + a label). The user picks the target column;
  everything else is inferred and can be overridden.
* **ZIP of images** – one folder per class (``cat/1.jpg``), optionally already
  split (``train/cat/1.jpg`` + ``test/cat/9.jpg``).

Every dataset gets a fixed, reproducible **train / validation / test** split
(seed stored with the dataset):

* ``random`` – stratified for classification;
* ``group``  – all rows of a group (customer, patient, document, photo session…)
  stay in the same split, so the model is tested on unseen groups;
* ``time``   – rows are ordered by a date/time column; the newest rows are the
  test split and the ones before them the validation split (no future data in
  training);
* ``provided`` – image ZIPs with ``train/``, optional ``val/`` and ``test/`` folders.

Models are fitted on train, chosen on validation and the chosen model is
measured once on test (see ``app.arena.stats``). Each dataset also gets a
profile with data-quality and leakage warnings (exact and near-duplicates across
splits, group overlap, single columns that predict the target almost perfectly,
class imbalance, missing values…).

Storage (``data/arena/datasets/<id>/``): the original CSV as ``data.csv`` or the
extracted images under ``images/`` plus ``manifest.json``, and ``split.json``
with the row indices of every split and the split's SHA-256.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import warnings
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from app.core.errors import ValidationFailed

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff"}
SPLIT_DIRS = {"train": "train", "training": "train", "test": "test", "testing": "test",
              "val": "val", "valid": "val", "validation": "val", "dev": "val"}
SPLIT_STRATEGIES = ("random", "group", "time")
_DATE_LIKE = re.compile(r"^\s*(\d{4}[-/.]\d{1,2}[-/.]\d{1,2}([ T]\d{1,2}:\d{2}(:\d{2}(\.\d+)?)?)?"
                        r"|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}( \d{1,2}:\d{2}(:\d{2})?)?)\s*(Z|[+-]\d{2}:?\d{2})?\s*$")
TARGET_HINTS = ("label", "target", "class", "y", "category", "intent", "outcome", "species", "churn", "price",
                "sentiment", "diagnosis", "result")
ID_NAME = re.compile(r"(^|[_\s-])(id|uuid|guid|index|idx|row)$|^unnamed: ?\d+$", re.IGNORECASE)
MAX_IMAGE_FILE = 25 * 1024 * 1024
MAX_ZIP_TOTAL = 4 * 1024**3


# ----------------------------------------------------------------------------- CSV helpers
def decode_text(raw: bytes) -> str:
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ValidationFailed("The file is not a readable text/CSV file.")


def _delimiter(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        counts = {d: sample.count(d) for d in (",", ";", "\t", "|")}
        return max(counts, key=counts.get) if any(counts.values()) else ","


def read_csv(raw: bytes, *, max_rows: int, max_columns: int) -> pd.DataFrame:
    text = decode_text(raw)
    if not text.strip():
        raise ValidationFailed("The CSV file is empty.")
    sep = _delimiter(text[:65536])
    try:
        df = pd.read_csv(io.StringIO(text), sep=sep, nrows=max_rows + 1, low_memory=False, skipinitialspace=True)
    except (pd.errors.ParserError, ValueError) as exc:
        raise ValidationFailed(f"Could not parse the CSV file: {str(exc)[:200]}",
                               hint="Save it as a comma-separated UTF-8 CSV with a header row.") from exc
    if len(df) > max_rows:
        raise ValidationFailed(f"The CSV has more than {max_rows:,} rows.",
                               hint="Sample it down or raise ARENA_MAX_ROWS in the backend .env file.")
    if df.shape[1] > max_columns:
        raise ValidationFailed(f"The CSV has {df.shape[1]} columns (limit {max_columns}).")
    if df.shape[1] < 2:
        raise ValidationFailed("The CSV needs at least two columns: one or more features and a target.",
                               hint="Check the delimiter – the file was read as a single column.")
    cols, seen = [], set()
    for i, c in enumerate(df.columns):
        name = str(c).strip() or f"column_{i + 1}"
        while name in seen:
            name += "_dup"
        seen.add(name)
        cols.append(name)
    df.columns = cols
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def column_type(s: pd.Series) -> str:
    non_null = s.dropna()
    if non_null.empty:
        return "empty"
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    strs = non_null.astype(str)
    if looks_datetime(strs):
        return "datetime"
    lowered = set(strs.str.lower().unique()[:5])
    if len(lowered) <= 2 and lowered <= {"true", "false", "yes", "no", "0", "1"}:
        return "boolean"
    avg_len = float(strs.str.len().mean())
    words = float(strs.str.split().str.len().mean())
    uniq = strs.nunique() / max(1, len(strs))
    if avg_len >= 25 and words >= 3 and uniq >= 0.3:
        return "text"
    return "categorical"


def looks_datetime(strs: pd.Series) -> bool:
    sample = strs.head(300)
    if sample.empty or sample.str.match(_DATE_LIKE).mean() < 0.9:
        return False
    return bool(parse_times(sample).notna().mean() >= 0.9)


def parse_times(s: pd.Series) -> pd.Series:
    """Datetimes (or numbers, e.g. Unix time / year) as sortable values; NaN/NaT when unparseable."""
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(s, errors="coerce", format="mixed", utc=True)


def is_id_like(name: str, s: pd.Series) -> bool:
    non_null = s.dropna()
    if len(non_null) < 10 or non_null.nunique() != len(non_null):
        return False
    if ID_NAME.search(name.strip()):
        return True
    if pd.api.types.is_integer_dtype(non_null):  # 0..n-1 / 1..n row counters
        v = np.sort(non_null.to_numpy())
        return bool(np.all(np.diff(v) == 1))
    return False


def label_str(v) -> str:
    """Canonical text form of a class label (1.0 → "1", " cat " → "cat")."""
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return str(int(v)) if float(v).is_integer() else repr(float(v))
    return str(v).strip()


def suggest_target(df: pd.DataFrame) -> str:
    lower = {c.lower(): c for c in df.columns}
    for hint in TARGET_HINTS:
        if hint in lower:
            return lower[hint]
    return df.columns[-1]


def infer_task(s: pd.Series) -> str:
    non_null = s.dropna()
    if pd.api.types.is_numeric_dtype(non_null) and not pd.api.types.is_bool_dtype(non_null):
        n_unique = non_null.nunique()
        has_fraction = bool((non_null.astype(float) % 1 != 0).any())
        if n_unique > 20 or (has_fraction and n_unique > 10):
            return "regression"
    return "classification"


def describe_columns(df: pd.DataFrame, limit: int = 3) -> list[dict]:
    out = []
    for c in df.columns:
        s = df[c]
        typ = "id" if is_id_like(c, s) else column_type(s)
        examples = [str(v)[:60] for v in s.dropna().unique()[:limit]]
        out.append({"name": c, "type": typ, "missing": int(s.isna().sum()), "unique": int(s.nunique(dropna=True)),
                    "examples": examples})
    return out


# ----------------------------------------------------------------------------- dataset specs
@dataclass
class CsvConfig:
    target: str
    task: str = "auto"  # auto | classification | regression
    text_column: str | None = None
    ignore: list[str] = field(default_factory=list)
    test_size: float = 0.2
    val_size: float = 0.2
    seed: int = 42
    split_strategy: str = "random"  # random | group | time
    group_column: str | None = None
    time_column: str | None = None


@dataclass
class Prepared:
    kind: str
    task: str
    target: str | None
    text_column: str | None
    features: list[str]
    ignored: list[str]
    types: dict[str, str]
    labels: list[str]
    n_rows: int
    train_idx: list[int]
    test_idx: list[int]
    split_source: str
    profile: dict
    warnings: list[str]
    val_idx: list[int] = field(default_factory=list)
    split_strategy: str = "random"
    group_column: str | None = None
    time_column: str | None = None
    groups: np.ndarray | None = None


def resolve_csv_schema(df: pd.DataFrame, cfg: CsvConfig) -> tuple[str, str, str | None, list[str], list[str], dict]:
    """Decide task, kind, text column, features and ignored columns (deterministic)."""
    if cfg.target not in df.columns:
        raise ValidationFailed(f"Target column '{cfg.target}' is not in the CSV.")
    task = cfg.task if cfg.task in ("classification", "regression") else infer_task(df[cfg.target])
    if task == "regression" and not pd.api.types.is_numeric_dtype(df[cfg.target].dropna()):
        converted = pd.to_numeric(df[cfg.target], errors="coerce")
        if converted.notna().mean() < 0.95:
            raise ValidationFailed("Regression needs a numeric target column.",
                                   hint="Choose classification, or pick a numeric target.")
    if cfg.split_strategy not in SPLIT_STRATEGIES:
        raise ValidationFailed(f"Unknown split strategy '{cfg.split_strategy}'.")
    special = {}
    if cfg.split_strategy == "group":
        if not cfg.group_column or cfg.group_column not in df.columns or cfg.group_column == cfg.target:
            raise ValidationFailed("A group split needs a group column (e.g. customer_id, patient_id, document).")
        special[cfg.group_column] = "group"
    if cfg.split_strategy == "time":
        if not cfg.time_column or cfg.time_column not in df.columns or cfg.time_column == cfg.target:
            raise ValidationFailed("A time split needs a date/time column.")
        special[cfg.time_column] = "time"
    types: dict[str, str] = {}
    ignored: list[str] = []
    features: list[str] = []
    for c in df.columns:
        if c == cfg.target:
            continue
        s = df[c]
        typ = "id" if is_id_like(c, s) else column_type(s)
        types[c] = typ
        if c in special or c in cfg.ignore or typ in ("id", "empty", "datetime") or s.nunique(dropna=True) <= 1:
            ignored.append(c)
        else:
            features.append(c)
    if cfg.text_column:
        if cfg.text_column not in features:
            raise ValidationFailed(f"Text column '{cfg.text_column}' is not a usable feature column.")
        types[cfg.text_column] = "text"
    if not features:
        raise ValidationFailed("No usable feature columns remain after removing the target, ID-like, empty and "
                               "constant columns.")
    text_cols = [c for c in features if types[c] == "text"]
    text_column = cfg.text_column
    if text_column is None and len(features) == 1 and len(text_cols) == 1:
        text_column = text_cols[0]
    kind = "text" if text_column and features == [text_column] else "tabular"
    if kind == "tabular" and text_column:  # text column chosen but other features exist → text-only dataset
        kind = "text"
        ignored += [c for c in features if c != text_column]
        features = [text_column]
    return task, kind, text_column, features, ignored, types


def clean_frame(df: pd.DataFrame, target: str, task: str, *, group_column: str | None = None,
                time_column: str | None = None) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Drop rows without a target / group / time and (classification) classes with < 2 rows. Deterministic."""
    warnings: list[str] = []
    n0 = len(df)
    df = df[df[target].notna()].copy()
    if len(df) < n0:
        warnings.append(f"{n0 - len(df)} rows without a target value were dropped.")
    if group_column:
        n1 = len(df)
        df = df[df[group_column].notna()]
        if len(df) < n1:
            warnings.append(f"{n1 - len(df)} rows without a group id were dropped.")
    if time_column:
        n1 = len(df)
        df = df[parse_times(df[time_column]).notna().to_numpy()]
        if len(df) < n1:
            warnings.append(f"{n1 - len(df)} rows with a missing or unreadable time were dropped.")
    if task == "regression":
        df[target] = pd.to_numeric(df[target], errors="coerce")
        bad = int(df[target].isna().sum())
        if bad:
            df = df[df[target].notna()]
            warnings.append(f"{bad} rows with a non-numeric target were dropped.")
        labels: list[str] = []
    else:
        df[target] = df[target].map(label_str)
        counts = df[target].value_counts()
        rare = sorted(counts[counts < 2].index.tolist())
        if rare:
            df = df[~df[target].isin(rare)]
            warnings.append(f"{len(rare)} class(es) with a single example were dropped: {', '.join(rare[:8])}.")
        labels = sorted(df[target].unique().tolist())
        if len(labels) < 2:
            raise ValidationFailed("Classification needs at least two classes with two or more examples each.")
        if len(labels) > 200:
            raise ValidationFailed(f"The target has {len(labels)} classes (limit 200).",
                                   hint="Did you pick an ID or free-text column as the target? Or choose regression.")
    source_rows = [int(i) for i in df.index]  # data-row numbers in the original file (header not counted)
    df = df.reset_index(drop=True)
    df.attrs["source_rows"] = source_rows
    if len(df) < 10:
        raise ValidationFailed("At least 10 usable rows are required.")
    return df, labels, warnings


def make_split(y: np.ndarray, task: str, test_size: float, seed: int) -> tuple[list[int], list[int]]:
    tr, te = _random_part(np.arange(len(y)), y, task, test_size, seed)
    return sorted(tr.tolist()), sorted(te.tolist())


def _random_part(idx: np.ndarray, y: np.ndarray, task: str, frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split ``idx`` into (rest, part) with ``part`` ≈ frac of ``idx`` (stratified when possible)."""
    yy = y[idx]
    strat = yy if task == "classification" else None
    if strat is not None:
        n_part = int(np.ceil(frac * len(idx)))
        _, counts = np.unique(yy, return_counts=True)
        if n_part < len(counts) or counts.min() < 2:  # too small to stratify: plain shuffle
            strat = None
    rest, part = train_test_split(idx, test_size=frac, random_state=seed, stratify=strat)
    return rest, part


def _group_part(idx: np.ndarray, y: np.ndarray, groups: np.ndarray, task: str, frac: float,
                seed: int) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

    g = groups[idx]
    n_groups = len(np.unique(g))
    if n_groups < 3:
        raise ValidationFailed(f"A group split needs at least 3 groups; the group column has {n_groups}.")
    if task == "classification":
        n_splits = max(2, min(n_groups, int(round(1 / frac))))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rest, part = next(StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(idx, y[idx], g))
            return idx[rest], idx[part]
        except ValueError:
            pass
    rest, part = next(GroupShuffleSplit(1, test_size=frac, random_state=seed).split(idx, groups=g))
    return idx[rest], idx[part]


def _time_part(idx: np.ndarray, order_key: np.ndarray, frac: float) -> tuple[np.ndarray, np.ndarray]:
    ordered = idx[np.argsort(order_key[idx], kind="stable")]
    n_part = max(1, int(np.ceil(frac * len(idx))))
    return ordered[:-n_part], ordered[-n_part:]


def make_split3(y: np.ndarray, task: str, test_size: float, val_size: float, seed: int, *, strategy: str = "random",
                groups: np.ndarray | None = None, times: np.ndarray | None = None
                ) -> tuple[list[int], list[int], list[int]]:
    """Train / validation / test row indices. ``val_size`` and ``test_size`` are fractions of all rows."""
    if not 0.05 <= test_size <= 0.5:
        raise ValidationFailed("Test size must be between 5% and 50%.")
    if not 0 <= val_size <= 0.4 or test_size + val_size > 0.7:
        raise ValidationFailed("Validation size must be between 0% and 40% (test + validation at most 70%).")
    idx = np.arange(len(y))
    if strategy == "group":
        assert groups is not None
        rest, test = _group_part(idx, y, groups, task, test_size, seed)
        if val_size:
            train, val = _group_part(rest, y, groups, task, val_size / (1 - test_size), seed + 1)
        else:
            train, val = rest, np.array([], dtype=int)
    elif strategy == "time":
        assert times is not None
        rest, test = _time_part(idx, times, test_size)
        if val_size:
            train, val = _time_part(rest, times, val_size / (1 - test_size))
        else:
            train, val = rest, np.array([], dtype=int)
    else:
        rest, test = _random_part(idx, y, task, test_size, seed)
        if val_size:
            train, val = _random_part(rest, y, task, val_size / (1 - test_size), seed + 1)
        else:
            train, val = rest, np.array([], dtype=int)
    if len(train) < 2:
        raise ValidationFailed("Too few rows remain for training; lower the validation/test sizes.")
    return sorted(train.tolist()), sorted(val.tolist()), sorted(test.tolist())


def split_sha256(train: list[int], val: list[int], test: list[int]) -> str:
    payload = json.dumps({"train": list(map(int, train)), "val": list(map(int, val)), "test": list(map(int, test))},
                         separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def time_order_key(s: pd.Series) -> np.ndarray:
    t = parse_times(s)
    if pd.api.types.is_datetime64_any_dtype(t):
        return t.astype("int64").to_numpy()
    return t.to_numpy(dtype=float)


def coerce_features(df: pd.DataFrame, features: list[str], types: dict[str, str]) -> pd.DataFrame:
    """Return the feature frame with consistent dtypes for the ML pipelines."""
    out = pd.DataFrame(index=df.index)
    for c in features:
        typ = types.get(c, "categorical")
        if typ == "numeric":
            out[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
        elif typ == "text":
            out[c] = df[c].fillna("").astype(str)
        else:  # categorical / boolean → strings, missing stays NaN for the imputer
            out[c] = df[c].map(lambda v: np.nan if pd.isna(v) else label_str(v)).astype(object)
    return out


def _row_keys(frame: pd.DataFrame) -> pd.Series:
    return pd.util.hash_pandas_object(frame.astype(str), index=False)


def profile_csv(df: pd.DataFrame, kind: str, task: str, target: str, features: list[str], types: dict[str, str],
                splits: dict[str, list[int]], labels: list[str], *, groups: np.ndarray | None = None,
                seed: int = 42) -> tuple[dict, list[str]]:
    from app.arena import leakage as L

    warns: list[str] = []
    X = df[features]
    n = len(df)
    train_idx = splits["train"]
    feat_keys = _row_keys(X)
    full_keys = _row_keys(df[features + [target]])
    dup_rows = int(full_keys.duplicated().sum())
    if dup_rows:
        warns.append(f"{dup_rows} duplicate rows ({dup_rows / n:.0%}).")
    conflicts = 0
    if task == "classification":
        g = pd.DataFrame({"k": feat_keys, "y": df[target]}).groupby("k")["y"].nunique()
        conflicts = int((g > 1).sum())
        if conflicts:
            warns.append(f"{conflicts} identical inputs have different labels – no model can get all of them right.")
    exact = L.key_overlap([str(k) for k in feat_keys.tolist()], splits)
    leaked = exact.get("train-test", 0)
    n_by = {k: len(v) for k, v in splits.items()}
    for pair, count in exact.items():
        if count:
            a, b = pair.split("-")
            warns.append(f"Leakage: {count} {b} rows ({count / max(1, n_by.get(b, 1)):.0%}) have exactly the same "
                         f"inputs as {a} rows, so {b} scores will look better than real-world performance.")
    if kind == "text":
        near = L.text_near_duplicates(df[features[0]].astype(str).tolist(), splits, seed=seed)
        warns += [w for w in L.overlap_warnings({"exact_normalized": {k: v - exact.get(k, 0) for k, v in
                                                                      near["exact_normalized"].items()},
                                                 "near": {k: max(0, v - near["exact_normalized"].get(k, 0)) for k, v in
                                                          near["near"].items()},
                                                 "checked": near["checked"]}, n_by)]
    else:
        keys = L.tabular_keys(X)
        normalized = L.key_overlap(keys, splits)
        near = {"method": "tabular", "exact": exact, "near": normalized,
                "note": "near = identical after rounding numbers to 4 significant digits and normalising text"}
        warns += L.overlap_warnings({"near": {k: max(0, v - exact.get(k, 0)) for k, v in normalized.items()}}, n_by)
    group_report = L.group_overlap(groups, splits)
    if group_report and any(group_report.values()):
        warns.append("Group leakage: some groups appear in more than one split: "
                     + ", ".join(f"{k} {v}" for k, v in group_report.items() if v) + ".")
    probe = None
    if kind == "tabular":
        y_all = df[target].to_numpy()
        probe = L.single_feature_probe(df[features], y_all, task, types, train_idx, seed=seed)
        for r in probe.get("suspicious", []):
            warns.append(f"Possible target leakage: column '{r['column']}' alone predicts the target almost perfectly "
                         f"({probe['metric']} {r['score']:.3f} on training rows). Check that it is not derived from the "
                         "outcome or recorded after it; exclude it if so.")
    if n < 100:
        warns.append(f"Small dataset ({n} rows): expect wide confidence intervals.")
    if splits.get("val") is not None and 0 < len(splits["val"]) < 30:
        warns.append(f"Small validation split ({len(splits['val'])} rows): the recommendation will often be "
                     "'provisional' because differences cannot be resolved.")
    columns = []
    for c in features:
        s = df[c]
        miss = float(s.isna().mean())
        if miss > 0.5:
            warns.append(f"Column '{c}' is {miss:.0%} missing.")
        info: dict = {"name": c, "type": types.get(c), "missing_pct": round(miss * 100, 1),
                      "unique": int(s.nunique(dropna=True))}
        if types.get(c) == "numeric":
            num = pd.to_numeric(s, errors="coerce")
            info.update({"min": _f(num.min()), "max": _f(num.max()), "mean": _f(num.mean())})
        elif types.get(c) == "categorical" and info["unique"] > 50 and info["unique"] > 0.5 * n:
            warns.append(f"Column '{c}' has {info['unique']} distinct values; only the 30 most frequent are "
                         "one-hot encoded.")
        columns.append(info)
    profile: dict = {"rows": n, "features": len(features), "columns": columns, "duplicate_rows": dup_rows,
                     "conflicting_inputs": conflicts, "train_test_overlap": leaked,
                     "split_sizes": {k: len(v) for k, v in splits.items()},
                     "leakage": {"exact": exact, "near_duplicates": near, "group_overlap": group_report,
                                 "single_feature_probe": probe}}
    if task == "classification":
        counts = df[target].value_counts()
        profile["class_counts"] = {lab: int(counts.get(lab, 0)) for lab in labels}
        for part in ("val", "test"):
            if splits.get(part):
                profile[f"class_counts_{part}"] = {lab: int((df[target].iloc[splits[part]] == lab).sum())
                                                   for lab in labels}
        ratio = counts.max() / max(1, counts.min())
        profile["imbalance_ratio"] = round(float(ratio), 2)
        if ratio >= 5:
            warns.append(f"Class imbalance {ratio:.0f}:1 – compare models on macro-F1 and balanced accuracy, "
                         "not plain accuracy.")
        rare = [lab for lab in labels if splits.get("test") and profile["class_counts_test"].get(lab, 0) < 5]
        if rare:
            warns.append(f"{len(rare)} class(es) have fewer than 5 test examples ({', '.join(rare[:5])}); their "
                         "per-class scores are very uncertain.")
    else:
        y = df[target].astype(float)
        profile["target_stats"] = {"min": _f(y.min()), "max": _f(y.max()), "mean": _f(y.mean()), "std": _f(y.std())}
    if kind == "text":
        lengths = df[features[0]].astype(str).str.split().str.len()
        profile["text_words_median"] = _f(lengths.median())
    return profile, warns


def _f(v) -> float | None:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(v) else round(v, 4)


def prepare_csv(raw: bytes, cfg: CsvConfig, *, max_rows: int, max_columns: int) -> tuple[pd.DataFrame, Prepared]:
    df = read_csv(raw, max_rows=max_rows, max_columns=max_columns)
    task, kind, text_column, features, ignored, types = resolve_csv_schema(df, cfg)
    group_col = cfg.group_column if cfg.split_strategy == "group" else None
    time_col = cfg.time_column if cfg.split_strategy == "time" else None
    df, labels, warns = clean_frame(df, cfg.target, task, group_column=group_col, time_column=time_col)
    y = df[cfg.target].to_numpy()
    groups = df[group_col].map(label_str).to_numpy() if group_col else None
    times = time_order_key(df[time_col]) if time_col else None
    train_idx, val_idx, test_idx = make_split3(y, task, cfg.test_size, cfg.val_size, cfg.seed,
                                               strategy=cfg.split_strategy, groups=groups, times=times)
    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    profile, w2 = profile_csv(df, kind, task, cfg.target, features, types, splits, labels, groups=groups,
                              seed=cfg.seed)
    dt_cols = [c for c in ignored if types.get(c) == "datetime" and c != time_col]
    if dt_cols:
        warns.append(f"Date/time columns are not used as features: {', '.join(dt_cols[:5])}. "
                     "Use the 'time' split if the data should be evaluated on the newest rows.")
    if ignored:
        warns.append(f"Ignored columns (ID-like, empty, constant, date/time, group or excluded): "
                     f"{', '.join(ignored[:10])}.")
    if cfg.split_strategy == "time":
        t = parse_times(df[time_col])
        profile["time_range"] = {part: [_time_str(t.iloc[ix].min()), _time_str(t.iloc[ix].max())]
                                 for part, ix in splits.items() if len(ix)}
    return df, Prepared(kind=kind, task=task, target=cfg.target, text_column=text_column, features=features,
                        ignored=ignored, types=types, labels=labels, n_rows=len(df), train_idx=train_idx,
                        test_idx=test_idx, val_idx=val_idx, split_source=cfg.split_strategy, profile=profile,
                        warnings=warns + w2, split_strategy=cfg.split_strategy, group_column=group_col,
                        time_column=time_col, groups=groups)


def _time_str(v) -> str | None:
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)[:40]


def preview_csv(raw: bytes, *, max_rows: int, max_columns: int) -> dict:
    df = read_csv(raw, max_rows=max_rows, max_columns=max_columns)
    target = suggest_target(df)
    task = infer_task(df[target])
    cols = describe_columns(df)
    feature_types = [c["type"] for c in cols if c["name"] != target and c["type"] not in ("id", "empty")]
    kind = "text" if feature_types == ["text"] else "tabular"
    head = df.head(8).astype(object).where(df.head(8).notna(), None)
    return {"format": "csv", "rows": len(df), "columns": cols, "suggested_target": target, "suggested_task": task,
            "suggested_kind": kind, "head": [{k: (str(v)[:120] if v is not None else None) for k, v in r.items()}
                                             for r in head.to_dict(orient="records")]}


def load_csv_dataset(path: Path, target: str, task: str, features: list[str], types: dict[str, str], *,
                     max_rows: int, max_columns: int, group_column: str | None = None,
                     time_column: str | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """Re-create the cleaned frame (deterministic) and return (features, y)."""
    raw = path.read_bytes()
    df = read_csv(raw, max_rows=max_rows, max_columns=max_columns)
    df, _, _ = clean_frame(df, target, task, group_column=group_column, time_column=time_column)
    X = coerce_features(df, features, types)
    y = df[target].to_numpy(dtype=float) if task == "regression" else df[target].astype(str).to_numpy()
    return X, y


# ----------------------------------------------------------------------------- image ZIP
@dataclass
class ZipEntry:
    info: zipfile.ZipInfo
    parts: tuple[str, ...]


def _safe_parts(name: str) -> tuple[str, ...] | None:
    name = name.replace("\\", "/")
    if name.startswith("/") or re.match(r"^[A-Za-z]:", name):
        raise ValidationFailed("The ZIP contains an absolute path and was rejected.", code="unsafe_archive")
    parts = tuple(p for p in PurePosixPath(name).parts if p not in ("", "."))
    if any(p == ".." for p in parts):
        raise ValidationFailed("The ZIP contains a '..' path and was rejected.", code="unsafe_archive")
    if not parts or parts[0] == "__MACOSX" or any(p.startswith(".") for p in parts):
        return None
    if PurePosixPath(parts[-1]).suffix.lower() not in IMAGE_EXTS:
        return None
    return parts


def scan_zip(zf: zipfile.ZipFile, *, max_images: int) -> tuple[list[ZipEntry], str, list[str]]:
    """Return (entries with class/split parts, split_source, warnings). Validates sizes and paths."""
    entries: list[ZipEntry] = []
    total = 0
    skipped = 0
    for info in zf.infolist():
        if info.is_dir():
            continue
        parts = _safe_parts(info.filename)
        if parts is None:
            skipped += 1
            continue
        if info.file_size > MAX_IMAGE_FILE:
            raise ValidationFailed(f"Image '{info.filename[:80]}' is larger than 25 MB.")
        if info.compress_size and info.file_size / info.compress_size > 200 and info.file_size > 10 * 1024 * 1024:
            raise ValidationFailed("The ZIP looks like a compression bomb and was rejected.", code="unsafe_archive")
        total += info.file_size
        if total > MAX_ZIP_TOTAL:
            raise ValidationFailed("The images in the ZIP are larger than 4 GB in total.")
        entries.append(ZipEntry(info, parts))
        if len(entries) > max_images:
            raise ValidationFailed(f"The ZIP has more than {max_images:,} images.",
                                   hint="Use a subset or raise ARENA_MAX_IMAGES in the backend .env file.")
    if not entries:
        raise ValidationFailed("No images found in the ZIP.",
                               hint="Put the images in one folder per class, e.g. cats/1.jpg, dogs/2.jpg.")
    # strip a common wrapper folder (e.g. "my_dataset/cats/1.jpg")
    while all(len(e.parts) > 2 for e in entries) and len({e.parts[0] for e in entries}) == 1 \
            and e_first_not_split(entries):
        entries = [ZipEntry(e.info, e.parts[1:]) for e in entries]
    warnings = [f"{skipped} non-image or hidden files were ignored."] if skipped else []
    firsts = {e.parts[0].lower() for e in entries}
    if firsts <= set(SPLIT_DIRS):  # train/<class>/x.jpg layout
        if any(len(e.parts) < 3 for e in entries):
            raise ValidationFailed("With train/ and test/ folders, put images in train/<class>/ and test/<class>/.")
        if {"train", "test"} <= {SPLIT_DIRS[f] for f in firsts}:
            return entries, "provided", warnings
        warnings.append("No test/ folder found: the images were pooled and split randomly.")
        return entries, "pooled", warnings
    if any(len(e.parts) < 2 for e in entries):
        raise ValidationFailed("Some images are not inside a class folder.",
                               hint="Use one folder per class: <class_name>/<image>.jpg")
    return entries, "random", warnings


def e_first_not_split(entries: list[ZipEntry]) -> bool:
    return entries[0].parts[0].lower() not in SPLIT_DIRS


def entry_label(e: ZipEntry, split_source: str) -> tuple[str, str | None]:
    if split_source == "provided":
        return e.parts[1].strip(), SPLIT_DIRS[e.parts[0].lower()]
    if split_source == "pooled":
        return e.parts[1].strip(), None
    return e.parts[0].strip(), None


def preview_zip(path: Path, *, max_images: int) -> dict:
    try:
        with zipfile.ZipFile(path) as zf:
            entries, split_source, warnings = scan_zip(zf, max_images=max_images)
    except zipfile.BadZipFile as exc:
        raise ValidationFailed("The file is not a valid ZIP archive.") from exc
    counts: dict[str, int] = {}
    for e in entries:
        lab, _ = entry_label(e, split_source)
        counts[lab] = counts.get(lab, 0) + 1
    return {"format": "zip", "images": len(entries), "classes": dict(sorted(counts.items())),
            "split_source": split_source, "warnings": warnings}


def extract_images(path: Path, dest: Path, *, test_size: float, seed: int, max_images: int,
                   val_size: float = 0.2) -> Prepared:
    """Safely extract images to dest/images and build manifest + train/validation/test split."""
    from PIL import Image, UnidentifiedImageError

    from app.arena import leakage as L

    try:
        zf = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise ValidationFailed("The file is not a valid ZIP archive.") from exc
    img_dir = dest / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    items: list[dict] = []
    bad = 0
    sizes: list[tuple[int, int]] = []
    modes: dict[str, int] = {}
    hashes: list[str] = []
    with zf:
        entries, split_source, warnings = scan_zip(zf, max_images=max_images)
        for i, e in enumerate(entries):
            label, split = entry_label(e, split_source)
            data = zf.read(e.info)
            try:
                with Image.open(io.BytesIO(data)) as im:
                    im.verify()
                if len(sizes) < 400:
                    with Image.open(io.BytesIO(data)) as im:
                        sizes.append(im.size)
                        modes[im.mode] = modes.get(im.mode, 0) + 1
            except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
                bad += 1
                continue
            ext = PurePosixPath(e.parts[-1]).suffix.lower()
            rel = f"images/{i:06d}{ext}"
            (dest / rel).write_bytes(data)
            hashes.append(hashlib.sha1(data, usedforsecurity=False).hexdigest())
            items.append({"file": rel, "label": label, "split": split, "source": "/".join(e.parts)[:300]})
    if bad:
        warnings.append(f"{bad} files could not be read as images and were skipped.")
    labels_all = [it["label"] for it in items]
    counts = pd.Series(labels_all).value_counts()
    rare = sorted(counts[counts < 2].index.tolist())
    if rare:
        keep = [k for k, it in enumerate(items) if it["label"] not in rare]
        items = [items[k] for k in keep]
        hashes = [hashes[k] for k in keep]
        warnings.append(f"{len(rare)} class(es) with a single image were dropped: {', '.join(rare[:8])}.")
    labels = sorted({it["label"] for it in items})
    if len(labels) < 2:
        raise ValidationFailed("Image classification needs at least two class folders with 2+ images each.")
    y = np.array([it["label"] for it in items])
    if split_source == "provided":
        train_idx = [k for k, it in enumerate(items) if it["split"] == "train"]
        val_idx = [k for k, it in enumerate(items) if it["split"] == "val"]
        test_idx = [k for k, it in enumerate(items) if it["split"] == "test"]
        if not train_idx or not test_idx:
            raise ValidationFailed("Both the train/ and test/ folders must contain images.")
        if not val_idx and val_size:
            frac = min(0.5, val_size / (1 - test_size))
            rest, val = _random_part(np.asarray(train_idx), y, "classification", frac, seed + 1)
            train_idx, val_idx = sorted(rest.tolist()), sorted(val.tolist())
            warnings.append("No val/ folder: the validation images were taken from train/ (test/ is unchanged).")
    else:
        train_idx, val_idx, test_idx = make_split3(y, "classification", test_size, val_size, seed)
        split_source = "random"
    for part, ix in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        for k in ix:
            items[k]["split"] = part
    dhashes = [L.dhash(dest / it["file"]) for it in items]
    for it, h in zip(items, dhashes, strict=True):
        it["dhash"] = h
    (dest / "manifest.json").write_text(json.dumps({"items": items}), encoding="utf-8")

    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    dup = len(hashes) - len(set(hashes))
    if dup:
        warnings.append(f"{dup} duplicate images (identical files).")
    exact = L.key_overlap(hashes, splits)
    for pair, count in exact.items():
        if count:
            a, b = pair.split("-")
            warnings.append(f"Leakage: {count} {b} images are identical files to {a} images.")
    near = L.image_near_duplicates(dhashes, splits)
    n_by = {k: len(v) for k, v in splits.items()}
    warnings += L.overlap_warnings({"near": {k: max(0, v - exact.get(k, 0)) for k, v in near["near"].items()}},
                                   n_by, what="images")
    leaked = exact.get("train-test", 0)
    class_counts = {lab: int((y == lab).sum()) for lab in labels}
    ratio = max(class_counts.values()) / max(1, min(class_counts.values()))
    if ratio >= 5:
        warnings.append(f"Class imbalance {ratio:.0f}:1 – compare models on macro-F1 and balanced accuracy.")
    if len(items) < 100:
        warnings.append(f"Small dataset ({len(items)} images): expect wide confidence intervals.")
    if val_idx and len(val_idx) < 30:
        warnings.append(f"Small validation split ({len(val_idx)} images): the recommendation will often be "
                        "'provisional'.")
    w = [s[0] for s in sizes]
    h = [s[1] for s in sizes]
    profile = {"rows": len(items), "class_counts": class_counts,
               "class_counts_val": {lab: int((y[val_idx] == lab).sum()) for lab in labels} if val_idx else {},
               "class_counts_test": {lab: int((y[test_idx] == lab).sum()) for lab in labels},
               "imbalance_ratio": round(float(ratio), 2), "duplicate_images": dup, "train_test_overlap": leaked,
               "split_sizes": n_by, "leakage": {"exact": exact, "near_duplicates": near},
               "image_width_median": int(np.median(w)) if w else None,
               "image_height_median": int(np.median(h)) if h else None, "color_modes": modes}
    return Prepared(kind="image", task="classification", target=None, text_column=None, features=[], ignored=[],
                    types={}, labels=labels, n_rows=len(items), train_idx=train_idx, test_idx=test_idx,
                    val_idx=val_idx, split_source=split_source, profile=profile, warnings=warnings,
                    split_strategy="provided" if split_source == "provided" else "random")


def load_manifest(dataset_dir: Path) -> list[dict]:
    return json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))["items"]


def save_split(dataset_dir: Path, train_idx: list[int], test_idx: list[int], val_idx: list[int] | None = None,
               **meta) -> str:
    """Persist the split membership (not just the seed) and return its SHA-256."""
    val_idx = list(val_idx or [])
    digest = split_sha256(train_idx, val_idx, test_idx)
    payload = {"train": train_idx, "val": val_idx, "test": test_idx, "sha256": digest, **meta}
    tmp = dataset_dir / "split.json.tmp"
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(dataset_dir / "split.json")
    return digest


def load_split(dataset_dir: Path) -> dict:
    """{"train", "val", "test", "sha256", ...}. v1.1 files have no "val" (→ empty list)."""
    data = json.loads((dataset_dir / "split.json").read_text(encoding="utf-8"))
    data.setdefault("val", [])
    data["sha256"] = split_sha256(data["train"], data["val"], data["test"])
    return data


def load_split_indices(dataset_dir: Path) -> tuple[list[int], list[int]]:
    data = load_split(dataset_dir)
    return data["train"], data["test"]
