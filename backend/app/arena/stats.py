"""Metrics, uncertainty, paired comparisons and the recommendation (protocol v2).

Classification predictions are integer class indices (``-1`` = an answer that
could not be mapped to a class, e.g. an unparseable LLM reply; it always counts
as wrong).

Evaluation protocol (see docs/BENCHMARK_METHODOLOGY.md §8)
---------------------------------------------------------
1. Models are fitted on the **training** split (tuning = cross-validation inside
   the training split).
2. The recommendation is chosen on the **validation** split only.
3. The **test** split is used once, afterwards, to estimate the performance of
   the model that was already chosen. Test scores of the other models are shown
   for information; picking the best of them would be optimistic (winner's bias).

Uncertainty
-----------
* Bootstrap percentile intervals over evaluation examples. The resampling unit
  is the example (``simple``), the example within its class (``stratified`` –
  class counts stay fixed, so macro-F1 is always defined) or the **group**
  (``cluster`` – used automatically when the dataset has a group column, because
  rows of the same group are not independent).
* Intervals describe uncertainty from the finite evaluation set only. They do
  not include variability from re-training or from a different split.

Comparisons
-----------
* Leader = best eligible model on validation. Every other eligible model is
  compared with the leader on the same examples (paired bootstrap of the
  difference Δ in the primary metric, oriented so that Δ > 0 means the leader is
  better).
* Difference test: two-sided bootstrap p-value for Δ = 0.
* Equivalence test (TOST): is |Δ| smaller than the practical-equivalence margin
  δ? Bootstrap p = max(P*(Δ ≥ δ), P*(Δ ≤ −δ)).
* Both families are Holm-adjusted over the m − 1 comparisons.
* Verdicts: ``equivalent`` (equivalence shown), ``worse`` (difference detected),
  ``inconclusive`` (neither – *not* a tie). Exact McNemar (classification) and
  Wilcoxon signed-rank on absolute errors (regression) are reported as
  supplementary tests.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

PRIMARY = {"classification": ("f1_macro", "Macro F1", True), "regression": ("r2", "R²", True)}
METRICS: dict[str, dict[str, tuple[str, bool]]] = {
    "classification": {"f1_macro": ("Macro F1", True), "balanced_accuracy": ("Balanced accuracy", True),
                       "accuracy": ("Accuracy", True)},
    "regression": {"r2": ("R²", True), "mae": ("MAE", False), "rmse": ("RMSE", False)},
}
DEFAULT_MARGIN = {"f1_macro": 0.02, "balanced_accuracy": 0.02, "accuracy": 0.02, "r2": 0.02}
RELATIVE_MARGIN = 0.05  # error metrics: 5 % of the leader's error
N_BOOT = 2000
ALPHA = 0.05


# ----------------------------------------------------------------------------- classification
def _confusion(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> np.ndarray:
    pred = np.where(y_pred < 0, k, y_pred)  # unparseable → extra column
    return np.bincount(y_true * (k + 1) + pred, minlength=k * (k + 1)).reshape(k, k + 1)


def _f1_from_cm(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = cm.shape[0]
    tp = np.diag(cm[:, :k]).astype(float)
    pred_count = cm[:, :k].sum(axis=0).astype(float)
    true_count = cm.sum(axis=1).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(pred_count > 0, tp / pred_count, 0.0)
        rec = np.where(true_count > 0, tp / true_count, 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    return prec, rec, f1


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str],
                           scores: np.ndarray | None = None) -> dict:
    """Point metrics. Macro averages run over the classes present in ``y_true``."""
    k = len(labels)
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    cm = _confusion(y_true, y_pred, k)
    prec, rec, f1 = _f1_from_cm(cm)
    support = cm.sum(axis=1)
    present = support > 0
    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    out = {
        "n": int(len(y_true)),
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(float(rec[present].mean()), 4) if present.any() else 0.0,
        "precision_macro": round(float(prec[present].mean()), 4) if present.any() else 0.0,
        "recall_macro": round(float(rec[present].mean()), 4) if present.any() else 0.0,
        "f1_macro": round(float(f1[present].mean()), 4) if present.any() else 0.0,
        "f1_weighted": round(float((f1 * support).sum() / max(1, support.sum())), 4),
        "unparseable": int((y_pred < 0).sum()),
        "classes_absent": [labels[i] for i in range(k) if not present[i]],
        "per_class": {labels[i]: {"precision": round(float(prec[i]), 4), "recall": round(float(rec[i]), 4),
                                  "f1": round(float(f1[i]), 4), "support": int(support[i])} for i in range(k)},
        "confusion_matrix": {"labels": labels, "matrix": cm[:, :k].tolist(),
                             "unparseable": cm[:, k].tolist() if cm[:, k].any() else None},
    }
    if scores is not None:
        out["roc_auc"] = _roc_auc(y_true, scores, k)
    return out


def _roc_auc(y_true: np.ndarray, scores: np.ndarray, k: int) -> float | None:
    from sklearn.metrics import roc_auc_score

    try:
        if k == 2:
            s = scores[:, 1] if scores.ndim == 2 and scores.shape[1] == 2 else scores.ravel()
            return round(float(roc_auc_score(y_true, s)), 4)
        if scores.ndim == 2 and scores.shape[1] == k and np.allclose(scores.sum(axis=1), 1.0, atol=1e-3):
            present = np.unique(y_true)
            if len(present) < k:
                return None
            return round(float(roc_auc_score(y_true, scores, multi_class="ovr", average="macro")), 4)
    except ValueError:
        return None
    return None


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Exact McNemar test on paired correct/incorrect outcomes (tests accuracy, not F1)."""
    from scipy.stats import binomtest

    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    p = 1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5).pvalue)
    return {"test": "McNemar (exact, accuracy)", "a_only_correct": b, "b_only_correct": c, "p_value": round(p, 4)}


# ----------------------------------------------------------------------------- regression
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(math.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else None
    # MAPE is undefined for zero targets and explodes near zero: only report it when every
    # |target| is at least 1 % of the mean |target|.
    scale = float(np.mean(np.abs(y_true))) if len(y_true) else 0.0
    mape = None
    if scale > 0 and bool(np.all(np.abs(y_true) >= 0.01 * scale)):
        mape = float(np.mean(np.abs(err / y_true)))
    return {"n": int(len(y_true)), "r2": round(r2, 4) if r2 is not None else None, "mae": round(mae, 6),
            "rmse": round(rmse, 6), "median_ae": round(float(np.median(np.abs(err))), 6) if len(err) else None,
            "mape": round(mape, 4) if mape is not None else None,
            **({"notes": ["R² is undefined: the target is constant on these examples."]} if r2 is None else {})}


def wilcoxon_abs_errors(err_a: np.ndarray, err_b: np.ndarray) -> dict:
    from scipy.stats import wilcoxon

    d = np.abs(err_a) - np.abs(err_b)
    name = "Wilcoxon signed-rank (|error|, zero differences dropped)"
    if np.allclose(d, 0):
        return {"test": name, "p_value": 1.0}
    try:
        p = float(wilcoxon(np.abs(err_a), np.abs(err_b), zero_method="wilcox").pvalue)
    except ValueError:
        p = 1.0
    return {"test": name, "p_value": round(p, 4)}


# ----------------------------------------------------------------------------- bootstrap
@dataclass
class Resampling:
    """How bootstrap resamples are drawn from the evaluation examples."""

    method: str = "simple"  # simple | stratified | cluster
    n_boot: int = N_BOOT
    seed: int = 42
    strata: np.ndarray | None = None  # class index per example (stratified)
    groups: np.ndarray | None = None  # group id per example (cluster)
    notes: list[str] = field(default_factory=list)

    def describe(self) -> dict:
        unit = {"simple": "example", "stratified": "example within its class (class counts fixed)",
                "cluster": "group (all rows of a group are resampled together)"}[self.method]
        return {"method": self.method, "resamples": self.n_boot, "unit": unit, "seed": self.seed}


def make_resampling(task: str, y: np.ndarray, groups: np.ndarray | None, n_boot: int, seed: int,
                    method: str = "auto") -> Resampling:
    if groups is not None and len(np.unique(groups)) >= 5 and method in ("auto", "cluster"):
        return Resampling("cluster", n_boot, seed, groups=np.asarray(groups))
    if task == "classification" and method in ("auto", "stratified"):
        return Resampling("stratified", n_boot, seed, strata=np.asarray(y))
    return Resampling("simple", n_boot, seed)


def resample_batches(n: int, rs: Resampling, positions: np.ndarray | None = None) -> Iterator[list[np.ndarray] | np.ndarray]:
    """Yield resample index arrays. Fixed-length methods yield 2-D batches, cluster yields lists."""
    rng = np.random.default_rng(rs.seed)
    sel = np.arange(n) if positions is None else positions
    if rs.method == "cluster" and rs.groups is not None:
        g = rs.groups[sel]
        uniq, inv = np.unique(g, return_inverse=True)
        members = [np.flatnonzero(inv == j) for j in range(len(uniq))]
        batch: list[np.ndarray] = []
        for _ in range(rs.n_boot):
            pick = rng.integers(0, len(uniq), size=len(uniq))
            batch.append(np.concatenate([members[j] for j in pick]))
            if len(batch) == 64:
                yield batch
                batch = []
        if batch:
            yield batch
        return
    chunk = max(1, min(500, 20_000_000 // max(1, len(sel))))
    if rs.method == "stratified" and rs.strata is not None:
        s = rs.strata[sel]
        groups = [np.flatnonzero(s == c) for c in np.unique(s)]
        done = 0
        while done < rs.n_boot:
            c = min(chunk, rs.n_boot - done)
            parts = [grp[rng.integers(0, len(grp), size=(c, len(grp)))] for grp in groups]
            yield np.concatenate(parts, axis=1)
            done += c
        return
    done = 0
    while done < rs.n_boot:
        c = min(chunk, rs.n_boot - done)
        yield rng.integers(0, len(sel), size=(c, len(sel)))
        done += c


def _cls_batch(yt: np.ndarray, yp: np.ndarray, k: int) -> dict[str, np.ndarray]:
    """Vectorised metrics for C resamples (rows of yt / yp)."""
    c, n = yt.shape
    pair = yt * (k + 1) + np.where(yp < 0, k, yp)
    offs = (np.arange(c) * (k * (k + 1)))[:, None]
    cm = np.bincount((pair + offs).ravel(), minlength=c * k * (k + 1)).reshape(c, k, k + 1)
    diag = np.arange(k)
    tp = cm[:, diag, diag].astype(float)
    pred_count = cm[:, :, :k].sum(axis=1).astype(float)
    true_count = cm.sum(axis=2).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(pred_count > 0, tp / pred_count, 0.0)
        rec = np.where(true_count > 0, tp / true_count, 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    present = true_count > 0
    n_present = np.maximum(present.sum(axis=1), 1)
    return {"accuracy": tp.sum(axis=1) / n, "f1_macro": (f1 * present).sum(axis=1) / n_present,
            "balanced_accuracy": (rec * present).sum(axis=1) / n_present}


def _reg_batch(yt: np.ndarray, yp: np.ndarray) -> dict[str, np.ndarray]:
    err = yp - yt
    sse = (err ** 2).sum(axis=1)
    sst = ((yt - yt.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.where(sst > 0, 1 - sse / sst, np.nan)
    return {"mae": np.abs(err).mean(axis=1), "rmse": np.sqrt((err ** 2).mean(axis=1)), "r2": r2}


def bootstrap_distributions(task: str, y: np.ndarray, preds: list[np.ndarray], k: int, rs: Resampling,
                            metrics: list[str]) -> list[dict[str, np.ndarray]]:
    """Metric values of every prediction vector on the *same* resamples (paired)."""
    n = len(y)
    out = [{m: [] for m in metrics} for _ in preds]

    def add(idx2d: np.ndarray) -> None:
        yt = y[idx2d]
        for j, p in enumerate(preds):
            vals = _cls_batch(yt, p[idx2d], k) if task == "classification" else _reg_batch(yt, p[idx2d])
            for m in metrics:
                out[j][m].append(vals[m])

    for batch in resample_batches(n, rs):
        if isinstance(batch, list):
            for idx in batch:
                add(idx[None, :])
        else:
            add(batch)
    return [{m: np.concatenate(v[m]) for m in metrics} for v in out]


def percentile_ci(values: np.ndarray, level: float = 0.95) -> list[float] | None:
    v = values[~np.isnan(values)]
    if len(v) < 20:
        return None
    lo, hi = np.percentile(v, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return [round(float(lo), 4), round(float(hi), 4)]


def bootstrap_cis(task: str, y: np.ndarray, pred: np.ndarray, k: int, rs: Resampling) -> dict:
    """95 % intervals for the reported metrics of one model."""
    if len(y) < 5:
        return {}
    metrics = list(METRICS[task])
    dist = bootstrap_distributions(task, y, [pred], k, rs, metrics)[0]
    out = {f"{m}_ci": percentile_ci(dist[m]) for m in metrics}
    if task == "regression":
        undefined = int(np.isnan(dist["r2"]).sum())
        if undefined:
            out["r2_ci_undefined_resamples"] = undefined
    return out


# kept for callers that used the v1.1 names
def bootstrap_classification(y_true: np.ndarray, y_pred: np.ndarray, k: int, seed: int = 42,
                             n_boot: int = N_BOOT) -> dict:
    return bootstrap_cis("classification", np.asarray(y_true), np.asarray(y_pred), k,
                         Resampling("stratified", n_boot, seed, strata=np.asarray(y_true)))


def bootstrap_regression(y_true: np.ndarray, y_pred: np.ndarray, seed: int = 42, n_boot: int = N_BOOT) -> dict:
    return bootstrap_cis("regression", np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float), 0,
                         Resampling("simple", n_boot, seed))


# ----------------------------------------------------------------------------- multiple comparisons
def holm(pvalues: list[float | None]) -> list[float | None]:
    """Holm step-down adjusted p-values (``None`` entries are left out of the family)."""
    idx = [i for i, p in enumerate(pvalues) if p is not None]
    m = len(idx)
    order = sorted(idx, key=lambda i: pvalues[i])  # type: ignore[arg-type,return-value]
    adjusted: list[float | None] = [None] * len(pvalues)
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * float(pvalues[i]))  # type: ignore[arg-type]
        running = max(running, val)
        adjusted[i] = round(running, 6)
    return adjusted


def bootstrap_p_values(delta: np.ndarray, margin: float) -> tuple[float, float]:
    """(two-sided p for Δ = 0, TOST p for |Δ| ≥ margin) from a paired bootstrap distribution."""
    d = delta[~np.isnan(delta)]
    b = len(d)
    if b == 0:
        return 1.0, 1.0
    p_le0 = (1 + np.sum(d <= 0)) / (b + 1)
    p_ge0 = (1 + np.sum(d >= 0)) / (b + 1)
    p_diff = min(1.0, 2 * min(p_le0, p_ge0))
    p_upper = (1 + np.sum(d >= margin)) / (b + 1)
    p_lower = (1 + np.sum(d <= -margin)) / (b + 1)
    return float(p_diff), float(max(p_upper, p_lower))


# ----------------------------------------------------------------------------- recommendation
@dataclass
class Policy:
    primary_metric: str
    margin: float | None = None  # None → default (absolute 0.02, or 5 % of the leader's error)
    objective: str = "efficiency"  # efficiency | quality
    alpha: float = ALPHA
    min_quality: float | None = None
    max_latency_ms: float | None = None
    max_size_mb: float | None = None
    max_memory_mb: float | None = None

    def constraints(self) -> dict:
        return {k: v for k, v in {"min_quality": self.min_quality, "max_latency_ms": self.max_latency_ms,
                                  "max_size_mb": self.max_size_mb, "max_memory_mb": self.max_memory_mb}.items()
                if v is not None}


def _violations(c: dict, metric: str, higher: bool, policy: Policy) -> list[str]:
    out = []
    q = c["metrics"].get(metric)
    if policy.min_quality is not None and q is not None:
        if (q < policy.min_quality) if higher else (q > policy.min_quality):
            out.append(f"{'below' if higher else 'above'} the required {metric} {policy.min_quality}")
    lat = c.get("latency_ms")
    if policy.max_latency_ms is not None:
        if lat is None:
            out.append("latency unknown")
        elif lat > policy.max_latency_ms:
            out.append(f"latency {lat:.1f} ms > {policy.max_latency_ms} ms")
    size = c.get("size_bytes")
    if policy.max_size_mb is not None and size is not None and size / 1024 / 1024 > policy.max_size_mb:
        out.append(f"size {size / 1024 / 1024:.1f} MB > {policy.max_size_mb} MB")
    mem = c.get("memory_mb")
    if policy.max_memory_mb is not None and mem is not None and mem > policy.max_memory_mb:
        out.append(f"memory {mem:.0f} MB > {policy.max_memory_mb} MB")
    return out


def _oriented(v: float | None, higher: bool) -> float:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return -math.inf
    return v if higher else -v


def compare(task: str, candidates: list[dict], y: np.ndarray, k: int, policy: Policy, rs: Resampling,
            split: str = "validation") -> dict:
    """Choose a model on one evaluation split.

    Each candidate: {"key", "name", "source", "pred" (aligned with ``idx``), "idx" (positions in ``y``),
    "metrics" (point metrics on those positions), "latency_ms", "size_bytes", "memory_mb", "caveats": [...]}.
    """
    metric = policy.primary_metric
    label, higher = METRICS[task][metric]
    usable = [c for c in candidates if c.get("metrics") and c["metrics"].get(metric) is not None]
    result: dict = {"split": split, "primary_metric": metric, "primary_label": label, "higher_is_better": higher,
                    "alpha": policy.alpha, "objective": policy.objective, "constraints": policy.constraints(),
                    "resampling": rs.describe(), "comparisons": {}, "excluded": {}, "ranking": [],
                    "recommendation": None, "status": "none", "shortlist": [], "warnings": []}
    if not usable:
        result["warnings"].append("No model produced usable predictions.")
        return result
    usable.sort(key=lambda c: (-_oriented(c["metrics"][metric], higher), c.get("latency_ms") or math.inf))
    result["ranking"] = [c["key"] for c in usable]
    baseline = next((c for c in usable if c["source"] == "baseline"), None)
    eligible = []
    for c in usable:
        if c["source"] == "baseline":
            continue
        v = _violations(c, metric, higher, policy)
        if v:
            result["excluded"][c["key"]] = v
        else:
            eligible.append(c)
    if not eligible:
        result["warnings"].append("No model meets the constraints." if result["excluded"] else
                                  "Only the baseline produced results.")
        return result
    leader = eligible[0]
    margin = policy.margin
    margin_rule = "absolute (user)"
    if margin is None:
        if metric in DEFAULT_MARGIN:
            margin, margin_rule = DEFAULT_MARGIN[metric], "absolute (default)"
        else:
            margin = RELATIVE_MARGIN * abs(float(leader["metrics"][metric]) or 1.0)
            margin_rule = f"{RELATIVE_MARGIN:.0%} of the leader's {label}"
    result.update({"leader": leader["key"], "margin": round(float(margin), 6), "margin_rule": margin_rule})

    others = [c for c in eligible[1:]] + ([baseline] if baseline is not None else [])
    raw_diff: list[float | None] = []
    raw_tost: list[float | None] = []
    comps: list[dict] = []
    for c in others:
        comp = _paired(task, y, k, leader, c, metric, higher, float(margin), rs)
        comps.append(comp)
        is_family = c["source"] != "baseline"
        raw_diff.append(comp.get("p_diff") if is_family else None)
        raw_tost.append(comp.get("p_equivalence") if is_family else None)
    adj_diff, adj_tost = holm(raw_diff), holm(raw_tost)
    for c, comp, pd_, pt_ in zip(others, comps, adj_diff, adj_tost, strict=True):
        if c["source"] == "baseline":  # separate question: does the leader beat the baseline?
            comp["p_diff_holm"], comp["p_equivalence_holm"] = comp.get("p_diff"), comp.get("p_equivalence")
        else:
            comp["p_diff_holm"], comp["p_equivalence_holm"] = pd_, pt_
        comp["verdict"] = _verdict(comp, policy.alpha)
        result["comparisons"][c["key"]] = comp
    result["family_size"] = sum(1 for c in others if c["source"] != "baseline")
    result["correction"] = "Holm" if result["family_size"] > 1 else "none (single comparison)"

    equivalent = [c for c in eligible[1:] if result["comparisons"][c["key"]]["verdict"] == "equivalent"]
    inconclusive = [c for c in eligible[1:] if result["comparisons"][c["key"]]["verdict"] == "inconclusive"]
    pool = [leader] + equivalent
    if policy.objective == "quality":
        pick = leader
    else:
        pick = min(pool, key=lambda c: (c.get("latency_ms") if c.get("latency_ms") is not None else math.inf,
                                        c.get("size_bytes") or math.inf))
    status = "provisional" if inconclusive else "clear"
    lv = leader["metrics"][metric]
    if pick is leader:
        reason = f"{leader['name']} has the best validation {label} ({lv:.3f})"
        if equivalent:
            reason += f"; {len(equivalent)} other model(s) are practically equivalent (within ±{margin:.3g}) but not faster"
        reason += "."
    else:
        comp = result["comparisons"][pick["key"]]
        reason = (f"{pick['name']} is practically equivalent to the validation leader {leader['name']} "
                  f"(Δ{label} = {comp['delta']:+.3f}, 90% CI {comp['ci90'][0]:+.3f} to {comp['ci90'][1]:+.3f}, "
                  f"inside the margin ±{margin:.3g}) and it is faster "
                  f"({_fmt_ms(pick.get('latency_ms'))} vs {_fmt_ms(leader.get('latency_ms'))} per example).")
    if inconclusive:
        reason += (f" {len(inconclusive)} model(s) could be as good: the validation data can neither separate them "
                   "from the leader nor show they are equivalent.")
    result["shortlist"] = [leader["key"]] + [c["key"] for c in inconclusive]
    result["status"] = status
    result["recommendation"] = {"key": pick["key"], "name": pick["name"], "reason": reason, "status": status,
                                "equivalent_to_leader": [c["key"] for c in equivalent],
                                "caveats": list(pick.get("caveats") or [])}
    if baseline is not None:
        bcomp = result["comparisons"].get(baseline["key"])
        if bcomp and bcomp.get("ci95") and bcomp["ci95"][0] <= 0:
            result["warnings"].append("The best model is not detectably better than the trivial baseline on the "
                                      "validation split – the features may not predict the target.")
        weak = [c["name"] for c in eligible if _oriented(c["metrics"][metric], higher)
                <= _oriented(baseline["metrics"][metric], higher)]
        if weak:
            result["warnings"].append("Not better than the baseline: " + ", ".join(weak[:6]) + ".")
    result["pareto"] = _pareto([c for c in usable if c["source"] != "baseline"], metric, higher)
    return result


def _paired(task: str, y: np.ndarray, k: int, a: dict, b: dict, metric: str, higher: bool, margin: float,
            rs: Resampling) -> dict:
    common, ia, ib = np.intersect1d(a["idx"], b["idx"], return_indices=True)
    out: dict = {"n": int(len(common)), "a": a["key"]}
    if len(common) < 10:
        out.update({"note": "too few shared examples for a comparison", "p_diff": None, "p_equivalence": None})
        return out
    yt = y[common]
    pa, pb = np.asarray(a["pred"])[ia], np.asarray(b["pred"])[ib]
    sub = Resampling(rs.method, rs.n_boot, rs.seed,
                     strata=rs.strata[common] if rs.strata is not None else None,
                     groups=rs.groups[common] if rs.groups is not None else None)
    da, db = bootstrap_distributions(task, yt, [pa, pb], k, sub, [metric])
    sign = 1.0 if higher else -1.0
    delta = sign * (da[metric] - db[metric])  # > 0: the leader is better
    if task == "classification":
        point = sign * (_cls_batch(yt[None, :], pa[None, :], k)[metric][0] - _cls_batch(yt[None, :], pb[None, :], k)[metric][0])
        supp = mcnemar(pa == yt, pb == yt)
    else:
        point = sign * (_reg_batch(yt[None, :], pa[None, :])[metric][0] - _reg_batch(yt[None, :], pb[None, :])[metric][0])
        supp = wilcoxon_abs_errors(pa - yt, pb - yt)
    p_diff, p_tost = bootstrap_p_values(delta, margin)
    ci95, ci90 = percentile_ci(delta, 0.95), percentile_ci(delta, 0.90)
    out.update({"delta": round(float(point), 4), "ci95": ci95, "ci90": ci90, "p_diff": round(p_diff, 6),
                "p_equivalence": round(p_tost, 6), "margin": round(margin, 6), "supplementary": supp,
                "undefined_resamples": int(np.isnan(delta).sum())})
    return out


def _verdict(comp: dict, alpha: float) -> str:
    if comp.get("p_diff_holm") is None:
        return "not_tested"
    if comp.get("p_equivalence_holm") is not None and comp["p_equivalence_holm"] < alpha:
        return "equivalent"
    if comp["p_diff_holm"] < alpha:
        return "worse" if comp["delta"] > 0 else "better"
    return "inconclusive"


VERDICT_TEXT = {
    "equivalent": "practically equivalent to the leader (difference within the margin)",
    "worse": "worse than the leader (difference detected)",
    "better": "better than the leader on paired resamples (point estimates disagree – check the CI)",
    "inconclusive": "difference not detected, equivalence not shown – more data needed",
    "not_tested": "not compared (too few shared examples)",
}


def _pareto(rows: list[dict], metric: str, higher: bool) -> list[str]:
    pts = [(r["key"], r["metrics"][metric], r.get("latency_ms")) for r in rows
           if r.get("latency_ms") is not None and r["metrics"].get(metric) is not None]
    front = []
    for key, q, lat in pts:
        dominated = any((q2 >= q if higher else q2 <= q) and lat2 <= lat and (q2 != q or lat2 != lat)
                        for k2, q2, lat2 in pts if k2 != key)
        if not dominated:
            front.append(key)
    return front


def _fmt_ms(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v:.2f} ms" if v < 10 else f"{v:.0f} ms"
