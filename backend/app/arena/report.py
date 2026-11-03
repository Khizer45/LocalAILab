"""Serializers and CSV / Markdown reports for Model Arena runs."""

from __future__ import annotations

import csv
import io
from datetime import datetime

from app.models.arena import ArenaDataset, ArenaModel, ArenaResult, ArenaRun


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() + "Z" if dt else None


def _flip(ci: list | None) -> list | None:
    """Interval of −Δ (model − leader) from the interval of Δ (leader − model)."""
    return [-ci[1], -ci[0]] if ci and None not in ci else None


def dataset_out(d: ArenaDataset) -> dict:
    return {"id": d.id, "name": d.name, "kind": d.kind, "task": d.task, "target": d.target,
            "text_column": d.text_column, "columns": d.columns, "labels": d.labels, "n_rows": d.n_rows,
            "n_train": d.n_train, "n_val": d.n_val, "n_test": d.n_test, "test_size": d.test_size,
            "val_size": d.val_size, "seed": d.seed, "split_source": d.split_source, "group_column": d.group_column,
            "time_column": d.time_column, "split_sha256": d.split_sha256,
            "filename": d.filename, "size_bytes": d.size_bytes, "sha256": d.sha256,
            "profile": d.profile, "warnings": d.warnings, "created_at": _iso(d.created_at)}


def model_out(m: ArenaModel) -> dict:
    return {"id": m.id, "name": m.name, "filename": m.filename, "format": m.format, "kind": m.kind, "task": m.task,
            "role": m.role, "size_bytes": m.size_bytes, "sha256": m.sha256, "config": m.config,
            "inspection": m.inspection, "trusted": m.trusted, "provenance": m.provenance,
            "created_at": _iso(m.created_at)}


def run_out(r: ArenaRun) -> dict:
    return {"id": r.id, "name": r.name, "dataset_id": r.dataset_id, "dataset": r.dataset_info, "config": r.config,
            "contenders": r.contenders, "status": r.status, "protocol": r.protocol, "progress_done": r.progress_done,
            "progress_total": r.progress_total, "current_item": r.current_item, "error": r.error,
            "environment": r.environment, "summary": r.summary, "has_model_file": bool(r.best_model_path),
            "mlflow_run_id": r.mlflow_run_id, "created_at": _iso(r.created_at), "started_at": _iso(r.started_at),
            "finished_at": _iso(r.finished_at),
            "duration_s": round((r.finished_at - r.started_at).total_seconds(), 1)
            if r.finished_at and r.started_at else None}


def result_out(x: ArenaResult, *, with_predictions: bool = False) -> dict:
    d = {"id": x.id, "key": x.key, "source": x.source, "name": x.name, "status": x.status, "metrics": x.metrics,
         "val_metrics": x.val_metrics, "details": x.details, "error": x.error}
    if with_predictions:
        d["predictions"], d["evaluated_idx"] = x.predictions, x.evaluated_idx
        d["val_predictions"], d["val_idx"] = x.val_predictions, x.val_idx
    return d


def _ordered(run: ArenaRun, results: list[ArenaResult]) -> list[ArenaResult]:
    ranking = (run.summary or {}).get("ranking") or []
    pos = {k: i for i, k in enumerate(ranking)}
    return sorted(results, key=lambda r: (pos.get(r.key, 10_000), r.id))


CLS_COLS = ["accuracy", "balanced_accuracy", "f1_macro", "precision_macro", "recall_macro", "roc_auc"]
REG_COLS = ["r2", "mae", "rmse", "median_ae", "mape"]
PERF_COLS = ["fit_ms", "latency_ms_per_example", "latency_ms_single_p50", "latency_ms_single_p95", "throughput_per_s",
             "model_size_bytes", "peak_process_rss_mb"]


def _legacy_csv(run: ArenaRun, results: list[ArenaResult]) -> str:
    task = run.dataset_info.get("task")
    cols = CLS_COLS if task == "classification" else REG_COLS
    tests = (run.summary or {}).get("tests", {})
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["rank", "key", "name", "source", "status", *cols, "cv_mean", "cv_std", *PERF_COLS, "p_value_vs_best",
                "error"])
    for i, r in enumerate(_ordered(run, results), 1):
        m = r.metrics or {}
        cv = (r.details or {}).get("cv") or {}
        w.writerow([i if r.status == "ok" else "", r.key, r.name, r.source, r.status, *[m.get(c) for c in cols],
                    cv.get("mean"), cv.get("std"), *[m.get(c) for c in PERF_COLS],
                    tests.get(r.key, {}).get("p_value"), r.error or ""])
    return buf.getvalue()


def _f(v, digits: int = 3) -> str:
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _legacy_markdown(run: ArenaRun, results: list[ArenaResult]) -> str:
    ds = run.dataset_info
    s = run.summary or {}
    task = ds.get("task")
    lines = [f"# Model Arena report – {run.name}", "",
             "> **Legacy protocol (v1.1):** the recommendation was chosen on the same test split it is reported on, "
             "and p ≥ 0.05 was shown as a tie. Treat this report as exploratory; re-run the comparison in v1.2 for a "
             "validation-based choice with an unbiased test estimate.", "",
             f"- **Run ID:** {run.id} · **Status:** {run.status}",
             f"- **Dataset:** {ds.get('name')} ({ds.get('kind')} {task}, {ds.get('n_rows')} "
             f"{'images' if ds.get('kind') == 'image' else 'rows'}; "
             f"train {ds.get('n_train')} / test {ds.get('n_test')}, split {ds.get('split_source')}, seed {ds.get('seed')}; "
             f"sha256 `{(ds.get('sha256') or '')[:12]}…`)",
             f"- **Settings:** cross-validation {run.config.get('cv_folds')} folds, tuning "
             f"{'on' if run.config.get('tune') else 'off'}",
             f"- **Machine:** {(run.environment or {}).get('os', '')} · {(run.environment or {}).get('cpu_count_logical', '?')} CPUs · "
             f"{(run.environment or {}).get('ram_total_mb', '?')} MB RAM", ""]
    rec = s.get("recommendation")
    if rec:
        lines += [f"**Recommendation:** {rec['name']} – {rec['reason']}", ""]
    for w in s.get("warnings") or []:
        lines.append(f"> ⚠ {w}")
    if s.get("warnings"):
        lines.append("")
    tests = s.get("tests", {})
    if task == "classification":
        head = "| # | Model | Macro F1 (95% CI) | Accuracy | Balanced acc. | CV F1 (train) | Latency / example | Size | p vs best |"
        lines += [head, "|---|---|---|---|---|---|---|---|---|"]
    else:
        head = "| # | Model | R² (95% CI) | MAE | RMSE | CV R² (train) | Latency / example | Size | p vs best |"
        lines += [head, "|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(_ordered(run, results), 1):
        if r.status != "ok":
            lines.append(f"| – | {r.name} | error: {(r.error or '')[:80]} | | | | | | |")
            continue
        m = r.metrics or {}
        cv = (r.details or {}).get("cv") or {}
        cvs = f"{_f(cv.get('mean'))} ± {_f(cv.get('std'))}" if cv.get("mean") is not None else "–"
        lat = m.get("latency_ms_single_p50") or m.get("latency_ms_per_example")
        size = m.get("model_size_bytes")
        size_s = f"{size / 1024 / 1024:.2f} MB" if size else "–"
        p = tests.get(r.key, {}).get("p_value")
        if task == "classification":
            ci = m.get("f1_macro_ci") or [None, None]
            lines.append(f"| {i} | {r.name} | {_f(m.get('f1_macro'))} ({_f(ci[0])}–{_f(ci[1])}) | {_f(m.get('accuracy'))} | "
                         f"{_f(m.get('balanced_accuracy'))} | {cvs} | {_f(lat, 2)} ms | {size_s} | {_f(p, 4)} |")
        else:
            ci = m.get("r2_ci") or [None, None]
            lines.append(f"| {i} | {r.name} | {_f(m.get('r2'))} ({_f(ci[0])}–{_f(ci[1])}) | {_f(m.get('mae'))} | "
                         f"{_f(m.get('rmse'))} | {cvs} | {_f(lat, 2)} ms | {size_s} | {_f(p, 4)} |")
    lines += ["", "## How to read this", "",
              "- All models were evaluated on the **same held-out test split**; the test split was never used for "
              "training, model selection or tuning.",
              "- 95% confidence intervals come from 1,000 bootstrap resamples of the test set. Overlapping intervals "
              "mean the difference may not be real.",
              f"- {s.get('note', '')}",
              "- Latency is the median time for a single prediction on this computer (built-in image models include "
              "the feature extractor). LLM latency includes generation through Ollama.",
              "- CV = cross-validation on the training split only (mean ± std across folds)."]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- protocol v2
def to_csv(run: ArenaRun, results: list[ArenaResult]) -> str:
    s = run.summary or {}
    if s.get("protocol") != "v2":
        return _legacy_csv(run, results)
    task = run.dataset_info.get("task")
    metric = s.get("primary_metric")
    cols = CLS_COLS if task == "classification" else REG_COLS
    sel = s.get("selection") or {}
    comps = sel.get("comparisons") or {}
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["validation_rank", "key", "name", "source", "status", f"val_{metric}", "val_ci_low", "val_ci_high",
                "verdict_vs_leader", "delta_model_minus_leader", "delta_ci95_low", "delta_ci95_high", "p_difference_holm",
                "p_equivalence_holm", "excluded_by_constraints", *[f"test_{c}" for c in cols],
                f"test_{metric}_ci_low", f"test_{metric}_ci_high", "cv_mean", "cv_std", *PERF_COLS, "caveats",
                "error"])
    for r in _ordered(run, results):
        vm = r.val_metrics or {}
        tm = r.metrics or {}
        c = comps.get(r.key) or {}
        vci = vm.get(f"{metric}_ci") or [None, None]
        tci = tm.get(f"{metric}_ci") or [None, None]
        dci = _flip(c.get("ci95")) or [None, None]
        cv = (r.details or {}).get("cv") or {}
        rank = (s.get("ranking") or []).index(r.key) + 1 if r.key in (s.get("ranking") or []) else ""
        verdict = "leader" if r.key == sel.get("leader") else c.get("verdict", "")
        w.writerow([rank, r.key, r.name, r.source, r.status, vm.get(metric), vci[0], vci[1], verdict,
                    -c["delta"] if c.get("delta") is not None else None, dci[0], dci[1], c.get("p_diff_holm"), c.get("p_equivalence_holm"),
                    "; ".join((sel.get("excluded") or {}).get(r.key, [])), *[tm.get(x) for x in cols], tci[0], tci[1],
                    cv.get("mean"), cv.get("std"), *[tm.get(x) for x in PERF_COLS],
                    " | ".join((r.details or {}).get("caveats") or []), r.error or ""])
    return buf.getvalue()


def _ci(ci) -> str:
    if not ci or ci[0] is None:
        return ""
    return f" ({_f(ci[0])}–{_f(ci[1])})"


def _p(v) -> str:
    if v is None:
        return "–"
    return "<0.001" if v < 0.001 else f"{v:.3f}"


def to_markdown(run: ArenaRun, results: list[ArenaResult]) -> str:
    s = run.summary or {}
    if s.get("protocol") != "v2":
        return _legacy_markdown(run, results)
    ds = run.dataset_info
    cfg = run.config or {}
    sel = s.get("selection") or {}
    comps = sel.get("comparisons") or {}
    metric, label = s.get("primary_metric"), s.get("primary_label")
    sp = s.get("split") or {}
    env = run.environment or {}
    unit = "images" if ds.get("kind") == "image" else "rows"
    lines = [f"# Model Arena report – {run.name}", "",
             f"- **Run ID:** {run.id} · **Status:** {run.status} · **Protocol:** v2 (choose on validation, report on test)",
             f"- **Dataset:** {ds.get('name')} ({ds.get('kind')} {ds.get('task')}, {ds.get('n_rows')} {unit}; "
             f"sha256 `{(ds.get('sha256') or '')[:12]}…`)",
             f"- **Split:** {ds.get('split_source')} · train {sp.get('n_train')} / validation {sp.get('n_val')} / "
             f"test {sp.get('n_test')} · seed {ds.get('seed')} · split sha256 `{(sp.get('sha256') or '')[:12]}…`"
             + (f" · groups: `{ds.get('group_column')}`" if ds.get("group_column") else "")
             + (f" · time: `{ds.get('time_column')}`" if ds.get("time_column") else ""),
             f"- **Policy:** primary metric {label} · objective {sel.get('objective')} · equivalence margin "
             f"±{_f(sel.get('margin'), 4)} ({sel.get('margin_rule', '–')}) · α = {sel.get('alpha')} · "
             f"correction {sel.get('correction', '–')} · constraints "
             f"{', '.join(f'{k}={v}' for k, v in (sel.get('constraints') or {}).items()) or 'none'}",
             f"- **Uncertainty:** {((sel.get('resampling') or {}).get('resamples'))} bootstrap resamples, unit: "
             f"{(sel.get('resampling') or {}).get('unit')}",
             f"- **Training settings:** cross-validation {cfg.get('cv_folds')} folds (training split only), tuning "
             f"{'on' if cfg.get('tune') else 'off'}",
             f"- **Machine:** {env.get('os', '')} · {env.get('cpu_count_logical', '?')} CPUs · "
             f"{env.get('ram_total_mb', '?')} MB RAM · Local AI Lab {env.get('app_version', '')}", ""]
    rec, final = s.get("recommendation"), s.get("final")
    if rec:
        lines += [f"## Recommendation: {rec['name']}", "",
                  f"**Status: {rec.get('status', '')}.** {rec['reason']}", ""]
        if final:
            lines += [f"**Final estimate on the test split:** {label} {_f(final.get('value'))}{_ci(final.get('ci'))}, "
                      f"n = {final.get('n')}. {final.get('note', '')}", ""]
        for c in rec.get("caveats") or []:
            lines.append(f"> ⚠ {c}")
        if rec.get("caveats"):
            lines.append("")
        if sel.get("shortlist") and len(sel["shortlist"]) > 1:
            names = {r.key: r.name for r in results}
            lines += ["**Shortlist (cannot be separated with this data):** "
                      + ", ".join(names.get(k, k) for k in sel["shortlist"]), ""]
    else:
        lines += ["## No recommendation", "", "No model met the constraints or produced usable results.", ""]
    for w_ in s.get("warnings") or []:
        lines.append(f"> ⚠ {w_}")
    if s.get("warnings"):
        lines.append("")

    lines += [f"## Selection on the validation split ({sp.get('n_val')} {unit})", "",
              f"| # | Model | {label} (95% CI) | Δ model − leader (95% CI) | Verdict | p diff (Holm) | p equiv (Holm) | "
              "Latency / example | Size |",
              "|---|---|---|---|---|---|---|---|---|"]
    ranking = s.get("ranking") or []
    for r in _ordered(run, results):
        if r.status != "ok":
            lines.append(f"| – | {r.name} | error: {(r.error or '')[:80]} | | | | | | |")
            continue
        vm = r.val_metrics or {}
        c = comps.get(r.key) or {}
        tm = r.metrics or {}
        lat = tm.get("latency_ms_single_p50") or tm.get("latency_ms_per_example")
        size = tm.get("model_size_bytes")
        excluded = (sel.get("excluded") or {}).get(r.key)
        verdict = ("**leader**" if r.key == sel.get("leader") else
                   ("excluded: " + "; ".join(excluded)) if excluded else c.get("verdict", "–"))
        delta = f"{-c['delta']:+.3f}{_ci(_flip(c.get('ci95')))}" if c.get("delta") is not None else "–"
        lines.append(f"| {ranking.index(r.key) + 1 if r.key in ranking else '–'} | {r.name} | "
                     f"{_f(vm.get(metric))}{_ci(vm.get(f'{metric}_ci'))} | {delta} | {verdict} | "
                     f"{_p(c.get('p_diff_holm'))} | {_p(c.get('p_equivalence_holm'))} | "
                     f"{_f(lat, 2) + ' ms' if lat else '–'} | {f'{size / 1024 / 1024:.2f} MB' if size else '–'} |")
    caveated = [(r.name, (r.details or {}).get("caveats")) for r in _ordered(run, results)
                if (r.details or {}).get("caveats")]
    if caveated:
        lines += ["", "**Caveats about uploaded models' training data:**", ""]
        lines += [f"- {name}: {' '.join(cav)}" for name, cav in caveated]
    lines += ["", f"## Test split ({sp.get('n_test')} {unit}) – for information", "",
              "Only the recommended model's test score is an unbiased estimate. The other rows are shown so you can "
              "see how models behaved on unseen data; do not pick a model from this table.", ""]
    if ds.get("task") == "classification":
        lines += ["| Model | Macro F1 (95% CI) | Accuracy | Balanced acc. | CV F1 (train) |", "|---|---|---|---|---|"]
    else:
        lines += ["| Model | R² (95% CI) | MAE | RMSE | CV R² (train) |", "|---|---|---|---|---|"]
    for r in _ordered(run, results):
        if r.status != "ok":
            continue
        m = r.metrics or {}
        cv = (r.details or {}).get("cv") or {}
        cvs = f"{_f(cv.get('mean'))} ± {_f(cv.get('std'))}" if cv.get("mean") is not None else "–"
        mark = " ★" if rec and r.key == rec["key"] else ""
        if ds.get("task") == "classification":
            lines.append(f"| {r.name}{mark} | {_f(m.get('f1_macro'))}{_ci(m.get('f1_macro_ci'))} | "
                         f"{_f(m.get('accuracy'))} | {_f(m.get('balanced_accuracy'))} | {cvs} |")
        else:
            lines.append(f"| {r.name}{mark} | {_f(m.get('r2'))}{_ci(m.get('r2_ci'))} | {_f(m.get('mae'))} | "
                         f"{_f(m.get('rmse'))} | {cvs} |")
    lines += ["", "## How to read this", "",
              "- **Verdicts** compare each model with the validation leader on the same examples (paired bootstrap): "
              "*equivalent* = the difference is shown to be smaller than the margin; *worse* = a difference was "
              "detected; *inconclusive* = neither – this is **not** a tie, it means the data cannot tell.",
              "- **Δ** is the model's score minus the leader's (negative = worse; for error metrics the sign is turned so "
              "that negative is still worse).",
              "- p-values are Holm-adjusted over all comparisons with the leader; the intervals are not, so an interval "
              "that excludes 0 can still come with an *inconclusive* verdict. Supplementary exact McNemar / "
              "Wilcoxon tests are in the JSON export.",
              "- Latency is the median time for a single prediction on this computer (built-in image models include "
              "the feature extractor; LLM latency includes generation).",
              "", "## Limitations", ""] + [f"- {x}" for x in s.get("limitations") or []]
    return "\n".join(lines) + "\n"
