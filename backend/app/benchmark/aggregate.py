"""Aggregation of benchmark results into per-model summaries and reports."""

from __future__ import annotations

import csv
import io
from collections import defaultdict

import numpy as np

from app.models.benchmark import BenchmarkResult, BenchmarkRun


def _pct(values: list[float], q: float) -> float | None:
    return round(float(np.percentile(values, q)), 1) if values else None


def _mean(values: list[float]) -> float | None:
    return round(float(np.mean(values)), 3) if values else None


BOOT = 2000
EQUIV_MARGIN = 0.05  # 5 quality points


def _wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def _case_means(rows: list[BenchmarkResult], field: str = "score") -> dict[str, float]:
    per_case: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        per_case[r.case_id].append(float(r.score) if field == "score" else (1.0 if r.passed else 0.0))
    return {k: float(np.mean(v)) for k, v in per_case.items()}


def _boot_mean_ci(values: np.ndarray, seed: int = 0) -> list[float] | None:
    """95 % percentile CI of a mean, resampling *cases* (repeats of a case are not independent)."""
    if len(values) < 5:
        return None
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(BOOT, len(values)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def summarize(results: list[BenchmarkResult]) -> list[dict]:
    """Per-model summary. Three quality views are reported separately:

    * ``effective_quality`` (= ``quality_score``) – mean score over all cases, failed/timed-out calls count as 0;
    * ``conditional_quality`` – mean score over the calls that completed;
    * ``completion_rate`` – share of calls that completed (no error/timeout).
    """
    by_model: dict[str, list[BenchmarkResult]] = defaultdict(list)
    for r in results:
        by_model[r.model].append(r)
    out = []
    for model, rows in by_model.items():
        ok = [r for r in rows if r.status == "ok"]
        cats: dict[str, list[BenchmarkResult]] = defaultdict(list)
        for r in rows:
            cats[r.category].append(r)
        per_case: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            per_case[r.case_id].append(r.score)
        stds = [float(np.std(v)) for v in per_case.values() if len(v) > 1]
        sources = {r.tokens_source for r in ok if r.tokens_per_s is not None}
        tps = [r.tokens_per_s for r in ok if r.tokens_per_s is not None]
        case_scores = np.array(list(_case_means(rows).values()))
        case_pass = np.array(list(_case_means(rows, "passed").values()))
        lat = [r.total_ms for r in ok if r.total_ms is not None]
        ttft = [r.ttft_ms for r in ok if r.ttft_ms is not None]
        ttfa = [getattr(r, "ttft_answer_ms", None) for r in ok]
        ttfa = [v for v in ttfa if v is not None]
        effective = _mean([r.score for r in rows])
        out.append({
            "model": model,
            "results": len(rows),
            "cases": len(per_case),
            "ok": len(ok),
            "errors": sum(r.status == "error" for r in rows),
            "timeouts": sum(r.status == "timeout" for r in rows),
            "quality_score": effective,  # kept for v1.1 clients; same as effective_quality
            "effective_quality": effective,
            "effective_quality_ci": _boot_mean_ci(case_scores),
            "conditional_quality": _mean([r.score for r in ok]),
            "completion_rate": round(len(ok) / len(rows), 4) if rows else None,
            "completion_rate_ci": _wilson(len(ok), len(rows)),
            "pass_rate": _mean([1.0 if r.passed else 0.0 for r in rows]),
            "pass_rate_ci": _boot_mean_ci(case_pass),
            "by_category": {
                c: {"n": len(v), "cases": len({r.case_id for r in v}), "quality_score": _mean([r.score for r in v]),
                    "pass_rate": _mean([1.0 if r.passed else 0.0 for r in v])}
                for c, v in sorted(cats.items())
            },
            "ttft_ms_p50": _pct(ttft, 50),
            "ttft_ms_p95": _pct(ttft, 95),
            "ttft_answer_ms_p50": _pct(ttfa, 50),
            "latency_ms_p50": _pct(lat, 50),
            "latency_ms_p95": _pct(lat, 95),
            "latency_n": len(lat),
            "p95_unstable": len(lat) < 20,
            "tokens_per_s_mean": round(float(np.mean(tps)), 2) if tps else None,
            "tokens_per_s_median": round(float(np.median(tps)), 2) if tps else None,
            "tokens_source": sources.pop() if len(sources) == 1 else ("mixed" if sources else None),
            "output_tokens_mean": _mean([r.output_tokens for r in ok if r.output_tokens is not None]),
            "load_ms_max": max((r.load_ms for r in ok if r.load_ms is not None), default=None),
            "peak_ram_used_mb": max((r.peak_ram_used_mb for r in rows if r.peak_ram_used_mb), default=None),
            "avg_cpu_pct": _mean([r.avg_cpu_pct for r in ok if r.avg_cpu_pct is not None]),
            "runtime_rss_mb_peak": max((r.runtime_rss_mb for r in rows if r.runtime_rss_mb), default=None),
            "model_mem_mb": max((r.model_mem_mb for r in rows if r.model_mem_mb), default=None),
            "score_std_across_repeats": round(float(np.mean(stds)), 4) if stds else None,
        })
    out.sort(key=lambda s: (-(s["quality_score"] or 0), s["latency_ms_p50"] or 1e12))
    return out


def compare_models(results: list[BenchmarkResult], margin: float = EQUIV_MARGIN, seed: int = 0) -> dict:
    """Paired comparison of the top model (effective quality) with every other model, case by case.

    Δ = leader − other in per-case mean score (repeats averaged). Bootstrap over cases; two-sided p for a
    difference and TOST p for equivalence within ±margin, both Holm-adjusted. A non-significant difference is
    reported as "inconclusive", never as a tie.
    """
    from app.arena.stats import holm

    by_model: dict[str, dict[str, float]] = {}
    grouped: dict[str, list[BenchmarkResult]] = defaultdict(list)
    for r in results:
        grouped[r.model].append(r)
    for m, rows in grouped.items():
        by_model[m] = _case_means(rows)
    if len(by_model) < 2:
        return {"leader": next(iter(by_model), None), "comparisons": {}, "margin": margin}
    leader = max(by_model, key=lambda m: np.mean(list(by_model[m].values())))
    rng = np.random.default_rng(seed)
    comps: dict[str, dict] = {}
    p_diff, p_eq, keys = [], [], []
    for m, scores in by_model.items():
        if m == leader:
            continue
        common = sorted(set(scores) & set(by_model[leader]))
        if len(common) < 5:
            comps[m] = {"n_cases": len(common), "verdict": "not_tested"}
            continue
        d = np.array([by_model[leader][c] - scores[c] for c in common])
        boot = d[rng.integers(0, len(d), size=(BOOT, len(d)))].mean(axis=1)
        pd_ = min(1.0, 2 * min((1 + np.sum(boot <= 0)) / (BOOT + 1), (1 + np.sum(boot >= 0)) / (BOOT + 1)))
        pe_ = max((1 + np.sum(boot >= margin)) / (BOOT + 1), (1 + np.sum(boot <= -margin)) / (BOOT + 1))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        comps[m] = {"n_cases": len(common), "delta": round(float(d.mean()), 4),
                    "ci95": [round(float(lo), 4), round(float(hi), 4)], "p_diff": round(float(pd_), 6),
                    "p_equivalence": round(float(pe_), 6)}
        p_diff.append(pd_)
        p_eq.append(pe_)
        keys.append(m)
    for m, a, b in zip(keys, holm(p_diff), holm(p_eq), strict=True):
        c = comps[m]
        c["p_diff_holm"], c["p_equivalence_holm"] = a, b
        c["verdict"] = ("equivalent" if b is not None and b < 0.05 else
                        "worse" if a is not None and a < 0.05 and c["delta"] > 0 else "inconclusive")
    return {"leader": leader, "comparisons": comps, "margin": margin, "unit": "case (repeats averaged)",
            "correction": "Holm", "resamples": BOOT}


def result_dict(r: BenchmarkResult, include_response: bool = True) -> dict:
    d = {c.name: getattr(r, c.name) for c in BenchmarkResult.__table__.columns}
    d["created_at"] = r.created_at.isoformat() if r.created_at else None
    if not include_response:
        d.pop("response", None)
    return d


def _ts(dt) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S") if dt else "–"


def run_dict(run: BenchmarkRun, with_snapshot: bool = False) -> dict:
    d = {c.name: getattr(run, c.name) for c in BenchmarkRun.__table__.columns}
    if not with_snapshot:
        d.pop("suite_snapshot", None)
    for k in ("created_at", "started_at", "finished_at"):
        d[k] = d[k].isoformat() if d[k] else None
    d["duration_s"] = (
        round((run.finished_at - run.started_at).total_seconds(), 1) if run.finished_at and run.started_at else None
    )
    return d


def to_csv(results: list[BenchmarkResult]) -> str:
    buf = io.StringIO()
    cols = [c.name for c in BenchmarkResult.__table__.columns if c.name not in ("eval_details",)]
    w = csv.writer(buf)
    w.writerow(cols)
    for r in results:
        w.writerow([getattr(r, c) for c in cols])
    return buf.getvalue()


METRIC_LEGEND = """| Metric | Source |
|---|---|
| Effective quality | Mean evaluator score over all cases; failed or timed-out calls count as 0 (95% CI: bootstrap over cases) |
| Conditional quality | Mean evaluator score over the calls that completed |
| Completion rate | Share of calls without error/timeout (95% Wilson interval) |
| Pass rate | Share of results that met the evaluator's pass rule (deterministic, no LLM judge) |
| TTFT (answer) | **Measured** time to the first *visible* answer token (TTFT counts hidden thinking tokens too) |
| TTFT, total latency | **Measured** by Local AI Lab around the streaming HTTP call |
| Output tokens, tokens/s, load time | **Reported by the runtime** (Ollama `eval_count`/`eval_duration`); marked *estimated* (chars ÷ 4) when not reported |
| Peak RAM, CPU % | **Measured** system-wide by a background sampler during each request |
| Runtime RSS | **Measured** resident memory of local Ollama processes (when visible) |
| Model memory | **Reported by the runtime** (`/api/ps` size) |"""


SIMULATED_BANNER = ("SIMULATED RUN – produced by the built-in Ollama simulator for testing/demos. These numbers are "
                    "not measurements of real models.")


def to_markdown(run: BenchmarkRun, summary: list[dict], banner: str | None = None,
                comparison: dict | None = None) -> str:
    env = run.environment or {}
    cfg = run.config or {}
    lines = [f"# Benchmark report – {run.name}", ""]
    if getattr(run, "simulated", False):
        banner = SIMULATED_BANNER
    if banner:
        lines += [f"> **{banner}**", ""]
    lines += [
        f"- **Run ID:** {run.id}  ",
        f"- **Status:** {run.status}  ",
        f"- **Suite:** `{run.suite_id}` v{run.suite_version} (file sha256 `{run.suite_hash[:12]}…`, cases used sha256 "
        f"`{(cfg.get('cases_sha256') or '')[:12]}…`, {len(cfg.get('case_ids') or [])} cases)  ",
        f"- **Started / finished (UTC):** {_ts(run.started_at)} / {_ts(run.finished_at)}  ",
        f"- **Config:** temperature={cfg.get('temperature')}, seed={cfg.get('seed')}, repeats={cfg.get('repeats')}, "
        f"warmup={cfg.get('warmup')}, timeout={cfg.get('timeout_s')}s, num_ctx={cfg.get('num_ctx')}, "
        f"model order={cfg.get('model_order', 'as_given')}, case order={cfg.get('case_order', 'suite')}, "
        f"interleave={cfg.get('interleave', False)}  ",
        f"- **Integrity:** {_integrity_text(getattr(run, 'integrity', None))}  ",
        f"- **Machine:** {env.get('os')} · {env.get('cpu_count_logical')} logical CPUs · "
        f"{env.get('ram_total_mb')} MB RAM · GPUs: {', '.join(g['name'] for g in env.get('gpus', [])) or 'none detected'}  ",
        f"- **Ollama version:** {env.get('ollama_version')}  ",
        "",
        "## Summary",
        "",
        "| Model | Effective quality (95% CI) | Conditional quality | Completion | Pass rate | TTFT answer p50 (ms) | "
        "Latency p50 / p95 (ms, n) | Tokens/s | Peak RAM (MB) | Errors / timeouts |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summary:
        tps = f"{s['tokens_per_s_mean']}" + (" (est.)" if s["tokens_source"] == "estimated" else "")
        ci = s.get("effective_quality_ci")
        ci_s = f" ({ci[0]}–{ci[1]})" if ci else ""
        p95 = f"{s['latency_ms_p95']}" + ("*" if s.get("p95_unstable") else "")
        lines.append(
            f"| {s['model']} | {s['quality_score']}{ci_s} | {s.get('conditional_quality')} | "
            f"{s.get('completion_rate')} | {s['pass_rate']} | {s.get('ttft_answer_ms_p50')} | "
            f"{s['latency_ms_p50']} / {p95} ({s.get('latency_n')}) | {tps} | {s['peak_ram_used_mb']} | "
            f"{s['errors']} / {s['timeouts']} |"
        )
    lines += ["", "\\* p95 from fewer than 20 measurements – unstable, one slow response can dominate it."]
    if comparison and comparison.get("comparisons"):
        lines += ["", f"## Paired comparison with the top model ({comparison['leader']})", "",
                  f"Case-by-case differences in score (repeats averaged), bootstrap over cases, Holm-adjusted. "
                  f"Equivalence margin ±{comparison['margin']}. *Inconclusive* means the data cannot tell – it is "
                  "not a tie.", "",
                  "| Model | Δ quality (leader − model, 95% CI) | Verdict | p diff (Holm) | p equiv (Holm) | Cases |",
                  "|---|---|---|---|---|---|"]
        for m, c in comparison["comparisons"].items():
            if c.get("delta") is None:
                lines.append(f"| {m} | – | {c.get('verdict')} | – | – | {c.get('n_cases')} |")
                continue
            lines.append(f"| {m} | {c['delta']:+.3f} ({c['ci95'][0]:+.3f} to {c['ci95'][1]:+.3f}) | {c['verdict']} | "
                         f"{c['p_diff_holm']:.3f} | {c['p_equivalence_holm']:.3f} | {c['n_cases']} |")
    lines += ["", "## Quality by category", ""]
    cats = sorted({c for s in summary for c in s["by_category"]})
    lines.append("| Model | " + " | ".join(cats) + " |")
    lines.append("|---|" + "---|" * len(cats))
    for s in summary:
        lines.append(f"| {s['model']} | " + " | ".join(
            str(s["by_category"].get(c, {}).get("quality_score", "–")) for c in cats) + " |")
    n_cases = len(cfg.get("case_ids") or [])
    lines += ["", "## Metric provenance", "", METRIC_LEGEND, "",
              f"Scores describe performance on this fixed set of {n_cases} cases with these evaluators; they are not "
              "a general ranking of models and do not show how a model will do on different prompts. Repeats "
              "measure consistency – they do not add independent cases."]
    return "\n".join(lines) + "\n"


def _integrity_text(integrity: dict | None) -> str:
    if not integrity:
        return "not recorded (run created before v1.2)"
    if integrity.get("complete"):
        return f"complete – {integrity['stored_results']} of {integrity['expected_results']} results stored"
    return (f"INCOMPLETE – {integrity.get('stored_results')} of {integrity.get('expected_results')} results stored, "
            f"{integrity.get('duplicates', 0)} duplicates")
