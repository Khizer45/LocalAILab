"""Agreement between the deterministic evaluators and adjudicated judgements.

Measures the error rate of the evaluators themselves on
``datasets/evaluator_calibration/*.jsonl`` (see the README there):

* false pass – the evaluator passed a response the graders judged wrong;
* false fail – the evaluator failed a response the graders judged right;
* Cohen's kappa – agreement beyond chance.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from app.benchmark.datasets import SuiteRepository
from app.benchmark.evaluators import evaluate_spec, spec_version


def _rates(rows: list[dict]) -> dict:
    tp = sum(r["evaluator_pass"] and r["human_pass"] for r in rows)
    tn = sum(not r["evaluator_pass"] and not r["human_pass"] for r in rows)
    fp = sum(r["evaluator_pass"] and not r["human_pass"] for r in rows)
    fn = sum(not r["evaluator_pass"] and r["human_pass"] for r in rows)
    n = len(rows)
    po = (tp + tn) / n if n else 0.0
    pe = (((tp + fp) * (tp + fn)) + ((tn + fn) * (tn + fp))) / (n * n) if n else 0.0
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    return {"n": n, "agreement": round(po, 4), "false_pass": fp, "false_fail": fn,
            "false_pass_rate": round(fp / (fp + tn), 4) if fp + tn else None,
            "false_fail_rate": round(fn / (fn + tp), 4) if fn + tp else None, "kappa": round(kappa, 4)}


def agreement(datasets_dir: Path, path: Path | None = None) -> dict:
    suites = SuiteRepository(datasets_dir).all()
    files = [path] if path else sorted((datasets_dir / "evaluator_calibration").glob("*.jsonl"))
    rows: list[dict] = []
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            case = next(c for c in suites[item["suite"]].cases if c.id == item["case_id"])
            ev = evaluate_spec(item["response"], case.evaluator)
            rows.append({"id": item["id"], "case_id": case.id, "evaluator": spec_version(case.evaluator),
                         "type": case.evaluator["type"], "human_pass": bool(item["human_pass"]),
                         "evaluator_pass": ev.passed, "adjudicated_by": item.get("adjudicated_by"),
                         "response": item["response"], "note": item.get("note")})
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    return {"overall": _rates(rows), "by_evaluator": {t: _rates(v) for t, v in sorted(by_type.items())},
            "draft_labels": sum(r["adjudicated_by"] in (None, "draft") for r in rows),
            "disagreements": [{k: r[k] for k in ("id", "case_id", "evaluator", "human_pass", "evaluator_pass",
                                                 "response", "note")}
                              for r in rows if r["human_pass"] != r["evaluator_pass"]]}
