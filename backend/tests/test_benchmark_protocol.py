"""LLM benchmark (v1.2): frozen suites, quality vs completion, paired comparisons, order, integrity, simulator flag."""

from __future__ import annotations

import json
import shutil

from conftest import wait_for

from app.benchmark.aggregate import compare_models, summarize
from app.benchmark.evaluators import spec_version
from app.benchmark.runner import execution_plan
from app.core.config import BACKEND_DIR
from app.models.benchmark import BenchmarkResult


def _run(c, **body):
    r = c.post("/api/benchmarks/runs", json=body)
    assert r.status_code == 201, r.text
    rid = r.json()["id"]
    return wait_for(lambda: (d := c.get(f"/api/benchmarks/runs/{rid}").json())["status"]
                    not in ("queued", "running") and d, timeout=60)


def _res(model, case, score, status="ok", rep=0, passed=None):
    return BenchmarkResult(model=model, case_id=case, category="c", repeat_index=rep, status=status, score=score,
                           passed=score >= 1 if passed is None else passed, total_ms=100.0, ttft_ms=10.0,
                           tokens_source="runtime")


def test_effective_conditional_quality_and_completion_are_separate():
    rows = [_res("m", f"c{i}", 1.0) for i in range(6)] + [_res("m", f"c{i}", 0.0, status="timeout")
                                                          for i in range(6, 8)]
    rows += [_res("m", "c8", 0.5), _res("m", "c9", 0.0, status="error")]
    s = summarize(rows)[0]
    assert s["completion_rate"] == 0.7
    assert s["effective_quality"] == s["quality_score"] == round(6.5 / 10, 3)
    assert s["conditional_quality"] == round(6.5 / 7, 3)  # failures excluded
    assert s["completion_rate_ci"][0] < 0.7 < s["completion_rate_ci"][1]
    assert s["latency_n"] == 7 and s["p95_unstable"] is True


def test_paired_comparison_reports_inconclusive_not_tie():
    rows = []
    for i in range(40):
        rows += [_res("A", f"c{i}", 1.0), _res("same", f"c{i}", 1.0), _res("B", f"c{i}", 0.0 if i % 2 else 1.0)]
    for i in range(6):  # few cases: cannot tell
        rows += [_res("A2", f"d{i}", 1.0 if i % 3 else 0.0), _res("B2", f"d{i}", 1.0 if i % 2 else 0.0)]
    comp = compare_models([r for r in rows if r.model in ("A", "same", "B")])
    assert comp["leader"] in ("A", "same")
    other = "same" if comp["leader"] == "A" else "A"
    assert comp["comparisons"][other]["verdict"] == "equivalent"
    assert comp["comparisons"]["B"]["verdict"] == "worse" and comp["correction"] == "Holm"
    small = compare_models([r for r in rows if r.model in ("A2", "B2")])
    assert next(iter(small["comparisons"].values()))["verdict"] == "inconclusive"


def test_execution_plan_order_options():
    plain = execution_plan(["a", "b"], ["1", "2"], 1, {})
    assert plain == [("a", "1", 0), ("a", "2", 0), ("b", "1", 0), ("b", "2", 0)]
    inter = execution_plan(["a", "b"], ["1", "2"], 1, {"interleave": True})
    assert inter == [("a", "1", 0), ("b", "1", 0), ("a", "2", 0), ("b", "2", 0)]
    shuffled = execution_plan(["a", "b", "c"], [str(i) for i in range(10)], 2,
                              {"model_order": "shuffled", "case_order": "shuffled", "seed": 3})
    assert sorted(shuffled) == sorted(execution_plan(["a", "b", "c"], [str(i) for i in range(10)], 2, {}))
    assert shuffled == execution_plan(["a", "b", "c"], [str(i) for i in range(10)], 2,
                                      {"model_order": "shuffled", "case_order": "shuffled", "seed": 3})


def test_run_records_versions_answer_ttft_order_integrity_and_simulator(client):
    run = _run(client, suite_id="quick", models=["sim-small:1b", "sim-medium:3b"], case_order="shuffled",
               model_order="shuffled", seed=5)
    assert run["status"] == "completed" and run["simulated"] is True
    assert run["integrity"] == {"expected_results": 16, "stored_results": 16, "duplicates": 0, "missing": 0,
                                "complete": True}
    assert sorted(r["order_index"] for r in run["results"]) == list(range(16))
    assert all("@" in r["evaluator_version"] for r in run["results"])
    assert all(r["ttft_answer_ms"] is not None for r in run["results"])
    assert run["config"]["evaluator_versions"]["numeric"] == "2" and len(run["config"]["cases_sha256"]) == 64
    s = run["summary"][0]
    assert s["completion_rate"] == 1.0 and s["effective_quality_ci"] is not None
    assert run["comparison"]["leader"] in ("sim-small:1b", "sim-medium:3b")
    md = client.get(f"/api/benchmarks/runs/{run['id']}/export?format=md").text
    assert "SIMULATED RUN" in md and "Integrity:** complete" in md and "Paired comparison" in md
    js = client.get(f"/api/benchmarks/runs/{run['id']}/export?format=json").json()
    assert len(js["suite_snapshot"]["cases"]) == 8 and "comparison" in js


def test_suite_is_frozen_when_the_run_is_submitted(make_app, tmp_path):
    datasets = tmp_path / "datasets"
    shutil.copytree(BACKEND_DIR / "datasets", datasets)
    c = make_app(datasets_dir=datasets)
    lock = c.app.state.ctx.job_lock
    assert lock.acquire("test holds the worker", None)
    try:
        r = c.post("/api/benchmarks/runs", json={"suite_id": "quick", "models": ["sim-small:1b"], "max_cases": 3})
        rid = r.json()["id"]
        path = datasets / "benchmark" / "quick_v1.json"
        suite = json.loads(path.read_text())
        original = {c["id"]: spec_version(c["evaluator"]) for c in suite["cases"]}
        for case in suite["cases"]:  # sabotage the file after submission
            case["evaluator"] = {"type": "regex", "pattern": "^IMPOSSIBLE$"}
        path.write_text(json.dumps(suite))
    finally:
        lock.release()
    run = wait_for(lambda: (d := c.get(f"/api/benchmarks/runs/{rid}").json())["status"] == "completed" and d)
    assert all(r["evaluator_version"] == original[r["case_id"]] for r in run["results"])
    assert any(r["passed"] for r in run["results"])  # the sabotaged "^IMPOSSIBLE$" rule was never used


def test_incomplete_results_fail_the_integrity_check(client, monkeypatch):
    from app.benchmark.runner import BenchmarkManager

    original = BenchmarkManager._store
    calls = {"n": 0}

    def flaky(self, run_id, result):
        calls["n"] += 1
        if calls["n"] == 2:  # lose one result, as a crash between generation and storage would
            return None
        return original(self, run_id, result)

    monkeypatch.setattr(BenchmarkManager, "_store", flaky)
    run = _run(client, suite_id="quick", models=["sim-small:1b"], max_cases=3)
    assert run["status"] == "failed" and "Integrity check failed" in run["error"]
    assert run["integrity"]["missing"] == 1
