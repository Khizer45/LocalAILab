from __future__ import annotations

import time

import pytest
from conftest import wait_for

from app.benchmark.datasets import SuiteRepository
from app.benchmark.evaluators import evaluate_spec
from app.core.config import BACKEND_DIR
from app.core.errors import InferenceTimeout
from app.services.inference import GenerationResult, stream_generate, strip_think
from app.services.model_registry import ResolvedModel


# ------------------------------------------------------------------ evaluators
@pytest.mark.parametrize("response,spec,passed", [
    ("Tokyo.", {"type": "exact_match", "answers": ["Tokyo"]}, True),
    ("The capital is Tokyo", {"type": "exact_match", "answers": ["Tokyo"]}, False),
    ("Answer: 102", {"type": "numeric", "value": 102}, True),
    ("6 x 17 = 103", {"type": "numeric", "value": 102}, False),
    ("It costs 2,500 rupees", {"type": "numeric", "value": 2500}, True),
    ("central processing unit", {"type": "contains_all", "keywords": ["central", "processing", "unit"]}, True),
    ("Yes.", {"type": "regex", "pattern": r"^\W*yes\W*$", "flags": "i"}, True),
    ("one two three", {"type": "word_limit", "max_words": 2}, False),
    ("- a\n- b\n- c", {"type": "bullet_count", "count": 3}, True),
    ("Here you go:\n- a\n- b\n- c", {"type": "bullet_count", "count": 3}, False),
    ('```json\n{"name": "Sara", "age": "29"}\n```', {"type": "json_object", "required_keys": ["name", "age"],
                                                       "expected": {"name": "sara", "age": 29}}, True),
    ("not json", {"type": "json_object", "required_keys": ["a"]}, False),
    ("[1,2,3,4,5]", {"type": "json_equals", "value": [1, 2, 3, 4, 5]}, True),
    ('["#AI", "#offline", "#MUST"]', {"type": "json_equals", "value": ["offline", "must", "ai"], "ignore_order": True,
                                       "strip_chars": "#"}, True),
    ("billing", {"type": "label", "labels": ["billing", "technical_support"], "answer": "billing"}, True),
    ("The intent is technical support.", {"type": "label", "labels": ["billing", "technical_support"],
                                          "answer": "technical_support"}, True),
    ("NOT IN CONTEXT", {"type": "abstain"}, True),
    ("The ticket costs 500.", {"type": "abstain"}, False),
    ("cloud computing for all", {"type": "regex", "pattern": r"\bcloud\b", "flags": "i", "negate": True}, False),
])
def test_evaluators(response, spec, passed):
    result = evaluate_spec(response, spec)
    assert result.passed is passed
    assert 0.0 <= result.score <= 1.0


def test_partial_credit_and_composites():
    r = evaluate_spec('{"name": "Hamza", "city": "Karachi", "age": 34}',
                      {"type": "json_object", "required_keys": ["name", "city", "age"],
                       "expected": {"name": "Hamza", "city": "Islamabad", "age": 34}})
    assert not r.passed and 0.4 < r.score < 1.0
    r = evaluate_spec("Solar panels cut electricity bills.", {"type": "all_of", "checks": [
        {"type": "contains_all", "keywords": ["solar", "electricity", "grid"], "threshold": 0.66},
        {"type": "word_limit", "max_words": 35}]})
    assert r.passed
    assert evaluate_spec("the cat sat", {"type": "rouge_l", "reference": "the cat sat on the mat"}).score > 0.5
    assert evaluate_spec("cat", {"type": "token_f1", "reference": "the cat"}).score == 1.0


def test_strip_think():
    assert strip_think("<think>reasoning</think>\nTokyo") == "Tokyo"
    assert strip_think("half reasoning</think> Tokyo") == "Tokyo"


# ------------------------------------------------------------------ suites
def test_all_suites_are_valid_and_versioned():
    suites = SuiteRepository(BACKEND_DIR / "datasets").all()
    assert {"quick", "core", "rag-grounding", "intent-classification"} <= set(suites)
    for s in suites.values():
        assert s.version and len(s.sha256) == 64 and s.cases
    assert len(suites["core"].cases) == 32
    assert suites["intent-classification"].summary()["case_count"] == 23
    # every reference answer must pass its own evaluator (dataset self-consistency)
    for s in suites.values():
        for case in s.cases:
            assert evaluate_spec(case.reference, case.evaluator).passed, (s.id, case.id)


# ------------------------------------------------------------------ timeout handling
class _SlowClient:
    def chat_stream(self, payload, timeout):
        for i in range(100):
            time.sleep(0.02)
            yield {"message": {"content": f"t{i} "}, "done": False}


def test_generation_deadline_raises_timeout():
    model = ResolvedModel(ref="slow", name="slow", provider="local", runs_in_cloud=False, client=_SlowClient())
    res = GenerationResult()
    with pytest.raises(InferenceTimeout):
        for _ in stream_generate(model, [], options={}, timeout_s=0.1, result=res):
            pass
    assert res.content  # partial output is kept


# ------------------------------------------------------------------ end-to-end runs
def _run(client, **body):
    r = client.post("/api/benchmarks/runs", json=body)
    assert r.status_code == 201, r.text
    run_id = r.json()["id"]
    wait_for(lambda: client.get(f"/api/benchmarks/runs/{run_id}").json()["status"] in ("completed", "failed", "cancelled"),
             timeout=60)
    return client.get(f"/api/benchmarks/runs/{run_id}").json()


def test_quick_suite_two_models(client):
    run = _run(client, suite_id="quick", models=["sim-small:1b", "sim-medium:3b"], repeats=2)
    assert run["status"] == "completed"
    assert run["progress_done"] == run["progress_total"] == 2 * 8 * 2
    assert len(run["results"]) == 32
    assert run["environment"]["ollama_version"] == "0.0.0-simulator"
    assert run["model_info"]["sim-small:1b"]["warmup"]["total_ms"] is not None
    summary = {s["model"]: s for s in run["summary"]}
    for s in summary.values():
        assert s["errors"] == 0 and s["timeouts"] == 0
        assert s["tokens_source"] == "runtime" and s["tokens_per_s_mean"] > 0
        assert s["ttft_ms_p50"] is not None and s["latency_ms_p95"] >= s["latency_ms_p50"]
        assert s["score_std_across_repeats"] == 0  # deterministic at temperature 0
        assert set(s["by_category"]) >= {"question_answering", "reasoning"}
    r = run["results"][0]
    assert r["eval_details"]["method"] and r["peak_ram_used_mb"] > 0


def test_runs_are_reproducible(client):
    a = _run(client, suite_id="core", models=["sim-small:1b"], max_cases=10)
    b = _run(client, suite_id="core", models=["sim-small:1b"], max_cases=10)
    assert [r["score"] for r in a["results"]] == [r["score"] for r in b["results"]]
    assert a["suite_hash"] == b["suite_hash"]


def test_category_filter_and_exports(client):
    run = _run(client, suite_id="core", models=["sim-medium:3b"], categories=["reasoning"], warmup=False)
    assert {r["category"] for r in run["results"]} == {"reasoning"}
    csv_text = client.get(f"/api/benchmarks/runs/{run['id']}/export?format=csv").text
    assert csv_text.splitlines()[0].startswith("id,run_id,model")
    md = client.get(f"/api/benchmarks/runs/{run['id']}/export?format=md").text
    assert "# Benchmark report" in md and "Metric provenance" in md
    js = client.get(f"/api/benchmarks/runs/{run['id']}/export?format=json").json()
    assert js["summary"][0]["model"] == "sim-medium:3b"
    ov = client.get("/api/benchmarks/overview").json()
    assert ov["completed_runs"] == 1 and ov["best"]["model"] == "sim-medium:3b"


def test_invalid_requests(client):
    assert client.post("/api/benchmarks/runs", json={"suite_id": "quick", "models": ["nope:1b"]}).status_code == 404
    assert client.post("/api/benchmarks/runs", json={"suite_id": "missing", "models": ["sim-small:1b"]}).status_code == 404
    assert client.post("/api/benchmarks/runs", json={"suite_id": "quick", "models": []}).status_code == 422
    r = client.post("/api/benchmarks/runs", json={"suite_id": "quick", "models": ["sim-small:1b"], "categories": ["poetry"]})
    assert r.status_code == 422


def test_cancel_and_delete(make_app):
    from conftest import _Server

    with _Server(speed=0.5) as slow_url:  # realistic delays so the run is still going
        c = make_app(ollama_base_url=slow_url)
        run = c.post("/api/benchmarks/runs", json={"suite_id": "core", "models": ["sim-medium:3b"]}).json()
        wait_for(lambda: c.get(f"/api/benchmarks/runs/{run['id']}").json()["status"] == "running")
        assert c.delete(f"/api/benchmarks/runs/{run['id']}").status_code == 409
        assert c.post(f"/api/benchmarks/runs/{run['id']}/cancel").status_code == 200
        final = wait_for(lambda: (d := c.get(f"/api/benchmarks/runs/{run['id']}").json())["status"] == "cancelled" and d)
        assert final["progress_done"] < final["progress_total"]
        assert c.delete(f"/api/benchmarks/runs/{run['id']}").status_code == 204


def test_mlflow_tracking(make_app):
    c = make_app(mlflow_enabled=True)
    run = _run(c, suite_id="quick", models=["sim-small:1b"], max_cases=2)
    assert run["mlflow_run_id"]
    import mlflow

    mlflow.set_tracking_uri(c.app.state.ctx.settings.mlflow_uri)
    data = mlflow.get_run(run["mlflow_run_id"]).data
    assert data.params["suite_id"] == "quick"
    assert any(k.endswith("quality_score") for k in data.metrics)
