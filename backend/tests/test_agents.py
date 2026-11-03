"""Agent Arena: tools, scoring, suites, runs (native + prompted), playground, privacy."""

from __future__ import annotations

import json

import pytest
from conftest import stream_events, wait_for

from app.agents.loop import parse_prompted
from app.agents.scoring import score_task
from app.agents.suites import AgentTask, parse_suite
from app.agents.tools import MISSING, ToolExecutor, ToolSpec, match_value, safe_eval, validate_args
from app.core.errors import ValidationFailed


def _wait(c, rid, headers=None):
    return wait_for(lambda: (d := c.get(f"/api/agents/runs/{rid}", headers=headers or {}).json())["status"]
                    not in ("queued", "running") and d, timeout=120)


# ----------------------------------------------------------------------------- units
def test_matchers():
    assert match_value("Lahore", "lahore ") and not match_value("Lahore", "Karachi")
    assert match_value(45, "45") and match_value(45, 45.0) and not match_value(45, 46)
    assert match_value({"contains": "invoice"}, "Re: Invoice 2041")
    assert match_value({"any_of": ["USD", "dollar"]}, "usd")
    assert match_value({"regex": "^A-\\d+$"}, "A-1001") and not match_value({"regex": "^A-\\d+$"}, "1001")
    assert match_value({"number": 434, "tolerance": 0.5}, 434.2)
    assert match_value({"absent": True}, MISSING) and not match_value({"exists": True}, MISSING)


def test_schema_validation_and_executor():
    schema = {"type": "object", "properties": {"n": {"type": "integer"}, "unit": {"type": "string", "enum": ["c", "f"]}},
              "required": ["n"], "additionalProperties": False}
    assert validate_args(schema, {"n": 3, "unit": "c"}) == []
    errs = validate_args(schema, {"unit": "k", "x": 1})
    assert any("missing required" in e for e in errs) and any("one of" in e for e in errs) and any("unknown" in e for e in errs)
    tool = ToolSpec(name="lookup", description="d", parameters={"type": "object", "properties": {"id": {"type": "string"}},
                                                                "required": ["id"]},
                    responses=[{"when": {"id": "A"}, "result": {"ok": 1}}, {"when": {}, "result": {"fallback": True}}])
    ex = ToolExecutor([tool, ToolSpec(name="calc", builtin="calculator")])
    assert ex.call("lookup", {"id": "a"})[0] == {"ok": 1}
    assert ex.call("lookup", {"id": "zzz"})[0] == {"fallback": True}
    res, meta = ex.call("lookup", {})
    assert not meta["valid"] and "Invalid arguments" in res["error"]
    res, meta = ex.call("nope", {})
    assert meta["unknown"] and "Unknown tool" in res["error"]
    assert ex.call("calc", '{"expression": "2480 * 17.5%"}')[0]["result"] == pytest.approx(434)


def test_calculator_is_safe():
    assert safe_eval("2 ** 10 + sqrt(16)") == 1028
    for bad in ("__import__('os').system('echo hi')", "(1).__class__", "open('x')", "9 ** 999999", "[x for x in ()]"):
        with pytest.raises((ValueError, SyntaxError)):
            safe_eval(bad)


def test_currency_converter_builtin():
    ex = ToolExecutor([ToolSpec(name="fx", builtin="currency_converter")])
    out = ex.call("fx", {"amount": 250, "from_currency": "usd", "to_currency": "PKR"})[0]
    assert out["converted"] == 70000
    assert "Unsupported" in ex.call("fx", {"amount": 1, "from_currency": "XXX", "to_currency": "PKR"})[0]["error"]


def test_scoring_checks():
    task = AgentTask(id="t", prompt="p", expect={
        "tool_calls": [{"name": "a", "arguments": {"x": 1}}, {"name": "b", "arguments": {"y": "z"}}],
        "order": "strict", "forbidden_tools": ["danger"], "final_answer": {"type": "contains_all", "keywords": ["done"]}})
    good = score_task(task, [{"name": "a", "arguments": {"x": 1}}, {"name": "b", "arguments": {"y": "Z"}}], "All done", "final")
    assert good["success"] and good["score"] == 1.0
    wrong_order = score_task(task, [{"name": "b", "arguments": {"y": "z"}}, {"name": "a", "arguments": {"x": 1}}], "done", "final")
    assert not wrong_order["checks"]["order"]["passed"] and wrong_order["checks"]["tool_selection"]["passed"]
    unsafe = score_task(task, [{"name": "danger", "arguments": {}}, {"name": "a", "arguments": {"x": 2}}], "done", "final")
    assert not unsafe["success"] and unsafe["checks"]["safety"]["forbidden_called"] == ["danger"]
    assert unsafe["checks"]["arguments"]["score"] == 0.0 and unsafe["checks"]["tool_selection"]["score"] == 0.5
    no_tool = AgentTask(id="n", prompt="p", expect={"no_tools": True})
    assert score_task(no_tool, [], "hi", "final")["success"]
    assert not score_task(no_tool, [{"name": "a", "arguments": {}}], "hi", "final")["success"]
    assert not score_task(no_tool, [], "", "max_steps")["checks"]["completed"]["passed"]


def test_prompted_protocol_parsing():
    assert parse_prompted('{"tool": "get_weather", "arguments": {"city": "Lahore"}}') == \
        ("tool", [{"name": "get_weather", "arguments": {"city": "Lahore"}}], None)
    assert parse_prompted('```json\n{"final_answer": "34 C"}\n```') == ("final", [], "34 C")
    assert parse_prompted("It is sunny.")[0] == "invalid"
    kind, calls, _ = parse_prompted('{"tool_calls": [{"name": "a", "arguments": "{\\"x\\": 1}"}]}')
    assert kind == "tool" and calls[0]["arguments"] == {"x": 1}


def test_suite_validation_errors():
    base = {"id": "my-suite", "name": "S", "tools": [{"name": "a", "description": "d"}],
            "tasks": [{"id": "1", "prompt": "p", "expect": {"tool_calls": [{"name": "a"}]}}]}
    assert parse_suite(base, "x").summary()["task_count"] == 1
    for bad, msg in [
        ({**base, "tools": base["tools"] * 2}, "unique"),
        ({**base, "tasks": [{"id": "1", "prompt": "p", "expect": {"tool_calls": [{"name": "zzz"}]}}]}, "not available"),
        ({**base, "tasks": [{"id": "1", "prompt": "p", "expect": {"final_answer": {"type": "magic"}}}]}, "evaluator"),
        ({**base, "id": "Bad Id!"}, "suite id"),
        ({**base, "tools": [{"name": "x", "builtin": "rm_rf"}]}, "built-in"),
    ]:
        with pytest.raises(ValidationFailed, match=msg):
            parse_suite(bad, "x")


# ----------------------------------------------------------------------------- API
def test_builtin_suites_are_listed_and_valid(client):
    suites = {s["id"]: s for s in client.get("/api/agents/suites").json()["suites"]}
    assert suites["agent-core"]["task_count"] == 23 and suites["agent-quick"]["task_count"] == 6
    assert set(suites["agent-core"]["categories"]) == {"single_tool", "argument_extraction", "multi_step", "no_tool",
                                                       "safety", "error_recovery"}
    d = client.get("/api/agents/suites/agent-core").json()["definition"]
    assert any(t["risky"] for t in d["tools"])


def test_run_native_and_prompted_fallback(client):
    r = client.post("/api/agents/runs", json={"suite_id": "agent-quick", "models": ["sim-medium:3b", "sim-small:1b"]})
    assert r.status_code == 201, r.text
    run = _wait(client, r.json()["id"])
    assert run["status"] == "completed", run["error"]
    info = run["model_info"]
    assert info["sim-medium:3b"]["mode"] == "native" and info["sim-small:1b"]["mode"] == "prompted"
    assert info["sim-small:1b"]["supports_native_tools"] is False
    assert len(run["results"]) == 12 and {r["mode"] for r in run["results"]} == {"native", "prompted"}
    summary = {s["model"]: s for s in run["summary"]}
    for s in summary.values():
        assert 0 <= s["success_rate"] <= 1 and s["by_category"]
    res = next(r for r in run["results"] if r["task_id"] == "ag-st-01" and r["model"] == "sim-medium:3b")
    detail = client.get(f"/api/agents/runs/{run['id']}/results/{res['id']}").json()
    types = [e["type"] for e in detail["trace"]]
    assert types[0] == "start" and types[-1] == "final"
    assert run["tasks"]["ag-st-01"]["prompt"].startswith("What's the weather")
    md = client.get(f"/api/agents/runs/{run['id']}/export?format=md").text
    assert "Leaderboard" in md and "sim-small:1b" in md
    assert client.get(f"/api/agents/runs/{run['id']}/export?format=csv").text.startswith("model,task_id")
    assert len(client.get(f"/api/agents/runs/{run['id']}/export?format=json").json()["results"][0]["trace"]) >= 2
    assert client.delete(f"/api/agents/runs/{run['id']}").status_code == 204


def test_forced_native_mode_is_rejected_for_models_without_tools(client):
    r = client.post("/api/agents/runs", json={"suite_id": "agent-quick", "models": ["sim-small:1b"], "mode": "native"})
    assert r.status_code == 422 and "native tool calling" in r.json()["error"]["message"]
    r = client.post("/api/agents/runs", json={"suite_id": "agent-quick", "models": ["sim-small:1b"], "mode": "prompted",
                                              "categories": ["no_tool"]})
    run = _wait(client, r.json()["id"])
    assert run["status"] == "completed" and {x["mode"] for x in run["results"]} == {"prompted"}


def test_custom_suite_lifecycle(client):
    tpl = client.get("/api/agents/template").json()
    assert client.post("/api/agents/suites/validate", json=tpl).json()["valid"]
    saved = client.post("/api/agents/suites", json=tpl)
    assert saved.status_code == 201 and saved.json()["source"] == "custom"
    ids = [s["id"] for s in client.get("/api/agents/suites").json()["suites"]]
    assert tpl["id"] in ids
    run = _wait(client, client.post("/api/agents/runs", json={"suite_id": tpl["id"], "models": ["sim-medium:3b"]}).json()["id"])
    assert run["status"] == "completed" and len(run["results"]) == 3
    clash = client.post("/api/agents/suites", json={**tpl, "id": "agent-core"})
    assert clash.status_code == 422 and "built-in" in clash.json()["error"]["message"]
    bad = client.post("/api/agents/suites", json={**tpl, "tasks": []})
    assert bad.status_code == 422
    assert client.delete(f"/api/agents/suites/{tpl['id']}").status_code == 204
    assert client.delete("/api/agents/suites/agent-core").status_code == 404


def test_playground_streams_and_searches_documents(client):
    up = client.post("/api/documents", files={"file": ("policy.txt", b"Our refund window is 30 days from delivery.", "text/plain")})
    assert up.status_code == 201
    events = stream_events(client.post("/api/agents/playground", json={
        "model": "sim-medium:3b", "prompt": "Please search my documents for the refund window.",
        "extra_builtins": ["search_documents"]}))
    tool = next(e for e in events if e["type"] == "tool")
    assert tool["name"] == "search_documents" and tool["result"]["results"][0]["file"] == "policy.txt"
    assert events[-1]["type"] == "final"
    events = stream_events(client.post("/api/agents/playground", json={
        "model": "sim-small:1b", "prompt": "What is the status of order A-1001?", "suite_id": "agent-core"}))
    assert events[0]["mode"] == "prompted" and events[-1]["type"] == "final"


def test_cloud_models_blocked_and_runs_isolated(make_app):
    c = make_app(auth_enabled=True)

    def login(u):
        c.post("/api/auth/register", json={"username": u, "password": "password123"})
        return {"Authorization": "Bearer " + c.post("/api/auth/login", json={"username": u, "password": "password123"}).json()["token"]}

    a, b = login("alice2"), login("bob2")
    r = c.post("/api/agents/runs", headers=a, json={"suite_id": "agent-quick", "models": ["sim-cloud:cloud"]})
    assert r.status_code == 403 and r.json()["error"]["code"] == "offline_mode_blocked"
    rid = c.post("/api/agents/runs", headers=a, json={"suite_id": "agent-quick", "models": ["sim-medium:3b"],
                                                      "categories": ["no_tool"]}).json()["id"]
    _wait(c, rid, a)
    assert c.get(f"/api/agents/runs/{rid}", headers=b).status_code == 404
    assert c.get("/api/agents/runs", headers=b).json()["runs"] == []
    tpl = c.get("/api/agents/template", headers=a).json()
    assert c.post("/api/agents/suites", headers=a, json=tpl).status_code == 201
    assert tpl["id"] not in [s["id"] for s in c.get("/api/agents/suites", headers=b).json()["suites"]]


def test_jobs_are_serialised_by_the_shared_lock(client):
    from app.core.jobs import JobLock

    lock = JobLock()
    assert lock.acquire("one") and lock.status() == {"busy": True, "holder": "one"}
    import threading

    stop = threading.Event()
    stop.set()
    assert lock.acquire("two", stop) is False
    lock.release()
    bench = client.post("/api/benchmarks/runs", json={"suite_id": "quick", "models": ["sim-small:1b"], "max_cases": 2}).json()
    agent = client.post("/api/agents/runs", json={"suite_id": "agent-quick", "models": ["sim-medium:3b"],
                                                  "categories": ["single_tool"]}).json()
    wait_for(lambda: client.get(f"/api/benchmarks/runs/{bench['id']}").json()["status"] == "completed")
    assert _wait(client, agent["id"])["status"] == "completed"
    json.dumps(agent)
