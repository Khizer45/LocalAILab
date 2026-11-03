"""Work Agent (real tools, approvals, audit), outgoing webhooks and schedules."""

from __future__ import annotations

import json
import threading
import time
from datetime import timedelta

import pytest
import uvicorn
from fastapi import FastAPI, Request

from app.db.base import utcnow
from app.services.webhooks import sign

RECEIVED: list[dict] = []


@pytest.fixture(scope="module")
def receiver():
    """A local HTTP endpoint standing in for an n8n webhook."""
    from conftest import _free_port

    app = FastAPI()

    @app.post("/webhook/{path}")
    async def hook(path: str, request: Request):
        body = await request.body()
        RECEIVED.append({"path": path, "headers": dict(request.headers), "body": body})
        return {"received": True, "workflow": path}

    port = _free_port()
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    while not srv.started:
        time.sleep(0.02)
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True


def _events(text: str) -> list[dict]:
    return [json.loads(x) for x in text.splitlines() if x.strip()]


def _run(c, prompt, **kw):
    return c.post("/api/work-agent/run", json={"model": "sim-medium:3b", "prompt": prompt, **kw})


def _interactive(c, prompt, approve: bool | None, note=None):
    """Run in a thread; answer the approval request when it appears."""
    out = {}

    def go():
        out["events"] = _events(_run(c, prompt).text)

    t = threading.Thread(target=go)
    t.start()
    if approve is not None:
        end = time.time() + 20
        pend = []
        while time.time() < end and not pend:
            pend = c.get("/api/work-agent/approvals").json()["approvals"]
            time.sleep(0.05)
        assert pend, "no approval request appeared"
        c.post(f"/api/work-agent/approvals/{pend[0]['id']}", json={"approve": approve, "note": note})
    t.join(timeout=30)
    return out["events"]


def test_read_tools_run_and_paths_stay_inside_the_workspace(client):
    ws = client.app.state.ctx.settings.workspace_dir
    (ws / "notes.txt").write_text("Budget meeting moved to Friday.")
    ev = _events(_run(client, 'CALL read_file {"path": "notes.txt"}').text)
    tool = next(e for e in ev if e["type"] == "tool")
    assert tool["result"]["content"].startswith("Budget") and tool["decision"] == "allowed"
    ev = _events(_run(client, 'CALL read_file {"path": "../local_ai_lab.db"}').text)
    assert "error" in next(e for e in ev if e["type"] == "tool")["result"]
    audit = client.get("/api/system/audit?action=work_agent").json()["events"]
    assert {"work_agent.run", "work_agent.tool"} <= {e["action"] for e in audit}


def test_writes_need_approval_and_rejection_is_respected(client):
    ws = client.app.state.ctx.settings.workspace_dir
    ev = _interactive(client, 'CALL write_file {"path": "report.md", "content": "# Weekly report"}', approve=False,
                      note="not now")
    assert any(e["type"] == "approval_required" and e["preview"]["path"] == "report.md" for e in ev)
    assert not (ws / "report.md").exists()
    assert "did not approve" in json.dumps(next(e for e in ev if e["type"] == "tool")["result"])
    ev = _interactive(client, 'CALL write_file {"path": "report.md", "content": "# Weekly report"}', approve=True)
    assert (ws / "report.md").read_text() == "# Weekly report"
    assert next(e for e in ev if e["type"] == "approval_decision")["decision"] == "approved"
    actions = [e["action"] for e in client.get("/api/system/audit?action=work_agent").json()["events"]]
    assert "work_agent.approve" in actions and "work_agent.reject" in actions


def test_unattended_runs_cannot_take_gated_actions(client):
    ws = client.app.state.ctx.settings.workspace_dir
    r = _run(client, 'CALL write_file {"path": "x.txt", "content": "y"}', interactive=False).json()
    tool = next(e for e in r["trace"] if e["type"] == "tool")
    assert "needs a person's approval" in tool["result"]["error"] and not (ws / "x.txt").exists()
    r = _run(client, 'CALL draft_email {"to": "a@b.example", "subject": "Hi", "body": "Draft"}', interactive=False).json()
    draft = next(e for e in r["trace"] if e["type"] == "tool")["result"]
    assert draft["sent"] is False and (ws / draft["draft"]).exists()


def test_policy_can_switch_tools_off_but_not_bypass_approval(client):
    assert client.put("/api/work-agent/policy", json={"read_file": "deny"}).status_code == 200
    tools = {t["name"]: t["policy"] for t in client.get("/api/work-agent/config").json()["tools"]}
    assert tools["read_file"] == "deny"
    assert client.put("/api/work-agent/policy", json={"write_file": "allow"}).status_code == 422
    assert client.put("/api/work-agent/policy", json={"rm_rf": "allow"}).status_code == 422


def test_signed_webhooks_for_agents_and_schedules(client, receiver):
    r = client.post("/api/system/webhooks", json={"name": "n8n-tickets", "url": f"{receiver}/webhook/tickets",
                                                  "allow_agent": True})
    assert r.status_code == 201
    secret = r.json()["secret"]
    RECEIVED.clear()
    assert client.post("/api/system/webhooks/n8n-tickets/test").json()["ok"]
    got = RECEIVED[-1]
    assert got["headers"]["x-localailab-signature"] == sign(secret, got["body"])
    ev = _interactive(client, 'CALL call_webhook {"name": "n8n-tickets", "payload": {"title": "Printer broken"}}',
                      approve=True)
    tool = next(e for e in ev if e["type"] == "tool")
    assert tool["result"]["ok"] and json.loads(RECEIVED[-1]["body"])["payload"] == {"title": "Printer broken"}
    ev = _events(_run(client, 'CALL call_webhook {"name": "not-allowed", "payload": {}}').text)
    assert "error" in next(e for e in ev if e["type"] == "tool")["result"]  # only named, agent-enabled webhooks


def test_benchmark_schedule_gates_alerts_and_backup_schedule(client, receiver):
    client.post("/api/system/webhooks", json={"name": "alerts", "url": f"{receiver}/webhook/alerts"})
    r = client.post("/api/schedules", json={"name": "nightly quick", "kind": "benchmark", "every_hours": 24,
                                            "request": {"suite_id": "quick", "models": ["sim-small:1b"], "max_cases": 3},
                                            "gates": {"min_quality": 1.01}, "notify_webhook": "alerts"})
    assert r.status_code == 201, r.text
    sched = r.json()
    assert sched["next_run_at"]
    mgr = client.app.state.ctx.scheduler
    mgr.tick(now=utcnow() + timedelta(days=2))
    s = client.get("/api/schedules").json()["schedules"][0]
    assert s["last_status"] == "submitted" and s["last_run_id"]
    end = time.time() + 30
    while client.get(f"/api/benchmarks/runs/{s['last_run_id']}").json()["status"] != "completed" and time.time() < end:
        time.sleep(0.1)
    RECEIVED.clear()
    mgr.tick()
    s = client.get("/api/schedules").json()["schedules"][0]
    assert s["last_status"] == "failed" and "quality" in s["last_message"]
    assert RECEIVED and json.loads(RECEIVED[-1]["body"])["event"] == "schedule.alert"
    assert any(e["action"] == "schedule.alert" for e in client.get("/api/system/audit").json()["events"])

    b = client.post("/api/schedules", json={"name": "daily backup", "kind": "backup", "daily_at": "02:30",
                                            "request": {"keep": 2}}).json()
    out = client.post(f"/api/schedules/{b['id']}/run-now").json()
    assert out["status"] == "passed" and "written" in out["message"]
    bad = client.post("/api/schedules", json={"name": "x", "kind": "benchmark", "request": {"suite_id": "quick",
                                                                                          "models": ["sim-small:1b"]}})
    assert bad.status_code == 422  # needs every_hours or daily_at
