"""Scheduled evaluations and backups.

A schedule submits a benchmark or Agent Arena run (or writes a backup) every N hours or daily at a fixed time.
When a scheduled run finishes its gates are checked (e.g. minimum quality, maximum safety-violation rate) and it
is compared with the previous run of the same schedule; failures and regressions are written to the audit log
and, if configured, sent to a webhook (e.g. an n8n workflow that posts to chat or opens a ticket).

Runs still go through the shared job lock, so a schedule never measures at the same time as another run.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta

from sqlalchemy import select

from app.core.audit import record
from app.core.errors import AppError
from app.db.base import utcnow
from app.models.automation import Schedule

log = logging.getLogger("app.scheduler")
POLL_S = 20


def next_time(s: Schedule, after: datetime) -> datetime | None:
    """Next run time (UTC, naive) after ``after``."""
    if s.every_hours:
        return after + timedelta(hours=int(s.every_hours))
    if s.daily_at:
        hh, mm = (int(x) for x in s.daily_at.split(":"))
        offset = datetime.now() - utcnow()  # server local time vs the naive-UTC timestamps stored in the database
        local_after = after + offset
        cand = local_after.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if cand <= local_after:
            cand += timedelta(days=1)
        return cand - offset
    return None


class ScheduleManager:
    def __init__(self, settings, SessionLocal, benchmarks, agents, engine):
        self.settings = settings
        self.SessionLocal = SessionLocal
        self.benchmarks = benchmarks
        self.agents = agents
        self.engine = engine
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True, name="scheduler")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.wait(POLL_S):
            try:
                self.tick()
            except Exception:  # noqa: BLE001
                log.exception("scheduler tick failed")

    # ------------------------------------------------------------ one pass (also called by tests)
    def tick(self, now: datetime | None = None) -> None:
        now = now or utcnow()
        with self.SessionLocal() as db:
            due = [s.id for s in db.scalars(select(Schedule).where(Schedule.enabled.is_(True)))
                   if s.next_run_at is not None and s.next_run_at <= now]
            pending = [s.id for s in db.scalars(select(Schedule).where(Schedule.last_status == "submitted"))]
        for sid in pending:
            self.evaluate(sid)
        for sid in due:
            self.run_now(sid, now)

    def run_now(self, schedule_id: int, now: datetime | None = None) -> dict:
        now = now or utcnow()
        with self.SessionLocal() as db:
            s = db.get(Schedule, schedule_id)
            if s is None:
                return {}
            kind, req, owner = s.kind, dict(s.request or {}), s.owner_id
        status, message, run_id = "submitted", None, None
        try:
            if kind == "benchmark":
                from app.benchmark.runner import BenchmarkRequest

                run_id = self.benchmarks.submit(owner, BenchmarkRequest(**req)).id
            elif kind == "agent":
                from app.agents.runner import AgentRunRequest

                run_id = self.agents.submit(owner, AgentRunRequest(**req)).id
            elif kind == "backup":
                from app.core.backup import create_backup, prune_backups
                from app.db.migrate import current_revision

                path = create_backup(self.settings.data_dir, self.settings.backups_dir,
                                     schema_revision=current_revision(self.engine), label="scheduled",
                                     include_mlflow=req.get("include_mlflow", True))
                removed = prune_backups(self.settings.backups_dir, int(req.get("keep", 7)))
                status, message = "passed", f"{path.name} written" + (f", {len(removed)} old removed" if removed else "")
        except AppError as exc:
            status, message = "error", exc.message
        except Exception as exc:  # noqa: BLE001
            status, message = "error", f"{type(exc).__name__}: {str(exc)[:300]}"
        with self.SessionLocal() as db:
            s = db.get(Schedule, schedule_id)
            s.last_run_at, s.last_run_id, s.last_status, s.last_message = now, run_id, status, message
            s.next_run_at = next_time(s, now) if s.enabled else None
            db.commit()
        if status == "error":
            self._alert(schedule_id, f"could not start: {message}")
        return {"status": status, "run_id": run_id, "message": message}

    def evaluate(self, schedule_id: int) -> None:
        """Once the submitted run has finished: gates + comparison with the schedule's previous run."""
        from app.agents.report import summarize as agent_summary
        from app.benchmark.aggregate import summarize as bench_summary
        from app.models.agents import AgentResult, AgentRun
        from app.models.benchmark import BenchmarkResult, BenchmarkRun

        with self.SessionLocal() as db:
            s = db.get(Schedule, schedule_id)
            if s is None or s.last_run_id is None:
                return
            run_cls, res_cls, summ = ((BenchmarkRun, BenchmarkResult, bench_summary) if s.kind == "benchmark"
                                      else (AgentRun, AgentResult, agent_summary))
            run = db.get(run_cls, s.last_run_id)
            if run is None or run.status in ("queued", "running"):
                return
            if run.status != "completed":
                s.last_status, s.last_message = "error", f"run {run.id} {run.status}: {run.error or ''}"[:500]
                db.commit()
                self._alert(s.id, s.last_message)
                return
            summary = summ(list(db.scalars(select(res_cls).where(res_cls.run_id == run.id))))
            gates = s.gates or {}
            fails: list[str] = []
            for m in summary:
                if s.kind == "benchmark":
                    q, comp = m.get("effective_quality"), m.get("completion_rate")
                    if gates.get("min_quality") is not None and (q or 0) < gates["min_quality"]:
                        fails.append(f"{m['model']}: quality {q} < {gates['min_quality']}")
                    if gates.get("min_completion") is not None and (comp or 0) < gates["min_completion"]:
                        fails.append(f"{m['model']}: completion {comp} < {gates['min_completion']}")
                else:
                    sr, vr = m.get("success_rate"), m.get("safety_violation_rate")
                    if gates.get("min_success") is not None and (sr or 0) < gates["min_success"]:
                        fails.append(f"{m['model']}: success {sr} < {gates['min_success']}")
                    if gates.get("max_violation_rate") is not None and (vr or 0) > gates["max_violation_rate"]:
                        fails.append(f"{m['model']}: safety violations {vr} > {gates['max_violation_rate']}")
            prev = db.scalars(select(run_cls).where(run_cls.owner_id == s.owner_id, run_cls.id < run.id,
                                                    run_cls.status == "completed",
                                                    run_cls.name == run.name).order_by(run_cls.id.desc())).first()
            drop = float(gates.get("max_drop", 0.05))
            if prev is not None:
                before = {m["model"]: m for m in summ(list(db.scalars(select(res_cls).where(res_cls.run_id == prev.id))))}
                key = "effective_quality" if s.kind == "benchmark" else "success_rate"
                for m in summary:
                    b = before.get(m["model"])
                    if b and b.get(key) is not None and m.get(key) is not None and b[key] - m[key] > drop:
                        fails.append(f"{m['model']}: {key} fell from {b[key]} to {m[key]} (run {prev.id} → {run.id})")
            s.last_status = "failed" if fails else "passed"
            s.last_message = "; ".join(fails)[:1000] if fails else f"run {run.id}: all gates passed"
            db.commit()
            sid, msg, failed = s.id, s.last_message, bool(fails)
        if failed:
            self._alert(sid, msg)

    def _alert(self, schedule_id: int, message: str) -> None:
        with self.SessionLocal() as db:
            s = db.get(Schedule, schedule_id)
            if s is None:
                return
            name, hook, owner = s.name, s.notify_webhook, s.owner_id
        record(self.SessionLocal, user_id=owner, username=None, action="schedule.alert", target=name,
               details={"message": message}, outcome="failed")
        if hook:
            from app.services import webhooks

            try:
                webhooks.send(webhooks.get(self.SessionLocal, hook), "schedule.alert",
                              {"schedule": name, "message": message})
            except AppError as exc:
                log.warning("schedule alert webhook failed", extra={"error": exc.message})
