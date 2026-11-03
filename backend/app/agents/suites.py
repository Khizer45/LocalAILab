"""Agent benchmark suites: tools + tasks + expectations, versioned and hashed.

Built-in suites live in ``backend/datasets/agent/*.json``. Users can add their
own suites (same JSON format) from the Agent Arena page; those are stored per
user in the database. See ``datasets/agent/README.md`` for the format.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from app.agents.tools import ToolSpec
from app.benchmark.evaluators import EVALUATORS, validate_spec
from app.core.errors import NotFoundError, ValidationFailed
from app.models.agents import AgentSuiteRecord
from app.models.benchmark import SuiteVersion

SUITE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{1,60}$")


class ExpectedCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    optional: bool = False
    example: dict[str, Any] | None = None  # concrete arguments (documentation / simulator)


class ForbiddenCall(BaseModel):
    """A call that must not happen. Empty ``arguments`` = any call of that tool."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    severity: Literal["severe", "minor"] = "severe"
    reason: str = Field("", max_length=300)


class Expect(BaseModel):
    tool_calls: list[ExpectedCall] = Field(default_factory=list)
    # other acceptable ways to solve the task (each a full list of calls); the best-matching path is scored
    alternatives: list[list[ExpectedCall]] = Field(default_factory=list, max_length=10)
    order: Literal["any", "strict"] = "any"
    forbidden_tools: list[str] = Field(default_factory=list)  # any call = severe violation
    forbidden_calls: list[ForbiddenCall] = Field(default_factory=list)
    # risky tools (send_email, delete_file…) that no accepted path uses count as severe violations unless allowed
    allow_unrequested_risky: bool = False
    no_tools: bool = False
    max_tool_calls: int | None = Field(None, ge=0, le=50)
    final_answer: dict | None = None

    @field_validator("final_answer")
    @classmethod
    def _evaluator(cls, v: dict | None) -> dict | None:
        if v is None:
            return v
        try:
            validate_spec(v)
        except ValueError as exc:
            raise ValueError(f"{exc} (evaluators: {', '.join(EVALUATORS)})") from exc
        return v

    @model_validator(mode="after")
    def _consistent(self) -> Expect:
        if self.no_tools and (self.tool_calls or any(self.alternatives)):
            raise ValueError("a task cannot expect tool calls and no_tools at the same time")
        if self.alternatives and not self.tool_calls:
            raise ValueError("alternatives need a main tool_calls path")
        return self

    def paths(self) -> list[list[ExpectedCall]]:
        return [self.tool_calls] + [p for p in self.alternatives if p]

    def has_safety_rules(self) -> bool:
        return bool(self.forbidden_tools or self.forbidden_calls)


class AgentTask(BaseModel):
    id: str = Field(min_length=1, max_length=100)
    category: str = Field("general", max_length=50)
    prompt: str = Field(min_length=1, max_length=8000)
    system: str | None = Field(None, max_length=8000)
    tools: list[str] | None = None  # subset of the suite's tools; None = all
    max_steps: int | None = Field(None, ge=1, le=20)
    expect: Expect = Field(default_factory=Expect)
    reference: str = Field("", max_length=4000)  # an ideal final answer (documentation / simulator)


class AgentSuite(BaseModel):
    id: str
    name: str = Field(min_length=1, max_length=200)
    version: str = Field("1.0.0", max_length=32)
    description: str = Field("", max_length=4000)
    system: str | None = Field(None, max_length=8000)
    tools: list[ToolSpec] = Field(default_factory=list, max_length=50)
    tasks: list[AgentTask] = Field(min_length=1, max_length=500)
    sha256: str = ""
    source: str = "file"

    @field_validator("id")
    @classmethod
    def _id(cls, v: str) -> str:
        if not SUITE_ID.match(v):
            raise ValueError("suite id: lowercase letters, digits, '-' and '_' (2-61 characters)")
        return v

    @model_validator(mode="after")
    def _check(self) -> AgentSuite:
        names = [t.name for t in self.tools]
        if len(names) != len(set(names)):
            raise ValueError("tool names must be unique")
        ids = [t.id for t in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("task ids must be unique")
        known = set(names)
        for t in self.tasks:
            avail = set(t.tools) if t.tools is not None else known
            if not avail <= known:
                raise ValueError(f"task {t.id}: unknown tools {sorted(avail - known)}")
            for path in t.expect.paths():
                for c in path:
                    if c.name not in avail:
                        raise ValueError(f"task {t.id}: expected tool '{c.name}' is not available to the task")
            for f in t.expect.forbidden_tools + [fc.name for fc in t.expect.forbidden_calls]:
                if f not in known:
                    raise ValueError(f"task {t.id}: forbidden tool '{f}' is not defined in the suite")
        return self

    def tools_for(self, task: AgentTask) -> list[ToolSpec]:
        if task.tools is None:
            return list(self.tools)
        wanted = set(task.tools)
        return [t for t in self.tools if t.name in wanted]

    def summary(self) -> dict:
        cats: dict[str, int] = {}
        for t in self.tasks:
            cats[t.category] = cats.get(t.category, 0) + 1
        return {"id": self.id, "name": self.name, "version": self.version, "description": self.description,
                "task_count": len(self.tasks), "tool_count": len(self.tools), "categories": cats,
                "tools": [t.name for t in self.tools], "sha256": self.sha256, "source": self.source}


def canonical_hash(data: dict) -> str:
    clean = {k: v for k, v in data.items() if k not in ("sha256", "source")}
    return hashlib.sha256(json.dumps(clean, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def parse_suite(data: Any, source: str) -> AgentSuite:
    if not isinstance(data, dict):
        raise ValidationFailed("The suite must be a JSON object.")
    try:
        suite = AgentSuite(**{k: v for k, v in data.items() if k not in ("sha256", "source")})
    except ValidationError as exc:
        first = exc.errors()[:4]
        msg = "; ".join(f"{'.'.join(str(x) for x in e['loc'])}: {e['msg']}" for e in first)
        raise ValidationFailed(f"Invalid agent suite – {msg}") from exc
    suite.sha256 = canonical_hash(suite.model_dump(mode="json"))
    suite.source = source
    return suite


class AgentSuiteRepository:
    def __init__(self, datasets_dir: Path, SessionLocal: sessionmaker[Session]):
        self.folder = datasets_dir / "agent"
        self.SessionLocal = SessionLocal

    def builtin(self) -> dict[str, AgentSuite]:
        out: dict[str, AgentSuite] = {}
        for path in sorted(self.folder.glob("*.json")):
            suite = parse_suite(json.loads(path.read_text(encoding="utf-8")), f"built-in:{path.name}")
            out[suite.id] = suite
        return out

    def user_suites(self, owner_id: int) -> dict[str, AgentSuite]:
        with self.SessionLocal() as db:
            rows = db.scalars(select(AgentSuiteRecord).where(AgentSuiteRecord.owner_id == owner_id)).all()
            return {r.suite_id: parse_suite(r.data, "custom") for r in rows}

    def all(self, owner_id: int) -> dict[str, AgentSuite]:
        return {**self.builtin(), **self.user_suites(owner_id)}

    def get(self, owner_id: int, suite_id: str) -> AgentSuite:
        suite = self.all(owner_id).get(suite_id)
        if suite is None:
            raise NotFoundError(f"Agent suite '{suite_id}' not found.")
        return suite

    def save(self, owner_id: int, data: dict) -> AgentSuite:
        suite = parse_suite(data, "custom")
        if suite.id in self.builtin():
            raise ValidationFailed(f"'{suite.id}' is the id of a built-in suite. Choose another id.")
        payload = suite.model_dump(mode="json", exclude={"sha256", "source"})
        with self.SessionLocal() as db:
            row = db.scalar(select(AgentSuiteRecord).where(AgentSuiteRecord.owner_id == owner_id,
                                                           AgentSuiteRecord.suite_id == suite.id))
            changed = row is None or row.sha256 != suite.sha256
            if row is None:
                row = AgentSuiteRecord(owner_id=owner_id, suite_id=suite.id, name=suite.name, version=suite.version,
                                       sha256=suite.sha256, data=payload)
                db.add(row)
            else:
                row.name, row.version, row.sha256, row.data = suite.name, suite.version, suite.sha256, payload
            if changed:  # keep every distinct version (runs also keep their own snapshot)
                n = db.scalar(select(func.count(SuiteVersion.id)).where(
                    SuiteVersion.owner_id == owner_id, SuiteVersion.kind == "agent",
                    SuiteVersion.suite_id == suite.id)) or 0
                db.add(SuiteVersion(owner_id=owner_id, kind="agent", suite_id=suite.id, version=f"{suite.version}#{n + 1}",
                                    name=suite.name, sha256=suite.sha256, data=payload))
            db.commit()
        return suite

    def versions(self, owner_id: int, suite_id: str) -> list[dict]:
        with self.SessionLocal() as db:
            rows = db.scalars(select(SuiteVersion).where(SuiteVersion.owner_id == owner_id, SuiteVersion.kind == "agent",
                                                         SuiteVersion.suite_id == suite_id).order_by(SuiteVersion.id))
            return [{"version": r.version, "sha256": r.sha256, "tasks": len(r.data.get("tasks", [])),
                     "created_at": r.created_at.isoformat() + "Z"} for r in rows]

    def delete(self, owner_id: int, suite_id: str) -> None:
        with self.SessionLocal() as db:
            row = db.scalar(select(AgentSuiteRecord).where(AgentSuiteRecord.owner_id == owner_id,
                                                           AgentSuiteRecord.suite_id == suite_id))
            if row is None:
                raise NotFoundError("Custom suite not found (built-in suites cannot be deleted).")
            db.delete(row)
            db.commit()
