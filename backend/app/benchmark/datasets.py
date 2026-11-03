"""Benchmark suite loading and validation.

Suites are versioned JSON files in ``backend/datasets/benchmark``. The
``intent-classification`` suite is generated from the intent dataset's test
split so that LLMs and the scikit-learn baseline are compared on identical
examples.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from app.benchmark.evaluators import validate_spec
from app.core.errors import NotFoundError
from app.ml.intent_data import DATASET_VERSION, load_split


class BenchmarkCase(BaseModel):
    id: str
    category: str
    prompt: str
    reference: str = ""
    system: str | None = None
    max_tokens: int | None = Field(None, ge=1, le=8192)
    evaluator: dict

    @field_validator("evaluator")
    @classmethod
    def _check_evaluator(cls, v: dict) -> dict:
        validate_spec(v)
        return v


class BenchmarkSuite(BaseModel):
    id: str
    name: str
    version: str
    description: str = ""
    system: str | None = None
    default_max_tokens: int = 200
    cases: list[BenchmarkCase]
    sha256: str = ""
    source: str = "file"

    def summary(self) -> dict:
        cats: dict[str, int] = {}
        for c in self.cases:
            cats[c.category] = cats.get(c.category, 0) + 1
        return {"id": self.id, "name": self.name, "version": self.version, "description": self.description,
                "case_count": len(self.cases), "categories": cats, "sha256": self.sha256, "source": self.source}


INTENT_PROMPT = (
    "Classify the customer message into exactly one of these intents: {labels}.\n"
    "Message: \"{text}\"\n"
    "Answer with the intent label only."
)


SUITE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{1,60}$")
VERSION = re.compile(r"^\d+\.\d+\.\d+$")


def canonical_sha(data: dict) -> str:
    clean = {k: v for k, v in data.items() if k not in ("sha256", "source", "version")}
    return hashlib.sha256(json.dumps(clean, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def bump(version: str) -> str:
    major, minor, patch = (int(x) for x in version.split("."))
    return f"{major}.{minor}.{patch + 1}"


class SuiteRepository:
    """Built-in suites (JSON files) plus each user's own suites with their full version history."""

    def __init__(self, datasets_dir: Path, SessionLocal=None):
        self.datasets_dir = datasets_dir
        self.SessionLocal = SessionLocal

    def _load_files(self) -> dict[str, BenchmarkSuite]:
        suites: dict[str, BenchmarkSuite] = {}
        folder = self.datasets_dir / "benchmark"
        for path in sorted(folder.glob("*.json")):
            raw = path.read_bytes()
            suite = BenchmarkSuite(**json.loads(raw), sha256=hashlib.sha256(raw).hexdigest(), source=path.name)
            ids = [c.id for c in suite.cases]
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate case ids in {path.name}")
            suites[suite.id] = suite
        return suites

    def _intent_suite(self) -> BenchmarkSuite | None:
        try:
            split = load_split(self.datasets_dir)
        except FileNotFoundError:
            return None
        labels = ", ".join(split.labels)
        cases = [
            BenchmarkCase(
                id=f"intent-test-{i:03d}", category="intent_classification",
                prompt=INTENT_PROMPT.format(labels=labels, text=text.replace('"', "'")),
                reference=label, max_tokens=16,
                evaluator={"type": "label", "labels": split.labels, "answer": label},
            )
            for i, (text, label) in enumerate(split.test)
        ]
        digest = hashlib.sha256((split.sha256 + INTENT_PROMPT).encode()).hexdigest()
        return BenchmarkSuite(
            id="intent-classification", name="Intent Classification (LLM vs ML baseline)", version=DATASET_VERSION,
            description=f"Zero-shot LLM classification of the {len(cases)} held-out test messages of the intent "
                        "dataset – the same test split used by the scikit-learn baseline on the ML Lab page.",
            default_max_tokens=16, cases=cases, sha256=digest, source="generated:intent_v1.csv",
        )

    def builtin(self) -> dict[str, BenchmarkSuite]:
        suites = self._load_files()
        intent = self._intent_suite()
        if intent:
            suites[intent.id] = intent
        return suites

    def _rows(self, owner_id: int, suite_id: str | None = None):
        from sqlalchemy import select

        from app.models.benchmark import SuiteVersion

        if self.SessionLocal is None:
            return []
        with self.SessionLocal() as db:
            q = select(SuiteVersion).where(SuiteVersion.owner_id == owner_id, SuiteVersion.kind == "benchmark")
            if suite_id:
                q = q.where(SuiteVersion.suite_id == suite_id)
            return list(db.scalars(q.order_by(SuiteVersion.id)))

    @staticmethod
    def _from_row(row) -> BenchmarkSuite:
        return BenchmarkSuite(**{k: v for k, v in row.data.items() if k not in ("sha256", "source")},
                              sha256=row.sha256, source=f"custom v{row.version}")

    def user_suites(self, owner_id: int) -> dict[str, BenchmarkSuite]:
        latest: dict[str, object] = {}
        for row in self._rows(owner_id):
            latest[row.suite_id] = row
        return {sid: self._from_row(r) for sid, r in latest.items() if not r.deleted}

    def all(self, owner_id: int | None = None) -> dict[str, BenchmarkSuite]:
        suites = self.builtin()
        if owner_id is not None:
            suites.update(self.user_suites(owner_id))
        return suites

    def get(self, suite_id: str, owner_id: int | None = None, version: str | None = None) -> BenchmarkSuite:
        if version and owner_id is not None:
            row = next((r for r in self._rows(owner_id, suite_id) if r.version == version), None)
            if row is not None:
                return self._from_row(row)
        suite = self.all(owner_id).get(suite_id)
        if suite is None:
            raise NotFoundError(f"Benchmark suite '{suite_id}' not found.")
        return suite

    def versions(self, owner_id: int, suite_id: str) -> list[dict]:
        return [{"version": r.version, "sha256": r.sha256, "name": r.name, "cases": len(r.data.get("cases", [])),
                 "created_at": r.created_at.isoformat() + "Z", "note": r.note, "deleted": r.deleted}
                for r in self._rows(owner_id, suite_id)]

    def save(self, owner_id: int, data: dict, note: str | None = None) -> BenchmarkSuite:
        """Validate and store a new version (identical content = no new version)."""
        from pydantic import ValidationError

        from app.core.errors import ValidationFailed
        from app.models.benchmark import SuiteVersion

        if not isinstance(data, dict):
            raise ValidationFailed("The suite must be a JSON object.")
        sid = str(data.get("id", ""))
        if not SUITE_ID.match(sid):
            raise ValidationFailed("Suite id: lowercase letters, digits, '-' and '_' (2-61 characters).")
        if sid in self.builtin():
            raise ValidationFailed(f"'{sid}' is the id of a built-in suite. Choose another id.")
        try:
            suite = BenchmarkSuite(**{k: v for k, v in data.items() if k not in ("sha256", "source")})
        except ValidationError as exc:
            msg = "; ".join(f"{'.'.join(str(x) for x in e['loc'])}: {e['msg']}" for e in exc.errors()[:4])
            raise ValidationFailed(f"Invalid benchmark suite – {msg}") from exc
        ids = [c.id for c in suite.cases]
        if len(ids) != len(set(ids)):
            raise ValidationFailed("Case ids must be unique.")
        payload = suite.model_dump(mode="json", exclude={"sha256", "source"})
        sha = canonical_sha(payload)
        rows = self._rows(owner_id, sid)
        last = rows[-1] if rows else None
        if last is not None and last.sha256 == sha and not last.deleted:
            return self._from_row(last)
        version = suite.version if VERSION.match(suite.version) else "1.0.0"
        used = {r.version for r in rows}
        if version in used:
            version = bump(max(used, key=lambda v: tuple(int(x) for x in v.split(".")) if VERSION.match(v) else (0,)))
        payload["version"] = version
        with self.SessionLocal() as db:
            db.add(SuiteVersion(owner_id=owner_id, kind="benchmark", suite_id=sid, version=version, name=suite.name,
                                sha256=sha, data=payload, note=(note or "")[:300] or None))
            db.commit()
        return self.get(sid, owner_id)

    def delete(self, owner_id: int, suite_id: str) -> None:
        """Hide a user suite. Its versions stay in the history (runs keep their own snapshots anyway)."""
        from app.models.benchmark import SuiteVersion

        rows = self._rows(owner_id, suite_id)
        if not rows:
            raise NotFoundError("Custom suite not found (built-in suites cannot be deleted).")
        last = rows[-1]
        with self.SessionLocal() as db:
            db.add(SuiteVersion(owner_id=owner_id, kind="benchmark", suite_id=suite_id, version=last.version + "-deleted",
                                name=last.name, sha256=last.sha256, data=last.data, deleted=True, note="deleted"))
            db.commit()
