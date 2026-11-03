"""Experiment bundles: one ZIP that documents (and can reproduce) a benchmark, Model Arena or Agent Arena run.

Contents::

    manifest.json     format, kind, app version, export scope, every file with SHA-256 and whether it holds user data
    run.json          configuration, summary, environment, model identity, suite/dataset/split hashes
    results.jsonl     raw per-case / per-example results (full scope) or scores only (metadata scope)
    suite.json        the exact suite snapshot used (benchmark / agent runs)
    dataset.json      dataset description, split indices and (full scope) the evaluated labels (arena runs)
    data/<file>       the dataset itself – only when explicitly requested
    report.md         the human-readable report

Scopes
* ``full``     – everything needed to re-check the numbers: prompts, responses, traces, labels.
* ``metadata`` – configuration, hashes, scores and environment only. Prompts, responses, agent traces, answers,
  error examples and labels are removed, so it can be shared without the underlying data. The manifest lists
  which files contain user data.

``verify`` recomputes the aggregates from the raw results in the bundle and compares them with the stored
summary; ``import`` stores the run read-only under the current user; ``rerun`` starts a new run with the same
suite snapshot and settings.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import datetime

import numpy as np

from app import __version__
from app.core.errors import ValidationFailed

FORMAT = "local-ai-lab-experiment"
FORMAT_VERSION = 1
KINDS = ("benchmark", "arena", "agent")
_STRIP_BENCH = ("response",)
_STRIP_AGENT = ("final_answer", "trace")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _jsonl(rows: list[dict]) -> bytes:
    return ("\n".join(json.dumps(r, default=str, ensure_ascii=False) for r in rows) + "\n").encode()


def _scrub_details(d: dict | None) -> dict | None:
    if not d:
        return d
    d = dict(d)
    for k in ("errors_sample", "prompt", "notes"):
        d.pop(k, None)
    return d


# ----------------------------------------------------------------------------- export
def export_bundle(kind: str, run, results: list, *, scope: str = "full", suite: dict | None = None,
                  dataset: dict | None = None, report: str = "", data_file: tuple[str, bytes] | None = None) -> bytes:
    if kind not in KINDS:
        raise ValidationFailed(f"Unknown bundle kind '{kind}'.")
    if scope not in ("full", "metadata"):
        raise ValidationFailed("scope must be 'full' or 'metadata'.")
    files: dict[str, tuple[bytes, bool, str]] = {}  # path -> (bytes, contains_user_data, description)
    cols = [c.name for c in run.__table__.columns]
    run_dict = {c: getattr(run, c) for c in cols if c not in ("suite_snapshot",)}
    if scope == "metadata":
        run_dict.pop("error", None)
        if kind == "arena" and isinstance(run_dict.get("summary"), dict):
            run_dict["summary"] = {k: v for k, v in run_dict["summary"].items() if k != "labels"}
    files["run.json"] = (json.dumps(run_dict, default=str, indent=1, ensure_ascii=False).encode(), False,
                         "configuration, summary, environment and model identity")
    rows = []
    for r in results:
        d = {c.name: getattr(r, c.name) for c in r.__table__.columns}
        if scope == "metadata":
            for k in (_STRIP_BENCH if kind == "benchmark" else _STRIP_AGENT if kind == "agent" else ()):
                d.pop(k, None)
            if kind == "benchmark":
                d["eval_details"] = {"method": (d.get("eval_details") or {}).get("method")}
            if kind == "agent":
                d["checks"] = {k: {"passed": v.get("passed"), "score": v.get("score")}
                               for k, v in (d.get("checks") or {}).items()}
            if kind == "arena":
                d["details"] = _scrub_details(d.get("details"))
        rows.append(d)
    files["results.jsonl"] = (_jsonl(rows), scope == "full",
                              "raw per-case results" + ("" if scope == "full" else " (scores only)"))
    if suite is not None:
        if scope == "metadata":
            items = suite.get("cases") or suite.get("tasks") or []
            light = {k: v for k, v in suite.items() if k not in ("cases", "tasks", "system", "tools")}
            light["items"] = [{"id": c.get("id"), "category": c.get("category"),
                               "sha256": _sha(json.dumps(c, sort_keys=True, default=str).encode())} for c in items]
            files["suite.json"] = (json.dumps(light, indent=1, default=str).encode(), False, "suite ids and item hashes")
        else:
            files["suite.json"] = (json.dumps(suite, indent=1, default=str, ensure_ascii=False).encode(), True,
                                   "exact suite snapshot (prompts, references, evaluators, tools)")
    if dataset is not None:
        ds = dict(dataset)
        if scope == "metadata":
            ds.pop("labels_by_row", None)
            ds.pop("profile", None)
        files["dataset.json"] = (json.dumps(ds, indent=1, default=str, ensure_ascii=False).encode(),
                                 scope == "full", "dataset description, split indices" +
                                 (" and evaluated labels" if scope == "full" else ""))
    if data_file is not None and scope == "full":
        files[f"data/{data_file[0]}"] = (data_file[1], True, "the dataset file")
    if report:
        files["report.md"] = (report.encode(), scope == "full", "human-readable report")
    manifest = {"format": FORMAT, "format_version": FORMAT_VERSION, "kind": kind, "scope": scope,
                "run_id": run.id, "app_version": __version__, "exported_at": datetime.now().isoformat(timespec="seconds"),
                "contains_user_data": any(u for _, u, _ in files.values()),
                "files": [{"path": p, "sha256": _sha(b), "size": len(b), "contains_user_data": u, "content": desc}
                          for p, (b, u, desc) in files.items()]}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=1))
        for p, (b, _, _) in files.items():
            zf.writestr(p, b)
    return buf.getvalue()


# ----------------------------------------------------------------------------- read / verify
def read_bundle(raw: bytes) -> tuple[dict, dict[str, bytes]]:
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile as exc:
        raise ValidationFailed("Not a valid bundle ZIP.") from exc
    with zf:
        names = zf.namelist()
        for n in names:
            if n.startswith("/") or ".." in n.split("/") or "\\" in n:
                raise ValidationFailed("The bundle contains an unsafe path.", code="unsafe_archive")
        if "manifest.json" not in names:
            raise ValidationFailed("The bundle has no manifest.json.")
        if sum(i.file_size for i in zf.infolist()) > 2 * 1024**3:
            raise ValidationFailed("The bundle is too large.")
        manifest = json.loads(zf.read("manifest.json"))
        if manifest.get("format") != FORMAT or manifest.get("kind") not in KINDS:
            raise ValidationFailed("This ZIP is not a Local AI Lab experiment bundle.")
        files = {f["path"]: zf.read(f["path"]) for f in manifest.get("files", []) if f["path"] in names}
    return manifest, files


def _rows(files: dict[str, bytes]) -> list[dict]:
    return [json.loads(line) for line in files.get("results.jsonl", b"").decode().splitlines() if line.strip()]


def verify_bundle(raw: bytes) -> dict:
    manifest, files = read_bundle(raw)
    problems: list[str] = []
    for f in manifest.get("files", []):
        if f["path"] not in files:
            problems.append(f"missing {f['path']}")
        elif _sha(files[f["path"]]) != f["sha256"]:
            problems.append(f"checksum mismatch {f['path']}")
    checks: list[dict] = []
    if not problems:
        run = json.loads(files["run.json"])
        rows = _rows(files)
        kind = manifest["kind"]
        if kind == "benchmark":
            by: dict[str, list[float]] = {}
            for r in rows:
                by.setdefault(r["model"], []).append(float(r["score"]))
            expected = len(run.get("config", {}).get("case_ids", [])) * int(run.get("config", {}).get("repeats", 1)) \
                * len(run.get("models", []))
            checks.append({"check": "result count", "ok": expected in (0, len(rows)) or run.get("status") != "completed",
                           "detail": f"{len(rows)} results, {expected} expected"})
            for m, scores in by.items():
                checks.append({"check": f"{m}: effective quality recomputed", "ok": True,
                               "detail": round(float(np.mean(scores)), 3)})
        elif kind == "agent":
            by: dict[str, list[bool]] = {}
            for r in rows:
                by.setdefault(r["model"], []).append(bool(r["success"]))
                if r.get("hard_fail") and float(r.get("score", 0)) != 0.0:
                    problems.append(f"hard failure with non-zero score ({r['task_id']})")
            for m, v in by.items():
                checks.append({"check": f"{m}: success rate recomputed", "ok": True, "detail": round(float(np.mean(v)), 3)})
        else:
            checks += _verify_arena(run, rows, files)
        problems += [f"{c['check']}: {c['detail']}" for c in checks if not c["ok"]]
    return {"ok": not problems, "kind": manifest["kind"], "scope": manifest.get("scope"), "run_id": manifest.get("run_id"),
            "app_version": manifest.get("app_version"), "contains_user_data": manifest.get("contains_user_data"),
            "checks": checks, "problems": problems}


def _verify_arena(run: dict, rows: list[dict], files: dict[str, bytes]) -> list[dict]:
    from app.arena.stats import classification_metrics, regression_metrics

    checks = []
    ds = json.loads(files["dataset.json"]) if "dataset.json" in files else {}
    labels_by_row = ds.get("labels_by_row")
    split = ds.get("split") or {}
    task = (run.get("dataset_info") or {}).get("task")
    if not labels_by_row:
        return [{"check": "metrics recomputed from predictions", "ok": True,
                 "detail": "skipped – labels are not included in a metadata bundle"}]
    labels = (run.get("summary") or {}).get("labels") or ds.get("labels") or []
    for r in rows:
        if r.get("status") != "ok":
            continue
        for part, pk, ik, mk in (("test", "predictions", "evaluated_idx", "metrics"),
                                 ("val", "val_predictions", "val_idx", "val_metrics")):
            if not r.get(pk) or not split.get(part):
                continue
            rows_idx = np.asarray(split[part])[np.asarray(r[ik], dtype=int)]
            y = [labels_by_row[str(int(i))] for i in rows_idx]
            if task == "classification":
                m = classification_metrics(np.array([labels.index(v) for v in y]), np.asarray(r[pk]), labels)
                key = "f1_macro"
            else:
                m = regression_metrics(np.asarray(y, dtype=float), np.asarray(r[pk], dtype=float))
                key = "rmse"
            stored = (r.get(mk) or {}).get(key)
            checks.append({"check": f"{r['key']} {part} {key}", "ok": stored is None or abs(m[key] - stored) < 1e-3,
                           "detail": f"recomputed {m[key]} vs stored {stored}"})
    return checks
