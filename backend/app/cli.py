"""Local AI Lab command line (run from the ``backend`` folder: ``python -m app.cli --help``; Windows:
``scripts\\lal.bat``).

Local commands work on the data folder directly (the server does not need to run):

    python -m app.cli db upgrade | db current
    python -m app.cli backup create [--label nightly] | backup list | backup verify FILE | backup restore FILE --yes
    python -m app.cli evaluators agreement
    python -m app.cli bundle verify FILE

Server commands talk to a running Local AI Lab (``--server``, default http://127.0.0.1:8000; token from
``--token`` or the ``LAL_TOKEN`` environment variable when authentication is on) and can act as **CI gates** –
the exit code is 0 when every gate passes, 1 when a gate fails and 2 on errors:

    python -m app.cli bench run --suite quick --model qwen3:1.7b --gate-quality 0.7 --gate-completion 0.95
    python -m app.cli agents run --suite agent-safety --model qwen3:4b --gate-max-violation-rate 0
    python -m app.cli arena run --dataset 3 --algorithm logistic_regression --gate-final 0.8
    python -m app.cli bundle export --kind benchmark --run 12 --scope metadata -o run12.zip
    python -m app.cli setup check
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

EXIT_OK, EXIT_GATE, EXIT_ERROR = 0, 1, 2


def _settings():
    from app.core.config import Settings

    return Settings()


def _print(obj, as_json: bool) -> None:
    if as_json:
        print(json.dumps(obj, indent=2, default=str))


# ----------------------------------------------------------------------------- local commands
def cmd_db(args) -> int:
    from app.db.migrate import current_revision, head_revision, upgrade_database
    from app.db.session import make_engine

    s = _settings()
    s.ensure_dirs()
    engine = make_engine(s.sqlalchemy_url)
    try:  # dispose: an open SQLite handle would block a later restore on Windows
        if args.action == "current":
            print(f"database revision: {current_revision(engine)} (latest: {head_revision()})")
            return EXIT_OK
        res = upgrade_database(engine, s.backups_dir)
    finally:
        engine.dispose()
    print(f"upgraded {res['from'] or 'empty/legacy'} -> {res['to']}" + (f"; backup: {res['backup']}" if res["backup"] else ""))
    return EXIT_OK


def _server_running(url: str) -> bool:
    import httpx

    try:
        return httpx.get(url.rstrip("/") + "/api/health", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


def cmd_backup(args) -> int:
    from app.core import backup as B
    from app.db.migrate import current_revision
    from app.db.session import make_engine

    s = _settings()
    if args.action == "create":
        rev = None
        if (s.data_dir / B.DB_NAME).exists():
            engine = make_engine(s.sqlalchemy_url)
            try:
                rev = current_revision(engine)
            finally:
                engine.dispose()
        path = B.create_backup(s.data_dir, Path(args.output) if args.output else s.backups_dir, schema_revision=rev,
                               include_mlflow=not args.no_mlflow, include_backbones=args.backbones, label=args.label)
        info = B.verify_backup(path)
        print(f"backup written: {path} ({info['files']} files, {info['bytes'] / 1024 / 1024:.1f} MB, verified)")
        if args.keep:
            removed = B.prune_backups(path.parent, args.keep)
            if removed:
                print(f"removed {len(removed)} older backup(s)")
        return EXIT_OK
    if args.action == "list":
        for b in B.list_backups(s.backups_dir):
            print(f"{b['created_at']}  {b['size_bytes'] / 1024 / 1024:8.1f} MB  {b['name']}")
        return EXIT_OK
    if args.action == "verify":
        print(json.dumps(B.verify_backup(Path(args.file)), indent=2))
        return EXIT_OK
    if args.action == "restore":
        if not args.yes:
            print("Restoring replaces the current data (it is moved to data/backups/replaced-<time>/). Add --yes.")
            return EXIT_ERROR
        if _server_running(args.server) and not args.force:
            print("Local AI Lab seems to be running. Stop it first (or use Settings → Backup & restore in the app, "
                  "which applies the restore at the next start).")
            return EXIT_ERROR
        res = B.restore_backup(Path(args.file), s.data_dir)
        print(f"restored {res['restored_from']}; previous data moved to {res['previous_data_moved_to']}")
        return EXIT_OK
    return EXIT_ERROR


def cmd_evaluators(args) -> int:
    from app.benchmark.calibration import agreement

    s = _settings()
    rep = agreement(s.datasets_dir, Path(args.file) if args.file else None)
    if args.json:
        _print(rep, True)
        return EXIT_OK
    o = rep["overall"]
    print(f"items: {o['n']}  agreement: {o['agreement']:.1%}  kappa: {o['kappa']}  false passes: {o['false_pass']}  "
          f"false fails: {o['false_fail']}")
    for t, r in rep["by_evaluator"].items():
        print(f"  {t:14s} n={r['n']:3d}  agreement={r['agreement']:.0%}  false pass={r['false_pass']}  "
              f"false fail={r['false_fail']}")
    if rep["draft_labels"]:
        print(f"note: {rep['draft_labels']} judgements are drafts – see datasets/evaluator_calibration/README.md")
    return EXIT_OK


# ----------------------------------------------------------------------------- server client
class Api:
    def __init__(self, server: str, token: str | None):
        import httpx

        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.http = httpx.Client(base_url=server.rstrip("/"), headers=headers, timeout=60)

    def call(self, method: str, path: str, **kw):
        r = self.http.request(method, path, **kw)
        if r.status_code >= 400:
            try:
                err = r.json().get("error", {})
                msg = err.get("message") or r.text
                hint = err.get("hint")
            except ValueError:
                msg, hint = r.text, None
            raise SystemExit(f"error {r.status_code}: {msg}" + (f"\nhint: {hint}" if hint else ""))
        return r

    def wait(self, path: str, timeout_s: int) -> dict:
        end = time.time() + timeout_s
        last = None
        while time.time() < end:
            run = self.call("GET", path).json()
            prog = f"{run.get('progress_done')}/{run.get('progress_total')}"
            if prog != last:
                print(f"  {run['status']} {prog} {run.get('current_item') or ''}".rstrip(), file=sys.stderr)
                last = prog
            if run["status"] not in ("queued", "running"):
                return run
            time.sleep(2)
        raise SystemExit(f"timed out after {timeout_s}s waiting for {path}")


def _gates(checks: list[tuple[str, float | None, float, str]]) -> tuple[int, list[str]]:
    """(name, value, threshold, op) → exit code and printable lines."""
    lines, failed = [], False
    for name, value, threshold, op in checks:
        ok = value is not None and (value >= threshold if op == ">=" else value <= threshold)
        failed |= not ok
        lines.append(f"{'PASS' if ok else 'FAIL'}  {name} = {value} (gate {op} {threshold})")
    return (EXIT_GATE if failed else EXIT_OK), lines


def cmd_bench(args) -> int:
    api = Api(args.server, args.token)
    body = {"suite_id": args.suite, "models": args.model, "repeats": args.repeats, "warmup": not args.no_warmup}
    if args.max_cases:
        body["max_cases"] = args.max_cases
    run = api.call("POST", "/api/benchmarks/runs", json=body).json()
    print(f"benchmark run {run['id']} started", file=sys.stderr)
    run = api.wait(f"/api/benchmarks/runs/{run['id']}?include_responses=false", args.timeout)
    if args.json:
        _print({k: run.get(k) for k in ("id", "status", "summary", "comparison", "integrity", "simulated")}, True)
    if run["status"] != "completed":
        print(f"run {run['status']}: {run.get('error') or ''}")
        return EXIT_ERROR
    checks = []
    for s in run["summary"]:
        print(f"{s['model']}: effective quality {s['effective_quality']} (CI {s.get('effective_quality_ci')}), "
              f"completion {s['completion_rate']}, pass rate {s['pass_rate']}, latency p50 {s['latency_ms_p50']} ms")
        if args.gate_quality is not None:
            checks.append((f"{s['model']} effective quality", s["effective_quality"], args.gate_quality, ">="))
        if args.gate_completion is not None:
            checks.append((f"{s['model']} completion rate", s["completion_rate"], args.gate_completion, ">="))
        if args.gate_latency_ms is not None:
            checks.append((f"{s['model']} latency p50 ms", s["latency_ms_p50"], args.gate_latency_ms, "<="))
    if run.get("simulated"):
        print("note: produced by the Ollama simulator – not a real-model measurement")
    code, lines = _gates(checks)
    print("\n".join(lines))
    return code


def cmd_agents(args) -> int:
    api = Api(args.server, args.token)
    body = {"suite_id": args.suite, "models": args.model, "repeats": args.repeats, "mode": args.mode}
    run = api.call("POST", "/api/agents/runs", json=body).json()
    print(f"agent run {run['id']} started", file=sys.stderr)
    run = api.wait(f"/api/agents/runs/{run['id']}", args.timeout)
    if args.json:
        _print({k: run.get(k) for k in ("id", "status", "summary", "warnings")}, True)
    if run["status"] != "completed":
        print(f"run {run['status']}: {run.get('error') or ''}")
        return EXIT_ERROR
    checks = []
    for s in run["summary"]:
        print(f"{s['model']}: success {s['success_rate']} (CI {s.get('success_rate_ci')}), safety violations "
              f"{s.get('safety_violation_rate')} on {s.get('safety_tasks')} tasks, hard failures {s.get('hard_fails')}, "
              f"protocol compliance {s.get('protocol_compliance_rate')}")
        if args.gate_success is not None:
            checks.append((f"{s['model']} success rate", s["success_rate"], args.gate_success, ">="))
        if args.gate_max_violation_rate is not None:
            checks.append((f"{s['model']} safety violation rate", s.get("safety_violation_rate") or 0.0,
                           args.gate_max_violation_rate, "<="))
    for w in run.get("warnings") or []:
        print(f"warning: {w}")
    code, lines = _gates(checks)
    print("\n".join(lines))
    return code


def cmd_arena(args) -> int:
    api = Api(args.server, args.token)
    algorithms = args.algorithm or []
    if not (algorithms or args.uploaded or args.llm):  # nothing chosen: the dataset's default algorithms
        ds = api.call("GET", f"/api/arena/datasets/{args.dataset}").json()
        algorithms = [a["id"] for a in ds.get("algorithms", []) if a.get("default") and a.get("available")
                      and a["id"] != "dummy"]
        print(f"no models given – using the default algorithms: {', '.join(algorithms)}", file=sys.stderr)
    body = {"dataset_id": args.dataset, "algorithms": algorithms, "uploaded_models": args.uploaded or [],
            "llm_models": args.llm or [], "cv_folds": args.cv, "tune": args.tune, "objective": args.objective}
    if args.margin is not None:
        body["equivalence_margin"] = args.margin
    run = api.call("POST", "/api/arena/runs", json=body).json()
    print(f"arena run {run['id']} started", file=sys.stderr)
    run = api.wait(f"/api/arena/runs/{run['id']}", args.timeout)
    s = run.get("summary") or {}
    if args.json:
        _print({k: run.get(k) for k in ("id", "status", "protocol")} | {"summary": s}, True)
    if run["status"] != "completed":
        print(f"run {run['status']}: {run.get('error') or ''}")
        return EXIT_ERROR
    rec, final = s.get("recommendation"), s.get("final")
    print(f"recommendation: {rec['name'] if rec else 'none'} ({(rec or {}).get('status')})")
    if final:
        print(f"final test estimate: {final['label']} {final['value']} (95% CI {final.get('ci')}), n={final['n']}, "
              f"{'unbiased' if final['unbiased'] else 'exploratory'}")
    for w in s.get("warnings") or []:
        print(f"warning: {w}")
    checks = []
    if args.gate_final is not None:
        checks.append(("final test estimate", (final or {}).get("value"), args.gate_final, ">="))
    if args.require_clear:
        checks.append(("recommendation is clear (1=yes)", 1.0 if (rec or {}).get("status") == "clear" else 0.0, 1.0, ">="))
    code, lines = _gates(checks)
    print("\n".join(lines))
    return code


def cmd_bundle(args) -> int:
    if args.action == "verify":
        from app.services.bundles import verify_bundle

        rep = verify_bundle(Path(args.file).read_bytes())
        print(json.dumps(rep, indent=2))
        return EXIT_OK if rep["ok"] else EXIT_GATE
    api = Api(args.server, args.token)
    r = api.call("GET", f"/api/bundles/{args.kind}/{args.run}", params={"scope": args.scope,
                                                                        "include_dataset": args.include_dataset})
    out = Path(args.output or f"experiment_{args.kind}_{args.run}_{args.scope}.zip")
    out.write_bytes(r.content)
    print(f"bundle written: {out} ({len(r.content) / 1024:.0f} KB)")
    return EXIT_OK


def cmd_setup(args) -> int:
    api = Api(args.server, args.token)
    rep = api.call("GET", "/api/setup/check").json()
    for c in rep["checks"]:
        print(f"{c['status'].upper():5s} {c['label']}: {c['detail']}" + (f"\n      fix: {c['fix']}" if c.get("fix") else ""))
    return EXIT_OK if rep["status"] != "fail" else EXIT_GATE


# ----------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lal", description="Local AI Lab command line")
    sub = p.add_subparsers(dest="cmd", required=True)

    def server_opts(sp):
        sp.add_argument("--server", default=os.environ.get("LAL_SERVER", "http://127.0.0.1:8000"))
        sp.add_argument("--token", default=os.environ.get("LAL_TOKEN"))
        sp.add_argument("--json", action="store_true", help="print the machine-readable result too")
        sp.add_argument("--timeout", type=int, default=7200, help="seconds to wait for the run")

    db = sub.add_parser("db", help="database schema")
    db.add_argument("action", choices=["upgrade", "current"])
    db.set_defaults(fn=cmd_db)

    b = sub.add_parser("backup", help="backup / restore")
    b.add_argument("action", choices=["create", "list", "verify", "restore"])
    b.add_argument("file", nargs="?")
    b.add_argument("--label")
    b.add_argument("--output", help="folder for the backup (default data/backups)")
    b.add_argument("--keep", type=int, help="keep only the newest N backups")
    b.add_argument("--no-mlflow", action="store_true")
    b.add_argument("--backbones", action="store_true", help="include the pretrained image backbones")
    b.add_argument("--yes", action="store_true")
    b.add_argument("--force", action="store_true")
    b.add_argument("--server", default=os.environ.get("LAL_SERVER", "http://127.0.0.1:8000"))
    b.set_defaults(fn=cmd_backup)

    e = sub.add_parser("evaluators", help="evaluator calibration")
    e.add_argument("action", choices=["agreement"])
    e.add_argument("--file")
    e.add_argument("--json", action="store_true")
    e.set_defaults(fn=cmd_evaluators)

    be = sub.add_parser("bench", help="LLM benchmark runs (CI gate)")
    be.add_argument("action", choices=["run"])
    be.add_argument("--suite", required=True)
    be.add_argument("--model", action="append", required=True)
    be.add_argument("--repeats", type=int, default=1)
    be.add_argument("--max-cases", type=int)
    be.add_argument("--no-warmup", action="store_true")
    be.add_argument("--gate-quality", type=float)
    be.add_argument("--gate-completion", type=float)
    be.add_argument("--gate-latency-ms", type=float)
    server_opts(be)
    be.set_defaults(fn=cmd_bench)

    ag = sub.add_parser("agents", help="Agent Arena runs (CI gate)")
    ag.add_argument("action", choices=["run"])
    ag.add_argument("--suite", required=True)
    ag.add_argument("--model", action="append", required=True)
    ag.add_argument("--repeats", type=int, default=1)
    ag.add_argument("--mode", default="auto", choices=["auto", "native", "prompted"])
    ag.add_argument("--gate-success", type=float)
    ag.add_argument("--gate-max-violation-rate", type=float)
    server_opts(ag)
    ag.set_defaults(fn=cmd_agents)

    ar = sub.add_parser("arena", help="Model Arena runs (CI gate)")
    ar.add_argument("action", choices=["run"])
    ar.add_argument("--dataset", type=int, required=True)
    ar.add_argument("--algorithm", action="append")
    ar.add_argument("--uploaded", type=int, action="append")
    ar.add_argument("--llm", action="append")
    ar.add_argument("--cv", type=int, default=5)
    ar.add_argument("--tune", action="store_true")
    ar.add_argument("--objective", default="efficiency", choices=["efficiency", "quality"])
    ar.add_argument("--margin", type=float)
    ar.add_argument("--gate-final", type=float, help="minimum final test estimate of the recommended model")
    ar.add_argument("--require-clear", action="store_true", help="fail unless the recommendation is 'clear'")
    server_opts(ar)
    ar.set_defaults(fn=cmd_arena)

    bu = sub.add_parser("bundle", help="experiment bundles")
    bu.add_argument("action", choices=["export", "verify"])
    bu.add_argument("file", nargs="?")
    bu.add_argument("--kind", choices=["benchmark", "arena", "agent"])
    bu.add_argument("--run", type=int)
    bu.add_argument("--scope", default="full", choices=["full", "metadata"])
    bu.add_argument("--include-dataset", action="store_true")
    bu.add_argument("-o", "--output")
    server_opts(bu)
    bu.set_defaults(fn=cmd_bundle)

    st = sub.add_parser("setup", help="first-run checks")
    st.add_argument("action", choices=["check"])
    server_opts(st)
    st.set_defaults(fn=cmd_setup)
    return p


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("LOCAL_AI_LAB_NO_APP", "1")
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args) or 0)
    except SystemExit as exc:
        if isinstance(exc.code, str):
            print(exc.code, file=sys.stderr)
            return EXIT_ERROR
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
