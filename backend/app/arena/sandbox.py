"""Run uploaded models in a separate, limited worker process.

ONNX files cannot contain Python code, but they are still untrusted input to a large native parser and runtime: a
malformed or hostile file could crash the process, allocate huge amounts of memory or run for a very long time.
Pickle/joblib files (administrators only) can run arbitrary code when loaded. So inspection, prediction and feature
extraction for *uploaded* models run in a short-lived child process:

* separate process – a crash (segfault, abort) only kills the child; the server reports an error;
* memory limit – the parent watches the child's resident memory and kills it above ``ARENA_SANDBOX_MEMORY_MB``
  (on Linux/macOS an address-space limit is also set inside the child);
* time limit – the child is killed after ``ARENA_SANDBOX_TIMEOUT_S``;
* no network – the child installs the offline socket guard with no allowed hosts (Python-level sockets);
* minimal environment – proxy variables are removed.

This reduces, but does not remove, the risk of running untrusted model files: the child still runs as the same
user and can read the same files. Keep ONNX Runtime updated, prefer ONNX over pickle, and only upload models from
sources you trust. Pre-trained backbones shipped with the app are SHA-256-verified and run in-process.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import time
import traceback
from typing import Any

from app.core.errors import AppError

log = logging.getLogger("app.arena.sandbox")


def _apply_limits(memory_mb: int) -> None:
    try:
        import resource  # POSIX only

        limit = int(memory_mb) * 1024 * 1024 * 4  # address space is much larger than RSS (thread arenas, mmap)
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        if hard == resource.RLIM_INFINITY or limit < hard:
            resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
    except (ImportError, ValueError, OSError):
        pass


def _child(conn, task: str, kwargs: dict, memory_mb: int) -> None:
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(k, None)
    try:
        _apply_limits(memory_mb)
        from app.core.network_guard import guard

        guard.configure(offline_only=True, allowed_hosts=[])
        guard.install_audit_hook()
        from app.arena import sandbox_tasks

        fn = sandbox_tasks.TASKS[task]
        conn.send(("ok", fn(**kwargs)))
    except AppError as exc:
        conn.send(("apperror", {"message": exc.message, "code": exc.code, "hint": exc.hint,
                                "status_code": exc.status_code}))
    except MemoryError:
        conn.send(("error", "The model needed more memory than the sandbox allows."))
    except BaseException as exc:  # noqa: BLE001 - report everything to the parent
        conn.send(("error", f"{type(exc).__name__}: {str(exc)[:400]}", traceback.format_exc()[-2000:]))
    finally:
        conn.close()


def run_isolated(task: str, kwargs: dict, *, timeout_s: float, memory_mb: int) -> Any:
    import psutil

    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_child, args=(child, task, kwargs, memory_mb), daemon=True, name=f"sandbox-{task}")
    proc.start()
    child.close()
    deadline = time.monotonic() + timeout_s
    limit = memory_mb * 1024 * 1024
    peak = 0
    try:
        ps = psutil.Process(proc.pid)
    except psutil.Error:
        ps = None
    result = None
    try:
        while True:
            if parent.poll(0.1):
                try:
                    result = parent.recv()
                except EOFError:
                    result = None
                break
            if not proc.is_alive():
                break
            if ps is not None:
                try:
                    rss = ps.memory_info().rss
                    peak = max(peak, rss)
                    if rss > limit:
                        proc.kill()
                        raise AppError(f"The model used more than {memory_mb} MB of memory and was stopped.",
                                       code="sandbox_memory", status_code=422,
                                       hint="Raise ARENA_SANDBOX_MEMORY_MB in the backend .env if the model is "
                                            "trusted and the computer has enough RAM.")
                except psutil.Error:
                    pass
            if time.monotonic() > deadline:
                proc.kill()
                raise AppError(f"The model did not finish within {timeout_s:.0f} s and was stopped.",
                               code="sandbox_timeout", status_code=422,
                               hint="Raise ARENA_SANDBOX_TIMEOUT_S or use fewer / smaller examples.")
    finally:
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
        parent.close()
    if result is None:
        raise AppError(f"The model crashed the isolated worker (exit code {proc.exitcode}).", code="sandbox_crash",
                       status_code=422, hint="The file may be corrupt or use an operator ONNX Runtime cannot run.")
    kind = result[0]
    if kind == "ok":
        out = result[1]
        if isinstance(out, dict):
            out.setdefault("sandbox_peak_rss_mb", round(peak / 1024 / 1024, 1))
        return out
    if kind == "apperror":
        e = result[1]
        raise AppError(e["message"], code=e["code"], status_code=e["status_code"], hint=e["hint"])
    log.warning("sandbox task failed", extra={"task": task, "error": result[1]})
    raise AppError(f"The model failed in the isolated worker: {result[1]}", code="sandbox_error", status_code=422)
