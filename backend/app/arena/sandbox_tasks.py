"""Work done inside the sandbox process (see ``app.arena.sandbox``). Also callable in-process."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

from app.arena.service import take
from app.arena.uploaded import inspect_model, load_predictor


class _Peak:
    def __init__(self) -> None:
        import psutil

        self.p = psutil.Process()
        self.peak = self.p.memory_info().rss
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(0.1):
            self.peak = max(self.peak, self.p.memory_info().rss)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self.peak = max(self.peak, self.p.memory_info().rss)


def inspect(path: str, fmt: str, kind: str, task: str, role: str, config: dict, trusted: bool) -> dict:
    info, cfg = inspect_model(Path(path), fmt, kind, task, role, config, trusted)
    return {"inspection": info, "config": cfg}


def evaluate(fmt: str, path: str, kind: str, task: str, config: dict, labels: list[str], columns: list[str] | None,
             X_val, X_test, singles: int) -> dict:
    predictor = load_predictor(fmt, Path(path), kind, task, config, labels, columns)
    with _Peak() as mem:
        pv = predictor.predict(X_val) if X_val is not None and len(X_val) else None
        t0 = time.perf_counter()
        p = predictor.predict(X_test)
        batch_ms = (time.perf_counter() - t0) * 1000
        timing_X = X_val if pv is not None else X_test
        times = []
        for i in range(min(singles, len(timing_X))):
            one = take(timing_X, [i])
            t1 = time.perf_counter()
            predictor.predict(one)
            times.append((time.perf_counter() - t1) * 1000)
    pack = lambda x: None if x is None else {"values": np.asarray(x.values), "scores": x.scores,  # noqa: E731
                                             "notes": list(x.notes or [])}
    return {"val": pack(pv), "test": pack(p), "batch_ms": batch_ms, "singles": times,
            "peak_rss_mb": round(mem.peak / 1024 / 1024, 1)}


def extract(path: str, preprocess: dict, image_paths: list[str], step: int = 32) -> dict:
    from app.arena import images as IMG

    sess, feat = IMG.feature_session(Path(path))
    pp = IMG.ImagePreprocess(**preprocess)
    paths = [Path(p) for p in image_paths]

    def fn(part):
        return sess.run_batched(IMG.preprocess(part, pp), output=feat).reshape(len(part), -1)

    t0 = time.perf_counter()
    chunks = [fn(paths[i:i + step]) for i in range(0, len(paths), step)]
    batch_ms = (time.perf_counter() - t0) * 1000 / max(1, len(paths))
    singles = []
    for p in paths[: min(10, len(paths))]:
        t1 = time.perf_counter()
        fn([p])
        singles.append((time.perf_counter() - t1) * 1000)
    X = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros((0, 1), np.float32)
    return {"X": X, "ms_per_image": round(batch_ms, 3), "ms_single_p50": round(float(np.median(singles)), 3)
            if singles else 0.0}


def crash_for_test(kind: str = "segfault") -> dict:  # pragma: no cover - exercised in a child process
    """Used by the tests to prove that a crashing model cannot take the server down."""
    if kind == "segfault":
        import ctypes

        ctypes.string_at(0)
    if kind == "memory":
        blob = [bytearray(50 * 1024 * 1024) for _ in range(400)]
        return {"n": len(blob)}
    if kind == "hang":
        time.sleep(3600)
    return {}


TASKS = {"inspect": inspect, "evaluate": evaluate, "extract": extract, "crash_for_test": crash_for_test}
