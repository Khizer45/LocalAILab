"""Uploaded models run in an isolated, limited worker: crashes, memory blow-ups and hangs cannot take the server down."""

from __future__ import annotations

import os
import sys

import pytest

from app.arena.sandbox import run_isolated
from app.core.errors import AppError


@pytest.mark.skipif(sys.platform == "win32", reason="segfault via ctypes is POSIX-specific in this test")
def test_a_crashing_model_only_kills_the_worker():
    with pytest.raises(AppError) as e:
        run_isolated("crash_for_test", {"kind": "segfault"}, timeout_s=60, memory_mb=2048)
    assert e.value.code == "sandbox_crash"
    assert os.getpid()  # this process is still alive


def test_memory_and_time_limits():
    with pytest.raises(AppError) as e:
        run_isolated("crash_for_test", {"kind": "memory"}, timeout_s=60, memory_mb=300)
    assert e.value.code in ("sandbox_memory", "sandbox_error", "sandbox_crash")
    with pytest.raises(AppError) as e:
        run_isolated("crash_for_test", {"kind": "hang"}, timeout_s=3, memory_mb=2048)
    assert e.value.code == "sandbox_timeout"


def test_inspection_errors_come_back_as_normal_errors(tmp_path):
    bad = tmp_path / "bad.onnx"
    bad.write_bytes(b"not an onnx file")
    with pytest.raises(AppError) as e:
        run_isolated("inspect", {"path": str(bad), "fmt": "onnx", "kind": "image", "task": "classification",
                                 "role": "predictor", "config": {}, "trusted": False}, timeout_s=120, memory_mb=2048)
    assert e.value.status_code in (400, 422)


def test_uploaded_model_results_say_they_ran_isolated(client):
    from pathlib import Path

    samples = Path(__file__).resolve().parents[2] / "samples" / "arena"
    with (samples / "shapes_images.zip").open("rb") as f:
        ds = client.post("/api/arena/datasets", files={"file": ("s.zip", f)}).json()
    with (samples / "shapes_classifier.onnx").open("rb") as f:
        m = client.post("/api/arena/models", files={"file": ("s.onnx", f)},
                        data={"kind": "image", "task": "classification", "normalization": "zero_one"}).json()
    assert m["inspection"]["isolated"] is True
    from conftest import wait_for

    rid = client.post("/api/arena/runs", json={"dataset_id": ds["id"], "uploaded_models": [m["id"]], "cv_folds": 0,
                                               "bootstrap_resamples": 300}).json()["id"]
    run = wait_for(lambda: (d := client.get(f"/api/arena/runs/{rid}").json())["status"] == "completed" and d, timeout=120)
    up = next(r for r in run["results"] if r["source"] == "uploaded")
    assert up["status"] == "ok" and up["details"]["isolated"] is True and up["metrics"]["peak_process_rss_mb"] > 0
