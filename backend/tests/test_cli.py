"""Command line: local backup/db/evaluator commands and server-side CI gates."""

from __future__ import annotations

import threading
import time

import pytest
import uvicorn

from app.cli import main
from app.core.config import Settings
from app.main import create_app


@pytest.fixture
def env(tmp_path, monkeypatch, fake_ollama_url):
    for k, v in {"DATA_DIR": str(tmp_path / "data"), "OLLAMA_BASE_URL": fake_ollama_url, "MLFLOW_ENABLED": "false",
                 "APP_ENV": "test", "DOC_PROCESSING_MODE": "inline", "EMBEDDING_MODEL": "sim-embed",
                 "DEFAULT_MODEL": "sim-small:1b", "LOG_LEVEL": "WARNING"}.items():
        monkeypatch.setenv(k, v)
    return tmp_path


@pytest.fixture
def server(env):
    from conftest import _free_port

    port = _free_port()
    app = create_app(Settings())
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    end = time.time() + 15
    while not srv.started and time.time() < end:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    t.join(timeout=10)


def test_local_db_backup_and_evaluator_commands(env, capsys):
    assert main(["db", "upgrade"]) == 0
    assert main(["db", "current"]) == 0
    assert "latest" in capsys.readouterr().out
    assert main(["backup", "create", "--label", "cli"]) == 0
    out = capsys.readouterr().out
    assert "verified" in out
    path = out.split("backup written: ")[1].split(" (")[0]
    assert main(["backup", "verify", path]) == 0
    assert main(["backup", "restore", path]) == 2  # needs --yes
    assert main(["backup", "restore", path, "--yes", "--server", "http://127.0.0.1:9"]) == 0
    assert main(["evaluators", "agreement"]) == 0
    assert "agreement" in capsys.readouterr().out


def test_ci_gates_against_a_running_server(server, capsys):
    ok = main(["bench", "run", "--suite", "quick", "--model", "sim-medium:3b", "--max-cases", "4",
               "--gate-completion", "1.0", "--gate-quality", "0.0", "--server", server])
    out = capsys.readouterr().out
    assert ok == 0 and "PASS" in out and "simulator" in out
    fail = main(["bench", "run", "--suite", "quick", "--model", "sim-small:1b", "--max-cases", "4",
                 "--gate-quality", "1.01", "--server", server])
    assert fail == 1 and "FAIL" in capsys.readouterr().out
    gate = main(["agents", "run", "--suite", "agent-safety", "--model", "sim-small:1b", "--gate-max-violation-rate",
                 "0", "--server", server])
    out = capsys.readouterr().out
    assert gate in (0, 1) and "safety violation rate" in out
    assert main(["setup", "check", "--server", server]) in (0, 1)
    assert main(["bench", "run", "--suite", "nope", "--model", "sim-small:1b", "--server", server]) == 2


def test_bundle_export_and_verify_via_cli(server, tmp_path, capsys):
    import httpx

    rid = httpx.post(f"{server}/api/benchmarks/runs", json={"suite_id": "quick", "models": ["sim-small:1b"],
                                                           "max_cases": 2}).json()["id"]
    end = time.time() + 30
    while httpx.get(f"{server}/api/benchmarks/runs/{rid}").json()["status"] != "completed" and time.time() < end:
        time.sleep(0.2)
    out = tmp_path / "b.zip"
    assert main(["bundle", "export", "--kind", "benchmark", "--run", str(rid), "--scope", "metadata", "-o", str(out),
                 "--server", server]) == 0
    assert main(["bundle", "verify", str(out)]) == 0


def test_arena_cli_uses_default_algorithms_and_reports_the_final_estimate(server, capsys):
    import httpx
    import numpy as np

    rng = np.random.default_rng(0)
    rows = ["x1,x2,label"] + [f"{a:.3f},{b:.3f},{'yes' if a + b > 1 else 'no'}"
                               for a, b in rng.random((240, 2))]
    with httpx.Client(base_url=server, trust_env=False, timeout=60) as c:
        ds = c.post("/api/arena/datasets", files={"file": ("toy.csv", "\n".join(rows).encode())},
                    data={"target": "label"}).json()
    code = main(["arena", "run", "--dataset", str(ds["id"]), "--cv", "0", "--gate-final", "0.5", "--server", server])
    out = capsys.readouterr()
    assert "using the default algorithms" in out.err
    assert "final test estimate" in out.out and "unbiased" in out.out
    assert code == 0
