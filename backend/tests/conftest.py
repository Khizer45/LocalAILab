"""Shared fixtures.

* A real HTTP server running the Ollama *simulator* (devtools/fake_ollama.py)
  is started once per test session on 127.0.0.1, so tests exercise the real
  HTTP/streaming code paths without needing downloaded models.
* `make_app` builds an isolated app (fresh temp data dir + SQLite DB).
* A global audit hook records every socket.connect so privacy tests can prove
  that no non-loopback connection was attempted.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

os.environ["LOCAL_AI_LAB_NO_APP"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402
import uvicorn  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.core.config import Settings  # noqa: E402
from app.core.network_guard import guard  # noqa: E402
from app.main import create_app  # noqa: E402
from devtools.fake_ollama import create_app as create_fake  # noqa: E402

# ---------------------------------------------------------------- connection recorder
CONNECTIONS: list[tuple[str, int]] = []
RECORDING = threading.Event()


def _recorder(event: str, args: tuple) -> None:
    if event == "socket.connect" and RECORDING.is_set():
        try:
            sock, address = args
            if sock.family in (socket.AF_INET, socket.AF_INET6):
                CONNECTIONS.append((str(address[0]), int(address[1])))
        except Exception:  # noqa: BLE001
            pass


sys.addaudithook(_recorder)


@pytest.fixture
def record_connections():
    CONNECTIONS.clear()
    RECORDING.set()
    yield CONNECTIONS
    RECORDING.clear()


# ---------------------------------------------------------------- fake Ollama servers
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Server:
    def __init__(self, speed: float = 0.0):
        self.port = _free_port()
        self.server = uvicorn.Server(uvicorn.Config(create_fake(speed), host="127.0.0.1", port=self.port,
                                                    log_level="error"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self) -> str:
        self.thread.start()
        deadline = time.time() + 10
        while not self.server.started:
            if time.time() > deadline:
                raise RuntimeError("fake ollama did not start")
            time.sleep(0.02)
        return f"http://127.0.0.1:{self.port}"

    def __exit__(self, *exc):
        self.server.should_exit = True
        self.thread.join(timeout=5)


@pytest.fixture(scope="session")
def fake_ollama_url():
    with _Server(speed=0.0) as url:
        yield url


@pytest.fixture(scope="session")
def fake_cloud_url():
    """Second simulator instance used as a stand-in for the Ollama Cloud API."""
    with _Server(speed=0.0) as url:
        yield url


# ---------------------------------------------------------------- app factory
@pytest.fixture
def make_app(tmp_path, fake_ollama_url):
    clients: list[TestClient] = []

    def _make(**overrides) -> TestClient:
        params = dict(
            app_env="test", data_dir=tmp_path / f"data{len(clients)}", ollama_base_url=fake_ollama_url,
            doc_processing_mode="inline", mlflow_enabled=False, embedding_model="sim-embed", log_json=True,
            log_level="WARNING", default_model="sim-small:1b",
        )
        params.update(overrides)
        settings = Settings(_env_file=None, **params)
        client = TestClient(create_app(settings))
        client.__enter__()
        clients.append(client)
        return client

    yield _make
    for c in clients:
        c.__exit__(None, None, None)
    guard.configure(offline_only=True)


@pytest.fixture
def client(make_app) -> TestClient:
    return make_app()


# ---------------------------------------------------------------- helpers
def stream_events(response) -> list[dict]:
    assert response.status_code == 200, response.text
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


def wait_for(predicate, timeout: float = 20.0, interval: float = 0.1):
    end = time.time() + timeout
    while time.time() < end:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    raise AssertionError("condition not met in time")


def make_pdf(path: Path, pages: list[str]) -> Path:
    """Write a simple text PDF without any third-party library (Helvetica, wrapped at ~85 characters)."""
    import textwrap

    def esc(t: str) -> str:
        return t.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    objects: list[bytes] = []
    n_pages = len(pages)
    font_id = 3 + 2 * n_pages
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n_pages))
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode())
    for i, text in enumerate(pages):
        lines = textwrap.wrap(text, 85) or [""]
        ops = ["BT", "/F1 11 Tf", "14 TL", "50 790 Td"] + [f"({esc(line)}) Tj T*" for line in lines] + ["ET"]
        stream = "\n".join(ops).encode("latin-1", errors="replace")
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents {4 + 2 * i} 0 R "
                       f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>".encode())
        objects.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for num, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{num} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))
    return path
