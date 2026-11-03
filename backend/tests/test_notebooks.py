"""The scikit-learn Colab notebook runs end to end: train on the training split only → ONNX with metadata →
upload → the Arena recognises the split and evaluates it. (The PyTorch notebook cannot run here: no torch.)"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from conftest import wait_for

pytest.importorskip("skl2onnx")
ROOT = Path(__file__).resolve().parents[2]


def _code(nb_name: str) -> list[str]:
    nb = json.loads((ROOT / "notebooks" / nb_name).read_text())
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    return ["\n".join(line for line in c.splitlines() if not line.lstrip().startswith(("%", "!"))) for c in cells]


@pytest.mark.parametrize("csv_name,settings", [
    ("customer_churn.csv", {"TARGET": "churn", "TEXT_COLUMN": None, "IGNORE": ["customer_id"], "MODEL": "random_forest"}),
    ("support_tickets.csv", {"TARGET": "category", "TEXT_COLUMN": "ticket", "IGNORE": [], "MODEL": "logistic_regression"}),
])
def test_sklearn_notebook_end_to_end(client, tmp_path, monkeypatch, csv_name, settings):
    src = ROOT / "samples" / "arena" / csv_name
    with src.open("rb") as f:
        ds = client.post("/api/arena/datasets", files={"file": (csv_name, f)}, data={"target": settings["TARGET"]}).json()
    (tmp_path / "data.csv").write_bytes(src.read_bytes())
    (tmp_path / "split.csv").write_text(client.get(f"/api/arena/datasets/{ds['id']}/split.csv").text)
    monkeypatch.chdir(tmp_path)
    ns: dict = {}
    for i, code in enumerate(_code("train_tabular_text_sklearn.ipynb")):
        if "TARGET =" in code and "TASK =" in code:  # the settings cell
            code += "\n" + "\n".join(f"{k} = {v!r}" for k, v in settings.items())
        exec(compile(code, f"cell{i}", "exec"), ns)  # noqa: S102 - our own notebook
    assert len(ns["train"]) == ds["n_train"]  # only the training rows were used
    kind = "text" if settings["TEXT_COLUMN"] else "tabular"
    with open("model.onnx", "rb") as f:
        m = client.post("/api/arena/models", files={"file": ("model.onnx", f)},
                        data={"kind": kind, "task": "classification"}).json()
    assert m["provenance"]["training_data"] == "train_split", m
    assert set(m["inspection"]["defaults_from_metadata"]) >= {"labels"}
    rid = client.post("/api/arena/runs", json={"dataset_id": ds["id"], "uploaded_models": [m["id"]],
                                               "algorithms": ["logistic_regression"] if kind == "tabular" else ["naive_bayes"],
                                               "cv_folds": 0, "bootstrap_resamples": 300}).json()["id"]
    run = wait_for(lambda: (d := client.get(f"/api/arena/runs/{rid}").json())["status"] == "completed" and d, timeout=180)
    up = next(r for r in run["results"] if r["source"] == "uploaded")
    assert up["status"] == "ok", up["error"]
    assert up["details"]["caveats"] == []  # split hash matches the declaration in the file
    assert up["metrics"]["accuracy"] > 0.5
    os.remove("model.onnx")


def test_pytorch_notebook_cells_compile_and_carry_the_metadata():
    code = _code("train_image_classifier_pytorch.ipynb")
    for i, c in enumerate(code):
        compile(c, f"cell{i}", "exec")
    joined = "\n".join(code)
    for key in ('"labels"', '"preprocess"', '"split_sha256"', '"trained_on": "train"', "normalization"):
        assert key in joined
    assert "early stopping uses a slice of the TRAINING images only" in joined
