"""Model Arena: dataset ingestion, built-in algorithms, uploaded models, statistics, security."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pytest
from conftest import wait_for

SAMPLES = Path(__file__).resolve().parents[2] / "samples" / "arena"


def _upload_dataset(c, name: str, headers=None, **form):
    with (SAMPLES / name).open("rb") as f:
        r = c.post("/api/arena/datasets", headers=headers or {}, files={"file": (name, f)},
                   data={k: str(v) for k, v in form.items()})
    assert r.status_code == 201, r.text
    return r.json()


def _run(c, headers=None, **body):
    r = c.post("/api/arena/runs", headers=headers or {}, json={"cv_folds": 0, **body})
    assert r.status_code == 201, r.text
    rid = r.json()["id"]
    return wait_for(lambda: (d := c.get(f"/api/arena/runs/{rid}", headers=headers or {}).json())["status"]
                    not in ("queued", "running") and d, timeout=180)


def _tiny_onnx_classifier(path: Path, n_in: int = 3 * 16 * 16, n_out: int = 3, image: bool = True) -> Path:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(0)
    W = rng.normal(0, 0.01, (n_in, n_out)).astype(np.float32)
    b = np.zeros(n_out, dtype=np.float32)
    shape = ["N", 3, 16, 16] if image else ["N", n_in]
    nodes = [helper.make_node("Flatten", ["x"], ["f"], axis=1), helper.make_node("Gemm", ["f", "W", "b"], ["logits"]),
             helper.make_node("Softmax", ["logits"], ["probs"], axis=1)]
    graph = helper.make_graph(nodes, "tiny", [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
                              [helper.make_tensor_value_info("probs", TensorProto.FLOAT, ["N", n_out])],
                              [numpy_helper.from_array(W, "W"), numpy_helper.from_array(b, "b")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, path)
    return path


# ----------------------------------------------------------------------------- datasets
def test_csv_preview_suggests_target_and_types(client):
    with (SAMPLES / "customer_churn.csv").open("rb") as f:
        r = client.post("/api/arena/datasets/preview", files={"file": ("customer_churn.csv", f)})
    assert r.status_code == 200, r.text
    p = r.json()
    assert p["suggested_target"] == "churn" and p["suggested_task"] == "classification"
    types = {c["name"]: c["type"] for c in p["columns"]}
    assert types["customer_id"] == "id" and types["tenure_months"] == "numeric" and types["contract"] == "categorical"
    assert len(p["head"]) == 8


def test_tabular_dataset_profile_and_stratified_split(client):
    ds = _upload_dataset(client, "customer_churn.csv", target="churn", test_size=0.2, seed=42)
    assert ds["kind"] == "tabular" and ds["task"] == "classification" and ds["labels"] == ["no", "yes"]
    assert "customer_id" in ds["columns"]["ignored"] and "customer_id" not in ds["columns"]["features"]
    assert ds["n_train"] + ds["n_val"] + ds["n_test"] == ds["n_rows"] == 700
    assert ds["n_test"] == 140 and ds["n_val"] == 140 and len(ds["split_sha256"]) == 64
    prof = ds["profile"]
    ratio_all = prof["class_counts"]["yes"] / ds["n_rows"]
    for part in ("val", "test"):  # stratified
        assert abs(ratio_all - prof[f"class_counts_{part}"]["yes"] / ds[f"n_{part}"]) < 0.03
    assert prof["leakage"]["exact"] == {"train-test": 0, "train-val": 0, "val-test": 0}
    detail = client.get(f"/api/arena/datasets/{ds['id']}").json()
    ids = {a["id"] for a in detail["algorithms"]}
    assert {"random_forest", "logistic_regression", "dummy"} <= ids and "ridge" not in ids


def test_regression_and_text_detection(client):
    houses = _upload_dataset(client, "house_prices.csv", target="price_lakh_pkr")
    assert houses["task"] == "regression" and houses["profile"]["target_stats"]["max"] > 0
    tickets = _upload_dataset(client, "support_tickets.csv", target="category")
    assert tickets["kind"] == "text" and tickets["text_column"] == "ticket" and len(tickets["labels"]) == 5


def test_bad_csv_inputs_are_rejected(client):
    r = client.post("/api/arena/datasets", files={"file": ("one.csv", b"only_one_column\n1\n2\n3\n")})
    assert r.status_code == 422 and "two columns" in r.json()["error"]["message"]
    r = client.post("/api/arena/datasets", files={"file": ("t.csv", b"a,b\nx,cat\ny,dog\n" * 10)},
                    data={"target": "b", "task": "regression"})
    assert r.status_code == 422 and "numeric target" in r.json()["error"]["message"]
    r = client.post("/api/arena/datasets", files={"file": ("x.exe", b"MZ....")})
    assert r.status_code == 422


def _zip_bytes(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _png(color: tuple[int, int, int]) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (20, 20), color).save(buf, format="PNG")
    return buf.getvalue()


def test_image_zip_layouts_and_zip_slip(client):
    # provided split: train/<class>, test/<class> inside a wrapper folder
    entries = {f"data/train/{c}/{i}.png": _png((i * 9, 50, 200) if c == "a" else (200, i * 9, 50))
               for c in ("a", "b") for i in range(6)}
    entries.update({f"data/test/{c}/{i}.png": _png((10, 10, 10 + i)) for c in ("a", "b") for i in range(3)})
    entries["data/__MACOSX/._junk.png"] = b"junk"
    r = client.post("/api/arena/datasets", files={"file": ("imgs.zip", _zip_bytes(entries))})
    assert r.status_code == 201, r.text
    d = r.json()
    assert d["kind"] == "image" and d["split_source"] == "provided" and d["n_test"] == 6
    assert d["n_train"] + d["n_val"] == 12 and d["n_val"] >= 2  # validation taken from train/, test/ unchanged
    assert any("duplicate" in w for w in d["warnings"])  # identical test images
    assert any("near-duplicates" in w for w in d["warnings"])  # the dark test images look alike

    evil = _zip_bytes({"../../escape.png": _png((1, 2, 3)), "a/1.png": _png((1, 2, 3))})
    r = client.post("/api/arena/datasets", files={"file": ("evil.zip", evil)})
    assert r.status_code == 422 and r.json()["error"]["code"] == "unsafe_archive"
    r = client.post("/api/arena/datasets", files={"file": ("flat.zip", _zip_bytes({"1.png": _png((0, 0, 0))}))})
    assert r.status_code == 422


# ----------------------------------------------------------------------------- comparisons
def test_tabular_comparison_with_uploaded_joblib_model(client, tmp_path):
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline

    ds = _upload_dataset(client, "customer_churn.csv", target="churn")
    df = pd.read_csv(SAMPLES / "customer_churn.csv")
    cols = ["tenure_months", "support_calls", "contract"]
    pipe = make_pipeline(_passthrough(cols), RandomForestClassifier(50, random_state=0))
    pipe.fit(df[cols], df["churn"])
    path = tmp_path / "rf.joblib"
    joblib.dump(pipe, path)
    with path.open("rb") as f:  # untrusted pickle is refused
        r = client.post("/api/arena/models", files={"file": ("rf.joblib", f)},
                        data={"kind": "tabular", "task": "classification"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "untrusted_pickle"
    with path.open("rb") as f:
        model = client.post("/api/arena/models", files={"file": ("rf.joblib", f)},
                            data={"kind": "tabular", "task": "classification", "trusted": "true", "name": "My RF"}).json()
    assert model["format"] == "joblib" and model["inspection"]["classes"] == ["no", "yes"]

    run = _run(client, dataset_id=ds["id"], algorithms=["logistic_regression", "random_forest"],
               uploaded_models=[model["id"]], cv_folds=2)
    assert run["status"] == "completed", run["error"]
    res = {r["key"]: r for r in run["results"]}
    assert set(res) == {"baseline:dummy", "builtin:logistic_regression", "builtin:random_forest", f"uploaded:{model['id']}"}
    assert all(r["status"] == "ok" for r in res.values()), [r["error"] for r in res.values()]
    lr = res["builtin:logistic_regression"]["metrics"]
    assert lr["f1_macro"] > res["baseline:dummy"]["metrics"]["f1_macro"]
    lo, hi = lr["f1_macro_ci"]
    assert lo <= lr["f1_macro"] <= hi
    assert res["builtin:logistic_regression"]["details"]["cv"]["folds"] == 2
    assert lr["latency_ms_single_p50"] > 0 and lr["model_size_bytes"] > 0
    assert res["builtin:logistic_regression"]["val_metrics"]["n"] == ds["n_val"]
    assert res["builtin:logistic_regression"]["details"]["fit_rows"] == ds["n_train"]
    s = run["summary"]
    assert run["protocol"] == "v2" and s["protocol"] == "v2"
    assert s["primary_metric"] == "f1_macro" and s["ranking"][-1] == "baseline:dummy"
    sel = s["selection"]
    assert sel["split"] == "validation" and s["recommendation"]["key"] in s["ranking"]
    assert sel["comparisons"]["baseline:dummy"]["verdict"] == "worse"
    up = res[f"uploaded:{model['id']}"]
    assert any("Training data unknown" in c for c in up["details"]["caveats"])
    assert s["final"]["split"] == "test" and s["final"]["unbiased"] is True

    md = client.get(f"/api/arena/runs/{run['id']}/export?format=md").text
    assert "Recommendation" in md and "Macro F1" in md and "Selection on the validation split" in md
    csv_text = client.get(f"/api/arena/runs/{run['id']}/export?format=csv").text
    assert csv_text.splitlines()[0].startswith("validation_rank,key,name")
    full = client.get(f"/api/arena/runs/{run['id']}/export?format=json").json()
    assert len(full["results"][0]["predictions"]) == ds["n_test"]
    assert len(full["results"][0]["val_predictions"]) == ds["n_val"]
    blob = client.get(f"/api/arena/runs/{run['id']}/model")
    assert blob.status_code == 200
    bundle = joblib.load(io.BytesIO(blob.content))
    assert bundle["labels"] == ["no", "yes"] and hasattr(bundle["pipeline"], "predict")


def _passthrough(cols):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OrdinalEncoder

    return ColumnTransformer([("num", "passthrough", cols[:2]), ("cat", OrdinalEncoder(), cols[2:])])


def test_regression_comparison_with_tuning(client):
    ds = _upload_dataset(client, "house_prices.csv", target="price_lakh_pkr")
    run = _run(client, dataset_id=ds["id"], algorithms=["ridge", "gradient_boosting"], tune=True, tune_iterations=3)
    assert run["status"] == "completed"
    res = {r["key"]: r for r in run["results"]}
    assert res["builtin:ridge"]["metrics"]["r2"] > 0.8 and abs(res["baseline:dummy"]["metrics"]["r2"]) < 0.1
    assert "best_params" in res["builtin:gradient_boosting"]["details"]["tuning"]
    assert run["summary"]["primary_metric"] == "r2"
    assert res["builtin:ridge"]["details"]["errors_sample"][0]["abs_error"] >= 0


def test_text_comparison_with_local_llm(client):
    ds = _upload_dataset(client, "support_tickets.csv", target="category")
    run = _run(client, dataset_id=ds["id"], algorithms=["naive_bayes"], llm_models=["sim-medium:3b"],
               llm_max_examples=20)
    assert run["status"] == "completed", run["error"]
    llm = next(r for r in run["results"] if r["source"] == "llm")
    assert llm["status"] == "ok" and llm["metrics"]["n"] == 20 and llm["details"]["evaluated"] == 20
    assert llm["details"]["notes"]  # says it was a sample
    assert llm["val_metrics"]["n"] == 20
    comp = run["summary"]["selection"]["comparisons"].get(llm["key"]) or {}
    assert comp.get("n") in (None, 20)  # compared on the shared examples only


def test_llm_only_for_text_classification(client):
    ds = _upload_dataset(client, "house_prices.csv", target="price_lakh_pkr")
    r = client.post("/api/arena/runs", json={"dataset_id": ds["id"], "algorithms": ["ridge"],
                                             "llm_models": ["sim-medium:3b"]})
    assert r.status_code == 422
    r = client.post("/api/arena/runs", json={"dataset_id": ds["id"], "algorithms": ["naive_bayes"]})
    assert r.status_code == 422  # naive bayes is classification only
    r = client.post("/api/arena/runs", json={"dataset_id": ds["id"]})
    assert r.status_code == 422  # nothing to compare


def test_image_comparison_with_onnx_classifier_and_feature_extractor(client, tmp_path):
    ds = _upload_dataset(client, "shapes_images.zip", test_size=0.25)
    assert ds["kind"] == "image" and ds["labels"] == ["circle", "square", "triangle"]
    with (SAMPLES / "shapes_classifier.onnx").open("rb") as f:
        clf = client.post("/api/arena/models", files={"file": ("shapes_classifier.onnx", f)},
                          data={"kind": "image", "task": "classification", "normalization": "zero_one"}).json()
    assert clf["config"]["preprocess"]["size"] == 32 and clf["config"]["preprocess"]["layout"] == "nchw"
    fx_path = _tiny_onnx_classifier(tmp_path / "backbone.onnx", n_out=10)
    with fx_path.open("rb") as f:
        fx = client.post("/api/arena/models", files={"file": ("backbone.onnx", f)},
                         data={"kind": "image", "role": "feature_extractor"}).json()
    assert fx["role"] == "feature_extractor" and fx["inspection"]["feature_dim"] == 768

    r = client.post("/api/arena/runs", json={"dataset_id": ds["id"], "algorithms": ["knn"],
                                             "image_features": ["backbone:mobilenetv2"]})
    assert r.status_code == 422 and "not downloaded" in r.json()["error"]["message"]

    run = _run(client, dataset_id=ds["id"], algorithms=["logistic_regression"], uploaded_models=[clf["id"]],
               image_features=["raw_pixels", f"model:{fx['id']}"])
    assert run["status"] == "completed", run["error"]
    res = {r["key"]: r for r in run["results"]}
    assert all(r["status"] == "ok" for r in res.values()), {k: r["error"] for k, r in res.items()}
    up = res[f"uploaded:{clf['id']}"]
    assert up["metrics"]["accuracy"] > 0.5 and any("alphabetically" in n for n in up["details"]["notes"])
    raw = res["builtin:logistic_regression@raw_pixels"]
    assert raw["details"]["feature_extractor"]["dim"] == 32 * 32 * 3
    assert raw["metrics"]["feature_ms_per_example"] > 0
    assert f"builtin:logistic_regression@model:{fx['id']}" in res


def test_wrong_output_size_and_unsupported_formats(client, tmp_path):
    ds = _upload_dataset(client, "shapes_images.zip")
    bad = _tiny_onnx_classifier(tmp_path / "five.onnx", n_out=5)
    with bad.open("rb") as f:
        m = client.post("/api/arena/models", files={"file": ("five.onnx", f)},
                        data={"kind": "image", "task": "classification"}).json()
    run = _run(client, dataset_id=ds["id"], uploaded_models=[m["id"]], include_baseline=True)
    res = next(r for r in run["results"] if r["source"] == "uploaded")
    assert res["status"] == "error" and "5 output classes" in res["error"]
    r = client.post("/api/arena/models", files={"file": ("model.h5", b"\x89HDF")}, data={"kind": "image"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "convert_to_onnx"
    r = client.post("/api/arena/models", files={"file": ("broken.onnx", b"not onnx")}, data={"kind": "image"})
    assert r.status_code == 422


def test_backbone_download_blocked_offline(client):
    r = client.post("/api/arena/backbones/mobilenetv2/download")
    assert r.status_code == 403 and r.json()["error"]["code"] == "offline_mode_blocked"
    cat = client.get("/api/arena/catalog").json()
    assert {b["id"] for b in cat["backbones"]} == {"mobilenetv2", "efficientnet_lite4"}
    assert any(a["id"] == "xgboost" for a in cat["algorithms"])


def test_arena_isolation_and_admin_only_pickles(make_app, tmp_path):
    c = make_app(auth_enabled=True)

    def login(u):
        c.post("/api/auth/register", json={"username": u, "password": "password123"})
        return {"Authorization": "Bearer " + c.post("/api/auth/login", json={"username": u, "password": "password123"}).json()["token"]}

    admin, bob = login("admin1"), login("bob1")
    ds = _upload_dataset(c, "house_prices.csv", headers=admin, target="price_lakh_pkr")
    assert c.get(f"/api/arena/datasets/{ds['id']}", headers=bob).status_code == 404
    assert c.delete(f"/api/arena/datasets/{ds['id']}", headers=bob).status_code == 404
    assert c.post("/api/arena/runs", headers=bob, json={"dataset_id": ds["id"], "algorithms": ["ridge"]}).status_code == 404
    from sklearn.dummy import DummyClassifier

    p = tmp_path / "d.joblib"
    joblib.dump(DummyClassifier().fit([[0], [1]], ["a", "b"]), p)
    with p.open("rb") as f:
        r = c.post("/api/arena/models", headers=bob, files={"file": ("d.joblib", f)},
                   data={"kind": "tabular", "trusted": "true"})
    assert r.status_code == 403
    run = _run(c, headers=admin, dataset_id=ds["id"], algorithms=["ridge"])
    assert c.get(f"/api/arena/runs/{run['id']}", headers=bob).status_code == 404
    assert c.get("/api/arena/runs", headers=bob).json()["runs"] == []


# ----------------------------------------------------------------------------- units
def test_statistics_helpers():
    from app.arena.stats import (
        Policy,
        bootstrap_classification,
        classification_metrics,
        compare,
        make_resampling,
        mcnemar,
    )

    y = np.array([0, 1, 2] * 20)
    good = y.copy()
    good[:3] = (good[:3] + 1) % 3
    bad = np.zeros_like(y)
    m = classification_metrics(y, good, ["a", "b", "c"])
    assert m["accuracy"] == pytest.approx(57 / 60, abs=1e-4) and m["confusion_matrix"]["matrix"][0][1] == 1
    ci = bootstrap_classification(y, good, 3)
    assert ci["accuracy_ci"][0] <= m["accuracy"] <= ci["accuracy_ci"][1]
    t = mcnemar(good == y, bad == y)
    assert t["p_value"] < 0.001 and t["a_only_correct"] > t["b_only_correct"]
    assert mcnemar(good == y, good == y)["p_value"] == 1.0
    unparsed = classification_metrics(y, np.full_like(y, -1), ["a", "b", "c"])
    assert unparsed["accuracy"] == 0 and unparsed["unparseable"] == 60
    idx = np.arange(len(y))
    rows = [{"key": "good", "name": "Good", "source": "builtin", "metrics": m, "pred": good, "idx": idx, "latency_ms": 5.0},
            {"key": "fast", "name": "Fast", "source": "builtin", "metrics": m, "pred": good, "idx": idx, "latency_ms": 1.0},
            {"key": "base", "name": "Base", "source": "baseline",
             "metrics": classification_metrics(y, bad, ["a", "b", "c"]), "pred": bad, "idx": idx, "latency_ms": 0.1}]
    rs = make_resampling("classification", y, None, 500, 0)
    s = compare("classification", rows, y, 3, Policy("f1_macro"), rs)
    assert s["leader"] == "fast"  # equal quality: the faster one leads
    assert s["recommendation"]["key"] == "fast"
    assert s["comparisons"]["good"]["verdict"] == "equivalent"  # identical predictions
    assert s["comparisons"]["base"]["verdict"] == "worse"
    assert s["recommendation"]["status"] == "clear"


def test_every_available_algorithm_trains_on_every_kind():
    import pandas as pd

    from app.arena.algorithms import ALGORITHMS, build_pipeline

    rng = np.random.default_rng(0)
    n = 60
    tab = pd.DataFrame({"x": rng.normal(size=n), "c": rng.choice(["a", "b"], n).astype(object)})
    texts = np.array([f"word{i % 3} item {i} " + ("good great" if i % 2 else "bad awful") for i in range(n)], dtype=object)
    img = rng.normal(size=(n, 20)).astype(np.float32)
    y_cls = np.array(["p", "q"] * (n // 2))
    y_reg = rng.normal(size=n)
    for algo in ALGORITHMS.values():
        if not algo.available():
            continue
        for kind, X in (("tabular", tab), ("text", texts), ("image", img)):
            for task, y in (("classification", y_cls), ("regression", y_reg)):
                if not algo.supports(kind, task):
                    continue
                pipe = build_pipeline(algo, task, kind, types={"x": "numeric", "c": "categorical"}, features=["x", "c"])
                yy = np.array([0, 1] * (n // 2)) if (algo.id == "xgboost" and task == "classification") else y
                pipe.fit(X, yy)
                assert len(pipe.predict(X[:5] if not isinstance(X, pd.DataFrame) else X.iloc[:5])) == 5, (algo.id, kind, task)


def test_feature_tensor_detection(tmp_path):
    import onnx

    from app.arena.images import feature_session, find_feature_tensor

    p = _tiny_onnx_classifier(tmp_path / "m.onnx")
    assert find_feature_tensor(onnx.load(str(p))) == "f"
    sess, feat = feature_session(p)
    out = sess.run({"x": np.zeros((2, 3, 16, 16), np.float32)}, [feat])[0]
    assert out.shape == (2, 768)
