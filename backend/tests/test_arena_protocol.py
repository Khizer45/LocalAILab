"""Model Arena protocol v2: selection on validation, statistics, leakage checks, splits, provenance.

These tests answer the review questions directly:
* the test split never influences fitting, tuning, selection or the recommendation;
* "difference not detected" is never reported as a tie; equivalence needs a margin;
* comparisons are Holm-corrected; bootstrap intervals have roughly nominal coverage and the
  difference test roughly nominal type-I error on synthetic data with known properties;
* group / time splits, near-duplicates, target leakage and group overlap are detected;
* exported raw predictions reproduce the reported metrics.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest
from conftest import wait_for

from app.arena import data as D
from app.arena import leakage as L
from app.arena.stats import (
    Policy,
    Resampling,
    bootstrap_cis,
    bootstrap_distributions,
    bootstrap_p_values,
    classification_metrics,
    compare,
    holm,
    make_resampling,
    regression_metrics,
)


def _csv_upload(c, name: str, df: pd.DataFrame, **form):
    buf = io.BytesIO(df.to_csv(index=False).encode())
    r = c.post("/api/arena/datasets", files={"file": (name, buf)}, data={k: str(v) for k, v in form.items()})
    return r


def _run(c, **body):
    r = c.post("/api/arena/runs", json={"cv_folds": 0, "bootstrap_resamples": 400, **body})
    assert r.status_code == 201, r.text
    rid = r.json()["id"]
    return wait_for(lambda: (d := c.get(f"/api/arena/runs/{rid}").json())["status"] not in ("queued", "running")
                    and d, timeout=180)


def _cand(key, y, pred, latency=1.0, source="builtin", labels=("a", "b")):
    return {"key": key, "name": key, "source": source, "pred": np.asarray(pred), "idx": np.arange(len(y)),
            "metrics": classification_metrics(y, pred, list(labels)), "latency_ms": latency, "size_bytes": 1000,
            "memory_mb": 10.0, "caveats": []}


# ----------------------------------------------------------------------------- statistics units
def test_holm_adjustment_matches_textbook_values():
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
    assert holm([0.5, None, 0.01]) == [0.5, None, 0.02]
    assert holm([]) == []


def test_difference_not_detected_is_inconclusive_not_a_tie():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 30)
    a = np.where(rng.random(30) < 0.8, y, 1 - y)
    b = np.where(rng.random(30) < 0.75, y, 1 - y)
    rs = make_resampling("classification", y, None, 1000, 0)
    s = compare("classification", [_cand("a", y, a), _cand("b", y, b, latency=0.5)], y, 2,
                Policy("accuracy", margin=0.02), rs)
    verdicts = {k: v["verdict"] for k, v in s["comparisons"].items()}
    assert "inconclusive" in verdicts.values()  # 30 examples cannot show a 2-point equivalence
    assert s["status"] == "provisional" and len(s["shortlist"]) == 2
    assert s["recommendation"]["key"] == s["leader"]  # an inconclusive model is never preferred for speed


def test_equivalence_needs_the_margin_and_enough_data():
    rng = np.random.default_rng(2)
    n = 3000
    y = rng.integers(0, 2, n)
    a = np.where(rng.random(n) < 0.85, y, 1 - y)
    b = np.where(rng.random(n) < 0.85, y, 1 - y)
    rs = make_resampling("classification", y, None, 1000, 0)
    wide = compare("classification", [_cand("a", y, a, 5.0), _cand("b", y, b, 1.0)], y, 2,
                   Policy("accuracy", margin=0.04), rs)
    other = "b" if wide["leader"] == "a" else "a"
    assert wide["comparisons"][other]["verdict"] == "equivalent"
    assert wide["recommendation"]["key"] == "b"  # equivalent and faster
    tight = compare("classification", [_cand("a", y, a, 5.0), _cand("b", y, b, 1.0)], y, 2,
                    Policy("accuracy", margin=0.001), rs)
    assert tight["comparisons"][other]["verdict"] != "equivalent"


def test_clear_difference_is_worse_and_holm_family_size():
    rng = np.random.default_rng(3)
    n = 400
    y = rng.integers(0, 2, n)
    good = np.where(rng.random(n) < 0.95, y, 1 - y)
    cands = [_cand("good", y, good)] + [_cand(f"weak{i}", y, np.where(rng.random(n) < 0.6, y, 1 - y), 0.1)
                                        for i in range(3)]
    s = compare("classification", cands, y, 2, Policy("f1_macro"), make_resampling("classification", y, None, 800, 0))
    assert s["leader"] == "good" and s["family_size"] == 3 and s["correction"] == "Holm"
    for i in range(3):
        c = s["comparisons"][f"weak{i}"]
        assert c["verdict"] == "worse" and c["p_diff_holm"] >= c["p_diff"]
    assert s["recommendation"]["key"] == "good" and s["status"] == "clear"


def test_constraints_exclude_models_with_reasons():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 200)
    best = np.where(rng.random(200) < 0.95, y, 1 - y)
    ok = np.where(rng.random(200) < 0.9, y, 1 - y)
    s = compare("classification", [_cand("slow", y, best, latency=50.0), _cand("quick", y, ok, latency=2.0)], y, 2,
                Policy("f1_macro", max_latency_ms=10), make_resampling("classification", y, None, 300, 0))
    assert "slow" in s["excluded"] and "latency" in s["excluded"]["slow"][0]
    assert s["recommendation"]["key"] == "quick"
    none = compare("classification", [_cand("slow", y, best, latency=50.0)], y, 2,
                   Policy("f1_macro", max_latency_ms=1, min_quality=0.99), make_resampling("classification", y, None, 300, 0))
    assert none["recommendation"] is None and none["excluded"]["slow"]


def test_baseline_is_never_recommended_and_flags_useless_features():
    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, 300)
    noise = rng.integers(0, 2, 300)
    majority = np.full(300, int(np.bincount(y).argmax()))
    s = compare("classification", [_cand("base", y, majority, source="baseline"), _cand("noise", y, noise)], y, 2,
                Policy("accuracy"), make_resampling("classification", y, None, 500, 0))
    assert s["recommendation"]["key"] == "noise"
    assert any("not detectably better than the trivial baseline" in w for w in s["warnings"])


def test_pareto_front():
    from app.arena.stats import _pareto

    rows = [{"key": k, "metrics": {"f1_macro": q}, "latency_ms": lat}
            for k, q, lat in (("a", 0.9, 10), ("b", 0.8, 1), ("c", 0.7, 5), ("d", 0.9, 12))]
    assert _pareto(rows, "f1_macro", True) == ["a", "b"]


def test_metric_edge_cases_are_defined():
    m = regression_metrics(np.array([0.0, 1, 2, 3]), np.array([0.1, 1, 2, 3]))
    assert m["mape"] is None  # a zero target makes MAPE undefined
    const = regression_metrics(np.array([2.0, 2, 2]), np.array([1.0, 2, 3]))
    assert const["r2"] is None and const["notes"]
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2])  # class 2 has one example
    ci = bootstrap_cis("classification", y, y, 3, make_resampling("classification", y, None, 300, 0))
    assert ci["f1_macro_ci"] == [1.0, 1.0]  # stratified resampling keeps every class present


@pytest.mark.parametrize("method", ["simple", "stratified"])
def test_bootstrap_interval_coverage_is_near_nominal(method):
    """95 % CI for accuracy of a classifier with known true accuracy 0.8 (n = 150)."""
    rng = np.random.default_rng(10)
    covered, sims = 0, 300
    for s in range(sims):
        y = rng.integers(0, 2, 150)
        pred = np.where(rng.random(150) < 0.8, y, 1 - y)
        rs = Resampling(method, 400, s, strata=y if method == "stratified" else None)
        ci = bootstrap_cis("classification", y, pred, 2, rs)["accuracy_ci"]
        covered += ci[0] <= 0.8 <= ci[1]
    assert 0.90 <= covered / sims <= 0.99


def test_difference_test_type_one_error_is_near_alpha():
    """Two models with the same true accuracy: the bootstrap test should reject ≈ 5 % of the time."""
    rng = np.random.default_rng(11)
    rejections, sims = 0, 300
    for s in range(sims):
        y = rng.integers(0, 2, 200)
        a = np.where(rng.random(200) < 0.8, y, 1 - y)
        b = np.where(rng.random(200) < 0.8, y, 1 - y)
        da, db = bootstrap_distributions("classification", y, [a, b], 2, Resampling("simple", 400, s), ["accuracy"])
        p, _ = bootstrap_p_values(da["accuracy"] - db["accuracy"], 0.02)
        rejections += p < 0.05
    assert rejections / sims <= 0.09


def test_cluster_bootstrap_widens_intervals_for_grouped_rows():
    rng = np.random.default_rng(12)
    groups = np.repeat(np.arange(40), 10)
    group_correct = rng.random(40) < 0.7  # all rows of a group are right or wrong together
    y = rng.integers(0, 2, 400)
    pred = np.where(group_correct[groups], y, 1 - y)
    simple = bootstrap_cis("classification", y, pred, 2, Resampling("simple", 800, 0))["accuracy_ci"]
    cluster = bootstrap_cis("classification", y, pred, 2, make_resampling("classification", y, groups, 800, 0))["accuracy_ci"]
    assert (cluster[1] - cluster[0]) > 2 * (simple[1] - simple[0])


# ----------------------------------------------------------------------------- selection invariance
def test_selection_never_uses_test_labels(make_app):
    """Changing every test label must not change the recommendation (only the final estimate)."""
    from app.arena.runner import ArenaManager
    from app.arena.service import LoadedDataset

    rng = np.random.default_rng(20)
    n = 300
    y = rng.integers(0, 2, n)
    train, val, test = np.arange(0, 100), np.arange(100, 200), np.arange(200, 300)
    preds = {"m1": np.where(rng.random(n) < 0.9, y, 1 - y), "m2": np.where(rng.random(n) < 0.7, y, 1 - y)}

    def summarize(y_all):
        data = LoadedDataset("tabular", "classification", ["a", "b"], None, None, np.array(["a", "b"])[y_all], train,
                             test, val_idx=val, split_sha256="x")
        evaluated = []
        for k, p in preds.items():
            splits = {}
            for part, rows in (("val", val), ("test", test)):
                splits[part] = {"pred": p[rows], "idx": np.arange(len(rows)),
                                "metrics": classification_metrics(y_all[rows], p[rows], ["a", "b"])}
            evaluated.append({"key": k, "name": k, "source": "builtin", "splits": splits, "latency_ms": 1.0,
                              "size_bytes": 1, "memory_mb": 1.0, "caveats": []})
        rs = {"val": make_resampling("classification", y_all[val], None, 300, 0),
              "test": make_resampling("classification", y_all[test], None, 300, 0)}
        return ArenaManager._summarize(None, data, {"bootstrap_resamples": 300}, evaluated, y_all, rs)  # type: ignore[arg-type]

    a = summarize(y)
    flipped = y.copy()
    flipped[test] = 1 - flipped[test]
    b = summarize(flipped)
    assert a["recommendation"] == b["recommendation"] and a["selection"] == b["selection"]
    assert a["final"]["value"] != b["final"]["value"]  # only the reported test estimate moves


def test_fit_and_tuning_only_ever_see_training_rows(client, monkeypatch):
    from sklearn.pipeline import Pipeline

    seen: list[set] = []
    original = Pipeline.fit

    def spy(self, X, y=None, **kw):
        if isinstance(X, pd.DataFrame):
            seen.append(set(X.index.tolist()))
        return original(self, X, y, **kw)

    monkeypatch.setattr(Pipeline, "fit", spy)
    from pathlib import Path

    with (Path(__file__).resolve().parents[2] / "samples" / "arena" / "customer_churn.csv").open("rb") as f:
        ds = client.post("/api/arena/datasets", files={"file": ("c.csv", f)}, data={"target": "churn"}).json()
    run = _run(client, dataset_id=ds["id"], algorithms=["logistic_regression", "decision_tree"], cv_folds=3, tune=True,
               tune_iterations=2)
    assert run["status"] == "completed"
    split = client.get(f"/api/arena/datasets/{ds['id']}/split.csv").text.splitlines()
    rows = [line.split(",") for line in split[2:]]
    train = {int(r[0]) for r in rows if r[1] == "train"}
    assert seen and all(s <= train for s in seen)  # no validation/test row ever reached fit()


# ----------------------------------------------------------------------------- splits & leakage
def _grouped_frame(n_groups=60, per=6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        base = rng.normal(size=2)
        label = "yes" if base[0] > 0 else "no"
        for _ in range(per):
            rows.append({"customer": f"C{g:03d}", "x1": base[0] + rng.normal(0, 0.05), "x2": base[1] + rng.normal(0, 0.05),
                         "label": label})
    return pd.DataFrame(rows)


def test_group_split_keeps_groups_together_and_uses_cluster_bootstrap(client):
    df = _grouped_frame()
    r = _csv_upload(client, "g.csv", df, target="label", split_strategy="group", group_column="customer")
    assert r.status_code == 201, r.text
    ds = r.json()
    assert ds["split_source"] == "group" and ds["group_column"] == "customer"
    assert "customer" not in ds["columns"]["features"]
    assert ds["profile"]["leakage"]["group_overlap"] == {"train-test": 0, "train-val": 0, "val-test": 0}
    run = _run(client, dataset_id=ds["id"], algorithms=["logistic_regression"])
    assert run["summary"]["selection"]["resampling"]["method"] == "cluster"

    rnd = _csv_upload(client, "r.csv", df, target="label").json()  # the random split mixes customers
    assert "customer" in rnd["columns"]["features"] or "customer" in rnd["columns"]["ignored"]
    r = _csv_upload(client, "bad.csv", df, target="label", split_strategy="group")
    assert r.status_code == 422


def test_time_split_puts_the_newest_rows_in_test(client):
    rng = np.random.default_rng(1)
    dates = pd.date_range("2024-01-01", periods=200, freq="D")
    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "x": rng.normal(size=200)})
    df["y"] = (df["x"] > 0).map({True: "up", False: "down"})
    df = df.sample(frac=1, random_state=0)  # file order is shuffled
    ds = _csv_upload(client, "t.csv", df, target="y", split_strategy="time", time_column="date").json()
    tr = ds["profile"]["time_range"]
    assert tr["train"][1] < tr["val"][0] <= tr["val"][1] < tr["test"][0]
    assert ds["n_test"] == 40 and ds["n_val"] == 40

    auto = _csv_upload(client, "t2.csv", df, target="y").json()
    assert auto["columns"]["types"]["date"] == "datetime" and "date" in auto["columns"]["ignored"]
    assert any("Date/time columns are not used" in w for w in auto["warnings"])


def test_target_leakage_probe_flags_a_column_that_encodes_the_outcome(client):
    rng = np.random.default_rng(2)
    n = 300
    x = rng.normal(size=n)
    y = np.where(x + rng.normal(0, 1.5, n) > 0, "churned", "stayed")
    df = pd.DataFrame({"x": x, "noise": rng.normal(size=n), "cancel_reason": np.where(y == "churned", "price", "none"),
                       "y": y})
    ds = _csv_upload(client, "leak.csv", df, target="y").json()
    probe = ds["profile"]["leakage"]["single_feature_probe"]
    assert [r["column"] for r in probe["suspicious"]] == ["cancel_reason"]
    assert any("Possible target leakage: column 'cancel_reason'" in w for w in ds["warnings"])


def test_text_near_duplicates_across_splits_are_reported():
    texts = [f"Order {i} arrived late and the box was damaged" for i in range(50)]
    texts += ["My invoice shows the wrong VAT number", "my invoice shows the WRONG vat number!!",
              "The app crashes when I open settings on Android", "the app crashes when i open the settings on android"]
    splits = {"train": list(range(0, 50)) + [50, 52], "val": [], "test": [51, 53]}
    rep = L.text_near_duplicates(texts, splits)
    assert rep["exact_normalized"]["train-test"] == 1  # case/punctuation only
    assert rep["near"]["train-test"] == 2  # plus one with a small wording change


def test_image_dhash_detects_resized_copies(tmp_path):
    from PIL import Image

    rng = np.random.default_rng(0)
    arr = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
    a, b, c = tmp_path / "a.png", tmp_path / "b.jpg", tmp_path / "c.png"
    Image.fromarray(arr).save(a)
    Image.fromarray(arr).resize((40, 40)).save(b, quality=85)
    Image.fromarray((rng.random((64, 64, 3)) * 255).astype(np.uint8)).save(c)
    hashes = [L.dhash(p) for p in (a, b, c)]
    rep = L.image_near_duplicates(hashes, {"train": [0], "val": [], "test": [1, 2]})
    assert rep["near"]["train-test"] == 1


def test_split_strategies_are_reproducible_and_hashed():
    y = np.array(["a", "b"] * 50)
    one = D.make_split3(y, "classification", 0.2, 0.2, 7)
    two = D.make_split3(y, "classification", 0.2, 0.2, 7)
    assert one == two and len(one[1]) == 20 and len(one[2]) == 20
    assert not (set(one[0]) & set(one[1]) or set(one[0]) & set(one[2]) or set(one[1]) & set(one[2]))
    assert D.split_sha256(*one) == D.split_sha256(*two) != D.split_sha256(*D.make_split3(y, "classification", 0.2, 0.2, 8))
    with pytest.raises(Exception):
        D.make_split3(y, "classification", 0.5, 0.4, 0)


# ----------------------------------------------------------------------------- legacy datasets, exploratory runs
def test_v11_dataset_gets_a_validation_split_without_touching_test(client):
    from pathlib import Path

    sample = Path(__file__).resolve().parents[2] / "samples" / "arena" / "house_prices.csv"
    with sample.open("rb") as f:
        ds = client.post("/api/arena/datasets", files={"file": ("h.csv", f)},
                         data={"target": "price_lakh_pkr", "val_size": "0"}).json()
    assert ds["n_val"] == 0
    before = client.get(f"/api/arena/datasets/{ds['id']}/split.csv").text.splitlines()[1:]
    test_before = {r for r in before if r.split(",")[1:2] == ["test"]}

    explo = _run(client, dataset_id=ds["id"], algorithms=["ridge"], add_validation_split=False)
    assert explo["summary"]["final"]["unbiased"] is False
    assert any("exploratory" in w for w in explo["summary"]["warnings"])

    run = _run(client, dataset_id=ds["id"], algorithms=["ridge"])
    after = client.get(f"/api/arena/datasets/{ds['id']}/split.csv").text.splitlines()[1:]
    assert {r for r in after if r.split(",")[1:2] == ["test"]} == test_before
    assert run["dataset"]["n_val"] > 0 and run["summary"]["final"]["unbiased"] is True
    upgraded = client.get(f"/api/arena/datasets/{ds['id']}").json()
    assert any("Validation split added in v1.2" in w for w in upgraded["warnings"])


# ----------------------------------------------------------------------------- provenance & reproducibility
def _meta_onnx(path, metadata: dict):
    import json

    import onnx
    from onnx import TensorProto, helper, numpy_helper

    W = np.array([[1.0, -1.0], [0.0, 0.0]], dtype=np.float32)
    nodes = [helper.make_node("Gemm", ["x", "W"], ["logits"]), helper.make_node("Softmax", ["logits"], ["p"], axis=1)]
    g = helper.make_graph(nodes, "m", [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 2])],
                          [helper.make_tensor_value_info("p", TensorProto.FLOAT, ["N", 2])], [numpy_helper.from_array(W, "W")])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    for k, v in metadata.items():
        e = m.metadata_props.add()
        e.key, e.value = f"local_ai_lab.{k}", v if isinstance(v, str) else json.dumps(v)
    onnx.save(m, path)
    return path


def test_onnx_metadata_fills_labels_and_declares_provenance(client, tmp_path):
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
    df["y"] = np.where(df["a"] > 0, "pos", "neg")
    ds = _csv_upload(client, "m.csv", df, target="y").json()
    path = _meta_onnx(tmp_path / "m.onnx", {"labels": ["pos", "neg"], "kind": "tabular", "task": "classification",
                                            "feature_columns": ["a", "b"], "trained_on": "train",
                                            "split_sha256": ds["split_sha256"], "dataset_sha256": ds["sha256"]})
    with path.open("rb") as f:
        m = client.post("/api/arena/models", files={"file": ("m.onnx", f)},
                        data={"kind": "tabular", "task": "classification"}).json()
    assert m["config"]["labels"] == ["pos", "neg"] and m["config"]["feature_columns"] == ["a", "b"]
    assert m["provenance"]["training_data"] == "train_split" and set(m["inspection"]["defaults_from_metadata"]) == {
        "labels", "feature_columns"}
    run = _run(client, dataset_id=ds["id"], uploaded_models=[m["id"]], algorithms=["logistic_regression"])
    up = next(r for r in run["results"] if r["source"] == "uploaded")
    assert up["status"] == "ok" and up["details"]["caveats"] == []  # split hash matches the declaration
    assert up["metrics"]["accuracy"] == 1.0

    with path.open("rb") as f:
        wrong = client.post("/api/arena/models", files={"file": ("m.onnx", f)},
                            data={"kind": "image", "task": "classification"})
    assert wrong.status_code == 422 and "metadata says" in wrong.json()["error"]["message"]


def test_exported_predictions_reproduce_reported_metrics(client):
    """FR-06: aggregates recomputed from raw per-example predictions match the stored/exported metrics."""
    from pathlib import Path

    sample = Path(__file__).resolve().parents[2] / "samples" / "arena" / "customer_churn.csv"
    with sample.open("rb") as f:
        ds = client.post("/api/arena/datasets", files={"file": ("c.csv", f)}, data={"target": "churn"}).json()
    run = _run(client, dataset_id=ds["id"], algorithms=["logistic_regression", "naive_bayes"])
    full = client.get(f"/api/arena/runs/{run['id']}/export?format=json").json()
    split = [line.split(",") for line in client.get(f"/api/arena/datasets/{ds['id']}/split.csv").text.splitlines()[2:]]
    y_all = pd.read_csv(sample)["churn"].astype(str).to_numpy()
    labels = ds["labels"]
    for part, pred_key, idx_key, metrics_key in (("test", "predictions", "evaluated_idx", "metrics"),
                                                 ("val", "val_predictions", "val_idx", "val_metrics")):
        rows = np.array([int(r[0]) for r in split if r[1] == part])
        y = np.array([labels.index(v) for v in y_all[rows]])
        for res in full["results"]:
            pred = np.array(res[pred_key])
            recomputed = classification_metrics(y[np.array(res[idx_key])], pred, labels)
            for k in ("accuracy", "f1_macro", "balanced_accuracy"):
                assert recomputed[k] == res[metrics_key][k], (part, res["key"], k)
    csv_text = client.get(f"/api/arena/runs/{run['id']}/export?format=csv").text
    lr = next(r for r in full["results"] if r["key"] == "builtin:logistic_regression")
    assert str(lr["metrics"]["f1_macro"]) in csv_text and str(lr["val_metrics"]["f1_macro"]) in csv_text
    md = client.get(f"/api/arena/runs/{run['id']}/export?format=md").text
    assert f"{lr['val_metrics']['f1_macro']:.3f}" in md
