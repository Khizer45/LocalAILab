from __future__ import annotations

from conftest import wait_for

from app.core.config import BACKEND_DIR
from app.ml.intent_data import load_split
from app.services.chunker import chunk_pages, normalize
from app.services.document_parser import decode_text, detect_type
from app.services.model_registry import is_cloud_tag


# ------------------------------------------------------------------ ML baseline
def test_split_is_stratified_and_disjoint():
    split = load_split(BACKEND_DIR / "datasets")
    assert len(split.labels) == 6
    tr, va, te = ({t for t, _ in s} for s in (split.train, split.val, split.test))
    assert not (tr & va) and not (tr & te) and not (va & te)
    counts = split.summary()["class_counts"]["test"]
    assert set(counts) == set(split.labels) and min(counts.values()) >= 3
    assert load_split(BACKEND_DIR / "datasets").test == split.test  # deterministic


def test_train_predict_and_compare(client):
    r = client.post("/api/ml/baseline/train")
    assert r.status_code == 201, r.text
    m = r.json()["metrics"]
    assert m["model_selection"]["selected"] in {"logistic_regression", "linear_svm", "naive_bayes"}
    test = m["test"]
    assert test["accuracy"] >= 0.6 and 0 <= test["f1_macro"] <= 1
    cm = test["confusion_matrix"]["matrix"]
    assert len(cm) == 6 and sum(map(sum, cm)) == test["n"] == 23
    assert set(test["per_class"]) == set(m["dataset"]["labels"])

    p = client.post("/api/ml/baseline/predict", json={"text": "I was charged twice on my card"}).json()
    assert p["label"] == "billing" and p["latency_ms"] >= 0

    run = client.post("/api/benchmarks/runs", json={"suite_id": "intent-classification", "models": ["sim-small:1b"],
                                                    "warmup": False}).json()
    wait_for(lambda: client.get(f"/api/benchmarks/runs/{run['id']}").json()["status"] == "completed", timeout=60)
    comp = client.get("/api/ml/comparison").json()
    approaches = {row["approach"] for row in comp["rows"]}
    assert approaches == {"Traditional ML (scikit-learn)", "LLM zero-shot (deep learning)"}
    llm = next(r for r in comp["rows"] if r["approach"].startswith("LLM"))
    assert llm["n"] == 23 and 0 <= llm["f1_macro"] <= 1


def test_predict_before_training(client):
    r = client.post("/api/ml/baseline/predict", json={"text": "hello"})
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_trained"


def test_ml_module_can_be_disabled(make_app):
    c = make_app(ml_baseline_enabled=False)
    assert c.get("/api/ml/dataset").status_code == 404


# ------------------------------------------------------------------ parsing / chunking
def test_chunker_respects_pages_size_and_overlap():
    text = " ".join(f"Sentence number {i} talks about local models." for i in range(120))
    chunks = chunk_pages([(1, text), (2, "Short second page.")], chunk_size=400, overlap=80)
    assert all(len(c.text) <= 480 for c in chunks)
    assert chunks[-1].page == 2 and chunks[-1].text == "Short second page."
    page1 = [c.text for c in chunks if c.page == 1]
    assert len(page1) > 3
    tail = page1[0][-40:].split(" ", 1)[1]
    assert tail[:15] in page1[1]  # overlap carried into the next chunk


def test_normalize_text():
    assert normalize("infor-\nmation  is\t\tkey\x00") == "information is key"


def test_detect_type_and_decode():
    assert detect_type("a.PDF", b"%PDF-1.4") == "pdf"
    assert detect_type("notes.markdown", b"# hi") == "md"
    assert decode_text("héllo".encode("utf-16")) == "héllo"
    assert decode_text("héllo".encode("latin-1")) == "héllo"


def test_cloud_tag_detection():
    assert is_cloud_tag("gemma4:cloud") and is_cloud_tag("gpt-oss:120b-cloud")
    assert is_cloud_tag("x:1b", {"remote_host": "https://ollama.com"})
    assert not is_cloud_tag("llama3.2:3b") and not is_cloud_tag("nomic-embed-text")


def test_env_lists_accept_comma_separated_values(monkeypatch):
    from app.core.config import Settings

    monkeypatch.setenv("OFFLINE_ALLOWED_HOSTS", "localhost,127.0.0.1,::1,ollama")
    monkeypatch.setenv("CORS_ORIGINS", '["http://127.0.0.1:8080"]')
    s = Settings(_env_file=None)
    assert s.offline_allowed_hosts == ["localhost", "127.0.0.1", "::1", "ollama"]
    assert s.cors_origins == ["http://127.0.0.1:8080"]


def test_env_example_parses(monkeypatch):
    from app.core.config import REPO_DIR, Settings

    s = Settings(_env_file=str(REPO_DIR / ".env.example"))
    assert s.offline_only_default is True and s.auth_enabled is False
    assert "::1" in s.offline_allowed_hosts and s.ollama_api_key in (None, "")


def test_numeric_evaluator_understands_number_words_and_answer_markers():
    from app.benchmark.evaluators import numeric

    assert numeric("There are seven continents.", {"value": 7}).passed
    assert numeric("twenty-one apples and three hundred pears", {"value": 300}).passed
    assert numeric("Answer: 42 (it took 3 steps)", {"value": 42}).passed
    assert numeric("The answer is 8, not 9.", {"value": 8}).passed
    assert not numeric("No number here", {"value": 1}).passed
