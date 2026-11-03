"""Local embedding backends.

* :class:`OllamaEmbedder` – a neural embedding model served by the *local*
  Ollama runtime (default ``nomic-embed-text``). Never uses cloud endpoints.
* :class:`HashingEmbedder` – a fully offline lexical fallback (scikit-learn
  ``HashingVectorizer``). No model download needed; works on any machine but
  only matches overlapping words, not meaning.
* :class:`HFEmbedder` (optional) – a Hugging Face Transformers encoder loaded
  strictly from local files (``local_files_only=True``); requires the optional
  ``requirements-ml.txt`` (torch + transformers).

Each embedder exposes a stable ``key`` that is stored with every document, so
queries are only compared with chunks embedded by the same model.
"""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

from app.services.ollama_client import OllamaClient

log = logging.getLogger("app.embeddings")


class Embedder(Protocol):
    key: str
    backend: str
    model: str
    default_min_score: float

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _normalize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    if m.ndim == 1:
        m = m[None, :]
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class HashingEmbedder:
    backend = "hashing"
    model = "hashing-v1"
    key = "hashing:hashing-v1"  # always "<backend>:<model>"
    default_min_score = 0.08
    DIM = 2048

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import HashingVectorizer

        self._vec = HashingVectorizer(
            n_features=self.DIM, alternate_sign=False, norm="l2", ngram_range=(1, 2),
            lowercase=True, stop_words="english", token_pattern=r"(?u)\b\w\w+\b",
        )

    def _embed(self, texts: list[str]) -> np.ndarray:
        return _normalize(self._vec.transform(texts).toarray())

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text])[0]


class OllamaEmbedder:
    backend = "ollama"
    default_min_score = 0.30
    BATCH = 16

    def __init__(self, client: OllamaClient, model: str, timeout_s: float = 120.0) -> None:
        self.client = client
        self.model = model
        self.key = f"ollama:{model}"
        self.timeout_s = timeout_s
        # nomic-embed-text is trained with task prefixes
        self._nomic = model.startswith("nomic-embed")

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        out: list[list[float]] = []
        prefix = "search_document: " if self._nomic else ""
        for i in range(0, len(texts), self.BATCH):
            batch = [prefix + t for t in texts[i:i + self.BATCH]]
            out.extend(self.client.embed(self.model, batch, timeout=self.timeout_s))
        return _normalize(np.array(out, dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        prefix = "search_query: " if self._nomic else ""
        return _normalize(np.array(self.client.embed(self.model, [prefix + text], timeout=self.timeout_s)))[0]


class HFEmbedder:  # pragma: no cover - optional heavy dependency, see docs
    backend = "huggingface"
    default_min_score = 0.30

    def __init__(self, model_path: str) -> None:
        import torch  # noqa: F401
        from transformers import AutoModel, AutoTokenizer

        self.model = model_path
        self.key = f"hf:{model_path}"
        self._tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self._model = AutoModel.from_pretrained(model_path, local_files_only=True).eval()

    def _embed(self, texts: list[str]) -> np.ndarray:
        import torch

        with torch.no_grad():
            enc = self._tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
            out = self._model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)  # mean pooling
        return _normalize(pooled.numpy())

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text])[0]


def to_bytes(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def from_bytes(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.float32)
