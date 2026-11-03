"""Retrieval-quality evaluation for document Q&A (Recall@k, MRR, nDCG@k, abstention).

The "context grounding" benchmark suite tests whether a *model* answers from
context it is given. This module tests the *search* that finds that context:
for a set of questions with known relevant passages it measures how often and
how high the relevant chunks are ranked.

Relevance is judged deterministically: a retrieved chunk is relevant to a
target when it comes from the named file (``filename`` or ``document_id``, if
given), from the given ``page`` (if given) and contains the ``contains`` text
(case- and whitespace-insensitive, if given).

Questions with no relevant passage (``"relevant": []``) measure abstention:
the share of them for which retrieval still returns chunks above the score
threshold – those are the questions where the assistant would answer instead of
saying it cannot find the information.
"""

from __future__ import annotations

import math
import re

import numpy as np
from pydantic import BaseModel, Field

from app.core.errors import ValidationFailed

_WS = re.compile(r"\s+")


class RelevantPassage(BaseModel):
    filename: str | None = Field(None, max_length=255)
    document_id: int | None = None
    page: int | None = None
    contains: str | None = Field(None, max_length=1000)


class RetrievalQuestion(BaseModel):
    id: str = Field(max_length=100)
    question: str = Field(min_length=1, max_length=2000)
    relevant: list[RelevantPassage] = Field(default_factory=list, max_length=20)


class RetrievalEvalRequest(BaseModel):
    name: str | None = Field(None, max_length=200)
    questions: list[RetrievalQuestion] = Field(min_length=1, max_length=1000)
    k_values: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    document_ids: list[int] | None = None


def _norm(s: str) -> str:
    return _WS.sub(" ", s).strip().lower()


def _matches(chunk: dict, target: RelevantPassage) -> bool:
    if target.document_id is not None and chunk["document_id"] != target.document_id:
        return False
    if target.filename and _norm(chunk["filename"]) != _norm(target.filename):
        return False
    if target.page is not None and chunk.get("page") != target.page:
        return False
    if target.contains and _norm(target.contains) not in _norm(chunk["text"]):
        return False
    return True


def _ci(values: list[float], seed: int = 0) -> list[float] | None:
    if len(values) < 5:
        return None
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, len(arr), size=(2000, len(arr)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def evaluate(documents, owner_id: int, req: RetrievalEvalRequest) -> dict:
    """``documents`` is the DocumentService (its ``retrieve`` is the production search)."""
    ks = sorted({k for k in req.k_values if 1 <= k <= 50}) or [5]
    kmax = max(ks)
    rows: list[dict] = []
    embedder = None
    min_score = None
    for q in req.questions:
        ranked = documents.retrieve(owner_id, q.question, top_k=kmax, document_ids=req.document_ids,
                                    min_score=-math.inf)
        embedder, threshold = ranked["embedder"], ranked["min_score"]
        min_score = threshold
        chunks = ranked["sources"]
        above = [c for c in chunks if c["score"] >= threshold]
        row: dict = {"id": q.id, "question": q.question, "answerable": bool(q.relevant),
                     "retrieved": [{"rank": i + 1, "filename": c["filename"], "page": c["page"],
                                    "chunk_index": c["chunk_index"], "score": c["score"],
                                    "relevant": any(_matches(c, t) for t in q.relevant),
                                    "above_threshold": c["score"] >= threshold} for i, c in enumerate(chunks)],
                     "returned_in_production": len(above)}
        if q.relevant:
            first = next((i + 1 for i, c in enumerate(chunks) if any(_matches(c, t) for t in q.relevant)), None)
            row["first_relevant_rank"] = first
            row["reciprocal_rank"] = 1.0 / first if first else 0.0
            for k in ks:
                top = chunks[:k]
                found = [any(_matches(c, t) for c in top) for t in q.relevant]
                row[f"recall@{k}"] = sum(found) / len(q.relevant)
                row[f"hit@{k}"] = 1.0 if any(found) else 0.0
                gains = [1.0 if any(_matches(c, t) for t in q.relevant) else 0.0 for c in top]
                dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
                ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(q.relevant))))
                row[f"ndcg@{k}"] = dcg / ideal if ideal else 0.0
            row["relevant_found_in_production"] = any(any(_matches(c, t) for t in q.relevant) for c in above)
        else:
            row["false_retrieval"] = bool(above)
        rows.append(row)
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    if not rows:
        raise ValidationFailed("No questions to evaluate.")
    summary: dict = {"questions": len(rows), "answerable": len(answerable), "unanswerable": len(unanswerable),
                     "embedder": embedder, "min_score": min_score, "k_values": ks}
    if answerable:
        summary["mrr"] = round(float(np.mean([r["reciprocal_rank"] for r in answerable])), 4)
        summary["mrr_ci"] = _ci([r["reciprocal_rank"] for r in answerable])
        for k in ks:
            for m in ("recall", "hit", "ndcg"):
                vals = [r[f"{m}@{k}"] for r in answerable]
                summary[f"{m}@{k}"] = round(float(np.mean(vals)), 4)
            summary[f"recall@{k}_ci"] = _ci([r[f"recall@{k}"] for r in answerable])
        summary["answer_context_rate"] = round(float(np.mean([r["relevant_found_in_production"]
                                                              for r in answerable])), 4)
    if unanswerable:
        summary["false_retrieval_rate"] = round(float(np.mean([r["false_retrieval"] for r in unanswerable])), 4)
    notes = ["Relevance is decided by the 'contains' text / file / page you gave, not by a model.",
             f"Computed on {len(answerable)} answerable questions; with fewer than ~50 questions the intervals are "
             "wide."]
    if not answerable:
        notes.append("No answerable questions: only abstention was measured.")
    return {"summary": summary, "questions": rows, "notes": notes}
