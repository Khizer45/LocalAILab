"""Versioned intent-classification dataset and its deterministic split.

The same test split is used by:
* the scikit-learn baseline (traditional ML), and
* the ``intent-classification`` benchmark suite (LLM zero-shot classification),
so both approaches are scored on exactly the same examples.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

from sklearn.model_selection import train_test_split

DATASET_ID = "intent"
DATASET_VERSION = "1.0.0"
SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15


@dataclass
class IntentSplit:
    labels: list[str]
    train: list[tuple[str, str]]
    val: list[tuple[str, str]]
    test: list[tuple[str, str]]
    sha256: str

    def summary(self) -> dict:
        def counts(rows):
            out: dict[str, int] = {}
            for _, y in rows:
                out[y] = out.get(y, 0) + 1
            return out

        return {
            "dataset_id": DATASET_ID, "version": DATASET_VERSION, "sha256": self.sha256, "seed": SEED,
            "labels": self.labels, "sizes": {"train": len(self.train), "val": len(self.val), "test": len(self.test)},
            "class_counts": {"train": counts(self.train), "val": counts(self.val), "test": counts(self.test)},
        }


def dataset_path(datasets_dir: Path) -> Path:
    return datasets_dir / "intent" / "intent_v1.csv"


def load_split(datasets_dir: Path) -> IntentSplit:
    path = dataset_path(datasets_dir)
    raw = path.read_bytes()
    with path.open(encoding="utf-8", newline="") as f:
        rows = [(r["text"].strip(), r["label"].strip()) for r in csv.DictReader(f) if r.get("text")]
    texts, labels = [r[0] for r in rows], [r[1] for r in rows]
    x_rest, x_test, y_rest, y_test = train_test_split(texts, labels, test_size=TEST_SIZE, random_state=SEED,
                                                      stratify=labels)
    val_frac = VAL_SIZE / (1 - TEST_SIZE)
    x_train, x_val, y_train, y_val = train_test_split(x_rest, y_rest, test_size=val_frac, random_state=SEED,
                                                      stratify=y_rest)
    return IntentSplit(
        labels=sorted(set(labels)),
        train=list(zip(x_train, y_train, strict=True)),
        val=list(zip(x_val, y_val, strict=True)),
        test=list(zip(x_test, y_test, strict=True)),
        sha256=hashlib.sha256(raw).hexdigest(),
    )
