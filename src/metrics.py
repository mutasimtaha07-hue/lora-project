"""Evaluation metrics: accuracy, F1, precision, recall.

`average="binary"` is appropriate for SST-2 (2 classes); AG News (4 classes)
should use `average="macro"` — see `DatasetSpec.metric_average` in
`src/data.py`, which each dataset's config already carries.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_classification_metrics(
    preds: np.ndarray, labels: np.ndarray, average: str = "binary"
) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average=average, zero_division=0),
        "precision": precision_score(labels, preds, average=average, zero_division=0),
        "recall": recall_score(labels, preds, average=average, zero_division=0),
    }


def make_compute_metrics_fn(average: str = "binary"):
    """Build a `transformers.Trainer`-compatible `compute_metrics` callable.

    `Trainer` calls this with an `EvalPrediction`-like object exposing
    `.predictions` (raw logits) and `.label_ids`.
    """

    def compute_metrics(eval_pred) -> dict[str, float]:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return compute_classification_metrics(preds, labels, average=average)

    return compute_metrics
