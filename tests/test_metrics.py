"""Tests for src/metrics.py.

These exercise `compute_classification_metrics` and `make_compute_metrics_fn`
against small, hand-picked toy arrays -- no model, no GPU, no downloaded data.
Expected values are cross-checked against scikit-learn's own functions called
directly with the same inputs, which is exactly what `src/metrics.py` wraps;
this catches wiring bugs (e.g. swapped arguments, wrong `average`) rather than
re-testing scikit-learn's own math.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.metrics import compute_classification_metrics, make_compute_metrics_fn

# --- Binary (SST-2-like) toy data -------------------------------------------
# Deliberately asymmetric: 2 false negatives, 1 false positive, so precision
# and recall come out different -- this would catch a bug that swapped
# `preds`/`labels` or swapped precision/recall internally.
BINARY_LABELS = np.array([1, 1, 1, 1, 0, 0, 0, 0])
BINARY_PREDS = np.array([1, 1, 0, 0, 0, 0, 0, 1])


def test_binary_metrics_match_sklearn_binary_average():
    result = compute_classification_metrics(BINARY_PREDS, BINARY_LABELS, average="binary")

    assert result["accuracy"] == pytest.approx(accuracy_score(BINARY_LABELS, BINARY_PREDS))
    assert result["f1"] == pytest.approx(f1_score(BINARY_LABELS, BINARY_PREDS, average="binary"))
    assert result["precision"] == pytest.approx(
        precision_score(BINARY_LABELS, BINARY_PREDS, average="binary")
    )
    assert result["recall"] == pytest.approx(
        recall_score(BINARY_LABELS, BINARY_PREDS, average="binary")
    )
    # Hand-verified sanity check: 5 of 8 predictions are correct.
    assert result["accuracy"] == pytest.approx(0.625)


def test_binary_metrics_precision_and_recall_are_not_swapped():
    result = compute_classification_metrics(BINARY_PREDS, BINARY_LABELS, average="binary")
    # Hand-verified: TP=2, FP=1, FN=2 -> precision=2/3, recall=2/4=0.5.
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(0.5)


# --- Macro / multiclass (AG-News-like, 4 classes) toy data ------------------
MULTI_LABELS = np.array([0, 0, 1, 1, 2, 2, 3, 3])
MULTI_PREDS = np.array([0, 1, 1, 1, 2, 3, 3, 3])


def test_macro_metrics_match_sklearn_macro_average():
    result = compute_classification_metrics(MULTI_PREDS, MULTI_LABELS, average="macro")

    assert result["accuracy"] == pytest.approx(accuracy_score(MULTI_LABELS, MULTI_PREDS))
    assert result["f1"] == pytest.approx(f1_score(MULTI_LABELS, MULTI_PREDS, average="macro"))
    assert result["precision"] == pytest.approx(
        precision_score(MULTI_LABELS, MULTI_PREDS, average="macro")
    )
    assert result["recall"] == pytest.approx(
        recall_score(MULTI_LABELS, MULTI_PREDS, average="macro")
    )
    # Hand-verified sanity check: 6 of 8 predictions are correct.
    assert result["accuracy"] == pytest.approx(0.75)


def test_average_argument_is_actually_threaded_through():
    # scikit-learn's average="binary" requires a strictly 2-class target and
    # raises ValueError on this 4-class toy data. If `compute_classification_
    # metrics` ever hardcoded "binary" (ignoring its `average` parameter),
    # the macro-average call below would raise instead of the call above
    # succeeding, or vice versa -- this proves the argument genuinely
    # controls behavior rather than being silently ignored.
    with pytest.raises(ValueError):
        compute_classification_metrics(MULTI_PREDS, MULTI_LABELS, average="binary")


def test_zero_division_suppresses_undefined_metric_warning():
    # Class "1" is never predicted, so its precision/recall/F1 are
    # mathematically undefined (0/0).
    #
    # NOTE on what this actually tests: in the installed scikit-learn
    # version, *omitting* zero_division=0 here would still return the same
    # numeric values asserted below -- scikit-learn's default already
    # substitutes 0.0 for the undefined class, it just also raises an
    # UndefinedMetricWarning while doing so. So the numeric assertions alone
    # (kept below for correctness) would NOT catch `zero_division=0` being
    # accidentally removed from src/metrics.py. The actual, real contract
    # `zero_division=0` provides is silence: it suppresses that warning.
    # `simplefilter("error")` below promotes any warning raised inside the
    # `with` block into an exception, so if `zero_division=0` were removed
    # from src/metrics.py, scikit-learn's UndefinedMetricWarning would fail
    # this test outright, exactly as a real regression should.
    labels = np.array([0, 0, 1, 1])
    preds = np.array([0, 0, 0, 0])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = compute_classification_metrics(preds, labels, average="macro")

    assert result["accuracy"] == pytest.approx(0.5)
    assert result["precision"] == pytest.approx(
        precision_score(labels, preds, average="macro", zero_division=0)
    )
    assert result["recall"] == pytest.approx(
        recall_score(labels, preds, average="macro", zero_division=0)
    )
    assert result["f1"] == pytest.approx(
        f1_score(labels, preds, average="macro", zero_division=0)
    )


# --- make_compute_metrics_fn: the Trainer-facing wrapper --------------------


def test_make_compute_metrics_fn_argmaxes_logits_and_applies_average():
    # Mimics what `transformers.Trainer` hands to `compute_metrics`: an
    # (logits, labels) pair, unpacked by `src/metrics.py` as
    # `logits, labels = eval_pred`. A plain tuple satisfies that unpacking.
    logits = np.array(
        [
            [2.0, -1.0],  # -> predicted class 0
            [0.1, 0.2],  # -> predicted class 1
            [-3.0, 1.0],  # -> predicted class 1
            [1.0, -1.0],  # -> predicted class 0
        ]
    )
    labels = np.array([0, 1, 1, 1])
    compute_metrics = make_compute_metrics_fn(average="binary")

    result = compute_metrics((logits, labels))

    expected_preds = np.argmax(logits, axis=-1)
    expected = compute_classification_metrics(expected_preds, labels, average="binary")
    assert result == pytest.approx(expected)


def test_make_compute_metrics_fn_uses_macro_average_when_configured():
    # Same idea, but for the 4-class (AG-News-like) configuration -- confirms
    # the `average` parameter passed to the factory actually reaches the
    # inner `compute_metrics` closure.
    logits = np.array(
        [
            [3.0, 0.0, 0.0, 0.0],  # -> 0
            [0.0, 3.0, 0.0, 0.0],  # -> 1
            [0.0, 0.0, 3.0, 0.0],  # -> 2
            [0.0, 0.0, 0.0, 3.0],  # -> 3
        ]
    )
    labels = np.array([0, 1, 2, 3])
    compute_metrics = make_compute_metrics_fn(average="macro")

    result = compute_metrics((logits, labels))

    assert result["accuracy"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(1.0)
