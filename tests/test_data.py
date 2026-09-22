"""Tests for src/data.py's dataset registry.

Only the registry metadata (`DatasetSpec` values and `get_spec`) is tested
here -- no dataset is downloaded and no tokenizer is loaded, so this file has
no network dependency and does not touch the Hugging Face cache. Tokenization
behavior (`tokenize_dataset`) is intentionally out of scope for this suite,
since exercising it correctly requires a real tokenizer.
"""

from __future__ import annotations

import pytest

from src.data import DATASET_REGISTRY, get_spec


def test_sst2_is_registered_as_binary_classification():
    spec = DATASET_REGISTRY["sst2"]

    assert spec.num_labels == 2
    assert spec.metric_average == "binary"
    assert spec.text_column == "sentence"
    assert spec.label_column == "label"
    assert spec.train_split == "train"
    # GLUE's SST-2 "test" split is unlabeled; "validation" is the labeled
    # held-out split this project evaluates on.
    assert spec.eval_split == "validation"
    assert spec.hf_path == "nyu-mll/glue"
    assert spec.hf_name == "sst2"


def test_ag_news_is_registered_as_four_class_classification():
    spec = DATASET_REGISTRY["ag_news"]

    assert spec.num_labels == 4
    assert spec.metric_average == "macro"
    assert spec.text_column == "text"
    assert spec.label_column == "label"
    assert spec.train_split == "train"
    # AG News ships only train/test; "test" is the split used here for both
    # model selection and final reporting (see report/final_report.md's
    # documented limitation on this).
    assert spec.eval_split == "test"
    assert spec.hf_path == "fancyzhx/ag_news"
    assert spec.hf_name is None


def test_registry_contains_exactly_the_two_expected_datasets():
    assert set(DATASET_REGISTRY.keys()) == {"sst2", "ag_news"}


def test_get_spec_returns_the_same_object_as_direct_registry_lookup():
    assert get_spec("sst2") is DATASET_REGISTRY["sst2"]
    assert get_spec("ag_news") is DATASET_REGISTRY["ag_news"]


def test_get_spec_raises_for_an_unknown_dataset_key():
    # `src/train.py`'s argparse `choices=list(DATASET_REGISTRY.keys())`
    # relies on unknown keys being rejected rather than silently accepted.
    with pytest.raises(KeyError):
        get_spec("not_a_real_dataset")
