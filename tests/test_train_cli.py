"""Tests for src/train.py's pure CLI/naming/subsetting logic.

`main()` itself is intentionally NOT tested here: it launches a real
`transformers.Trainer` training loop, which needs a downloaded model and
dataset and real training time -- exactly the kind of expensive integration
test this suite avoids. Everything below is pure argument parsing, string
formatting, or in-memory list-slicing logic that runs in a fraction of a
second with no external dependencies.

`parse_args` flag names/defaults are checked against both `src/train.py`'s
argparse definitions and the exact reproduction commands documented in
README.md's "Reproducibility" section, so these tests double as a guard
against the CLI and the README silently drifting apart.
"""

from __future__ import annotations

import argparse

import pytest
from datasets import Dataset

from src.train import build_run_name, default_lr, limit_split, parse_args

# --- default_lr --------------------------------------------------------------


def test_default_lr_for_lora_and_full_finetuning():
    # This exact pair of constants is what the project's own documented
    # incident (an AG News full-fine-tuning run collapsing at lr=2e-4)
    # hinges on -- see report/final_report.md Section 12.
    assert default_lr("lora") == pytest.approx(2e-4)
    assert default_lr("full") == pytest.approx(2e-5)


# --- build_run_name ------------------------------------------------------------


def test_build_run_name_for_lora():
    args = argparse.Namespace(method="lora", dataset="sst2", rank=8, seed=42)
    assert build_run_name(args) == "sst2_lora_r8_seed42"


def test_build_run_name_for_full_finetuning():
    args = argparse.Namespace(method="full", dataset="ag_news", seed=42)
    assert build_run_name(args) == "ag_news_full_seed42"


# --- parse_args ------------------------------------------------------------


def test_parse_args_matches_the_documented_sst2_lora_reproduction_command():
    # Mirrors README.md's "SST-2, LoRA" reproduction command verbatim.
    argv = [
        "--dataset", "sst2", "--method", "lora", "--rank", "8", "--seed", "42",
        "--model-name", "roberta-base", "--epochs", "3", "--batch-size", "32",
        "--eval-batch-size", "64", "--lr", "2e-4", "--weight-decay", "0.01",
        "--max-length", "128", "--checkpoints-dir", "checkpoints",
        "--results-dir", "results/raw",
    ]
    args = parse_args(argv)

    assert args.dataset == "sst2"
    assert args.method == "lora"
    assert args.rank == 8
    assert args.seed == 42
    assert args.model_name == "roberta-base"
    assert args.epochs == pytest.approx(3.0)
    assert args.batch_size == 32
    assert args.eval_batch_size == 64
    assert args.lr == pytest.approx(2e-4)
    assert args.weight_decay == pytest.approx(0.01)
    assert args.max_length == 128
    assert args.checkpoints_dir == "checkpoints"
    assert args.results_dir == "results/raw"


def test_parse_args_matches_the_documented_ag_news_full_finetuning_command():
    # Mirrors README.md's "AG News, full fine-tuning baseline" command --
    # note lr=2e-5, not 2e-4 (see the collapse incident referenced above).
    argv = [
        "--dataset", "ag_news", "--method", "full", "--seed", "42",
        "--model-name", "roberta-base", "--epochs", "3", "--batch-size", "32",
        "--eval-batch-size", "64", "--lr", "2e-5", "--weight-decay", "0.01",
        "--max-length", "128", "--checkpoints-dir", "checkpoints",
        "--results-dir", "results/raw",
    ]
    args = parse_args(argv)

    assert args.dataset == "ag_news"
    assert args.method == "full"
    assert args.seed == 42
    assert args.lr == pytest.approx(2e-5)


def test_parse_args_defaults_match_documented_lora_defaults():
    # With no flags at all, parse_args should reproduce the defaults the
    # README describes as the LoRA-sweep starting point (rank 8, seed 42).
    # --lr defaults to None here; main() resolves it via default_lr(method).
    args = parse_args([])

    assert args.dataset == "sst2"
    assert args.method == "lora"
    assert args.rank == 8
    assert args.seed == 42
    assert args.lr is None
    assert args.epochs == pytest.approx(3.0)
    assert args.batch_size == 32
    assert args.eval_batch_size == 64
    assert args.max_length == 128
    assert args.weight_decay == pytest.approx(0.01)
    assert args.max_train_samples is None
    assert args.max_eval_samples is None
    assert args.fp16 is False


def test_parse_args_rejects_unknown_dataset():
    with pytest.raises(SystemExit):
        parse_args(["--dataset", "not_a_real_dataset"])


def test_parse_args_rejects_unknown_method():
    with pytest.raises(SystemExit):
        parse_args(["--method", "not_a_real_method"])


# --- limit_split ------------------------------------------------------------


def _toy_dataset_dict(n_rows: int = 10) -> dict:
    # A plain dict stands in for a `DatasetDict` here: `limit_split` only
    # requires `raw[split_name]` to support `len()`, `.select()`, and item
    # assignment, all of which a dict of real `datasets.Dataset` objects
    # provides -- no download needed.
    return {"train": Dataset.from_dict({"text": [f"row {i}" for i in range(n_rows)]})}


def test_limit_split_leaves_split_untouched_when_max_samples_is_none():
    raw = _toy_dataset_dict(10)
    result, actual_count = limit_split(raw, "train", None, "train")

    assert actual_count == 10
    assert len(result["train"]) == 10


def test_limit_split_deterministically_keeps_the_first_n_rows():
    raw = _toy_dataset_dict(10)
    result, actual_count = limit_split(raw, "train", 3, "train")

    assert actual_count == 3
    assert len(result["train"]) == 3
    # First-N, no shuffling -- order must be preserved for reproducibility.
    assert result["train"]["text"] == ["row 0", "row 1", "row 2"]


def test_limit_split_clamps_when_max_samples_exceeds_available_rows():
    raw = _toy_dataset_dict(5)
    result, actual_count = limit_split(raw, "train", 1000, "train")

    assert actual_count == 5
    assert len(result["train"]) == 5
