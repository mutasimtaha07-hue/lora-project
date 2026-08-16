"""Dataset loading and tokenization.

Fully implemented and validated for SST-2 (the paper-reproduction dataset).
AG News is registered with the same interface so the extension experiment
(PROPOSAL.md Sec. "Extension dataset") is a config addition rather than a
rewrite, but it has not been exercised/downloaded yet — that happens in a
later implementation step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from datasets import DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True)
class DatasetSpec:
    hf_path: str
    hf_name: Optional[str]
    text_column: str
    label_column: str
    num_labels: int
    train_split: str
    eval_split: str
    # "binary" for 2-class tasks, "macro" for multi-class — passed straight
    # through to sklearn's precision/recall/F1 `average` argument.
    metric_average: str


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "sst2": DatasetSpec(
        hf_path="nyu-mll/glue",
        hf_name="sst2",
        text_column="sentence",
        label_column="label",
        num_labels=2,
        train_split="train",
        # GLUE's SST-2 "test" split has no labels (it's for leaderboard
        # submission); "validation" is the labeled held-out split to eval on.
        eval_split="validation",
        metric_average="binary",
    ),
    "ag_news": DatasetSpec(
        hf_path="fancyzhx/ag_news",
        hf_name=None,
        text_column="text",
        label_column="label",
        num_labels=4,
        train_split="train",
        eval_split="test",
        metric_average="macro",
    ),
}


def load_raw_dataset(dataset_key: str) -> DatasetDict:
    """Download (or load from HF cache) the raw, untokenized dataset."""
    spec = DATASET_REGISTRY[dataset_key]
    if spec.hf_name is not None:
        return load_dataset(spec.hf_path, spec.hf_name)
    return load_dataset(spec.hf_path)


def tokenize_dataset(
    raw: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    dataset_key: str,
    max_length: int = 128,
) -> DatasetDict:
    """Tokenize text columns, rename the label column to "labels", and drop
    everything the model doesn't need so the dataset is ready for a
    `transformers.Trainer` (or a plain PyTorch DataLoader) to consume.
    """
    spec = DATASET_REGISTRY[dataset_key]

    def _tokenize(batch):
        return tokenizer(
            batch[spec.text_column],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    tokenized = raw.map(_tokenize, batched=True)

    if spec.label_column != "labels":
        tokenized = tokenized.rename_column(spec.label_column, "labels")

    reference_split = spec.train_split
    keep_columns = {"input_ids", "attention_mask", "labels", "token_type_ids"}
    all_columns = tokenized[reference_split].column_names
    drop_columns = [c for c in all_columns if c not in keep_columns]
    tokenized = tokenized.remove_columns(drop_columns)

    tokenized.set_format(type="torch")
    return tokenized


def load_sst2(tokenizer: PreTrainedTokenizerBase, max_length: int = 128) -> DatasetDict:
    """Convenience wrapper: load + tokenize SST-2 in one call."""
    raw = load_raw_dataset("sst2")
    return tokenize_dataset(raw, tokenizer, "sst2", max_length=max_length)


def get_spec(dataset_key: str) -> DatasetSpec:
    return DATASET_REGISTRY[dataset_key]
