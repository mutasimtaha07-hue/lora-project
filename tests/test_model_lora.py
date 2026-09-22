"""Tests for src/model.py's LoRA / full-fine-tuning model construction.

These tests never download or load the real pretrained roberta-base weights.
`AutoModelForSequenceClassification.from_pretrained` is monkeypatched (only
inside this test module) to return a randomly-initialized model built from a
tiny synthetic `RobertaConfig` instead -- small enough to construct in well
under a second on CPU, while still using the real `RobertaForSequenceClassification`
/ `RobertaSelfAttention` classes, so its `query`/`key`/`value` submodule names
are identical to the real roberta-base architecture regardless of size.

No test here trains, calls `.backward()`, or creates an optimizer -- each one
only constructs a model and inspects `.named_modules()` / `.named_parameters()`.

The exact PEFT module/parameter naming asserted below (`lora_A`/`lora_B`
attributes, `classifier.modules_to_save.default.*` paths) was verified by
directly inspecting a real tiny LoRA-wrapped model in this environment's
installed `peft==0.20.0` before writing these assertions -- see the audit
notes; it is not guessed from documentation.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from transformers import RobertaConfig, RobertaForSequenceClassification

from src.model import LoRASettings, build_full_finetune_model, build_lora_model
from src.utils import count_total_parameters, count_trainable_parameters

# A tiny but architecturally-real RoBERTa config: 2 layers is enough for
# `target_modules=["query", "value"]` to be exercised more than once, while
# staying fast to construct on CPU.
_TINY_CONFIG_KWARGS = dict(
    vocab_size=99,
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=2,
    intermediate_size=64,
    max_position_embeddings=64,
    type_vocab_size=1,
)


def _fake_from_pretrained(model_name, num_labels=2, **kwargs):
    """Stand-in for `AutoModelForSequenceClassification.from_pretrained`.

    Ignores `model_name` entirely and returns a freshly (randomly)
    initialized tiny model -- no network access, no HF cache lookup.
    """
    config = RobertaConfig(num_labels=num_labels, **_TINY_CONFIG_KWARGS)
    return RobertaForSequenceClassification(config)


@pytest.fixture
def patched_from_pretrained():
    with patch(
        "src.model.AutoModelForSequenceClassification.from_pretrained",
        side_effect=_fake_from_pretrained,
    ) as mock_fp:
        yield mock_fp


def _lora_adapted_module_names(model) -> set[str]:
    """Return the last path segment of every module PEFT wrapped with LoRA.

    A LoRA-adapted `nn.Linear` carries `lora_A`/`lora_B` `ModuleDict`
    attributes (confirmed empirically for this project's peft version); its
    own name's final path segment (e.g. "...attention.self.query") is the
    projection it replaced.
    """
    names = set()
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            names.add(name.rsplit(".", 1)[-1])
    return names


def test_lora_targets_only_query_and_value(patched_from_pretrained):
    model = build_lora_model(num_labels=2, lora_settings=LoRASettings(r=4))

    adapted = _lora_adapted_module_names(model)

    assert adapted == {"query", "value"}


def test_trainable_params_increase_monotonically_with_rank(patched_from_pretrained):
    trainable_by_rank = {}
    for r in (1, 2, 4, 8):
        model = build_lora_model(num_labels=2, lora_settings=LoRASettings(r=r))
        trainable_by_rank[r] = count_trainable_parameters(model)

    assert trainable_by_rank[1] < trainable_by_rank[2] < trainable_by_rank[4] < trainable_by_rank[8]


def test_only_lora_and_classifier_params_are_trainable(patched_from_pretrained):
    model = build_lora_model(num_labels=2, lora_settings=LoRASettings(r=4))

    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable_names, "expected at least one trainable parameter"

    for name in trainable_names:
        assert "lora_" in name or "classifier" in name, (
            f"Unexpected trainable parameter outside LoRA adapters/classifier head: {name}"
        )

    # The frozen backbone must include, at minimum, the "key" projection
    # (never targeted by target_modules=["query", "value"]) and the word
    # embeddings -- both must remain non-trainable.
    frozen_names = [name for name, p in model.named_parameters() if not p.requires_grad]
    assert any(".key." in name for name in frozen_names)
    assert any("embeddings" in name for name in frozen_names)
    assert not any(".key." in name for name in trainable_names)


def test_full_finetune_makes_every_parameter_trainable(patched_from_pretrained):
    model = build_full_finetune_model(num_labels=2)

    total = count_total_parameters(model)
    trainable = count_trainable_parameters(model)

    assert total > 0
    assert trainable == total
