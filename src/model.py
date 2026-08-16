"""Model construction: RoBERTa-base for full fine-tuning or LoRA (via PEFT).

Verified against the installed transformers==5.15.0 / peft==0.20.0 API:
RoBERTa's self-attention submodules are named "query", "key", "value"
(transformers.models.roberta.modeling_roberta.RobertaSelfAttention), so
target_modules=["query", "value"] adapts exactly the Q and V projections,
matching the paper's setup.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, PreTrainedModel

MODEL_NAME = "roberta-base"

LORA_TARGET_MODULES: list[str] = ["query", "value"]


@dataclass
class LoRASettings:
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: list[str] = field(default_factory=lambda: list(LORA_TARGET_MODULES))
    bias: str = "none"


def build_full_finetune_model(
    num_labels: int, model_name: str = MODEL_NAME
) -> PreTrainedModel:
    """Full fine-tuning baseline: every parameter is trainable."""
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    for param in model.parameters():
        param.requires_grad = True
    return model


def build_lora_model(
    num_labels: int,
    lora_settings: LoRASettings | None = None,
    model_name: str = MODEL_NAME,
) -> PeftModel:
    """Frozen RoBERTa-base backbone + trainable LoRA adapters on Q/V projections."""
    settings = lora_settings or LoRASettings()
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=settings.r,
        lora_alpha=settings.lora_alpha,
        lora_dropout=settings.lora_dropout,
        target_modules=settings.target_modules,
        bias=settings.bias,
    )
    model = get_peft_model(base_model, peft_config)
    return model


def build_model(
    method: str,
    num_labels: int,
    model_name: str = MODEL_NAME,
    lora_settings: LoRASettings | None = None,
) -> PreTrainedModel:
    """Dispatch to the full-fine-tuning or LoRA builder by name."""
    if method == "full":
        return build_full_finetune_model(num_labels, model_name)
    if method == "lora":
        return build_lora_model(num_labels, lora_settings, model_name)
    raise ValueError(f"Unknown method {method!r}; expected 'full' or 'lora'.")
