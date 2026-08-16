"""Standalone evaluation entry point for a previously saved checkpoint/adapter.

Example usage (not run by this implementation step):

    python -m src.evaluate --dataset sst2 --method lora \
        --checkpoint checkpoints/sst2/sst2_lora_r8_seed42

For a LoRA run, --checkpoint should point at the directory PEFT saved the
adapter to (contains adapter_config.json + adapter weights); for a full
fine-tuning run it should point at the saved Hugging Face model directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from src.data import DATASET_REGISTRY, load_raw_dataset, tokenize_dataset
from src.metrics import make_compute_metrics_fn
from src.model import MODEL_NAME


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint.")
    parser.add_argument("--dataset", choices=list(DATASET_REGISTRY.keys()), default="sst2")
    parser.add_argument("--method", choices=["full", "lora"], default="lora")
    parser.add_argument("--model-name", default=MODEL_NAME, help="Base model (for LoRA adapters)")
    parser.add_argument("--checkpoint", required=True, help="Path to saved model/adapter dir")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--split", default=None, help="Override the dataset's default eval split")
    parser.add_argument("--output-file", default=None, help="Where to write the metrics JSON")
    return parser.parse_args(argv)


def load_model_for_eval(method: str, checkpoint: str, num_labels: int, model_name: str):
    if method == "full":
        return AutoModelForSequenceClassification.from_pretrained(
            checkpoint, num_labels=num_labels
        )
    if method == "lora":
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        return PeftModel.from_pretrained(base_model, checkpoint)
    raise ValueError(f"Unknown method {method!r}; expected 'full' or 'lora'.")


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    spec = DATASET_REGISTRY[args.dataset]
    eval_split = args.split or spec.eval_split

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    raw = load_raw_dataset(args.dataset)
    tokenized = tokenize_dataset(raw, tokenizer, args.dataset, max_length=args.max_length)

    model = load_model_for_eval(args.method, args.checkpoint, spec.num_labels, args.model_name)
    compute_metrics = make_compute_metrics_fn(average=spec.metric_average)

    training_args = TrainingArguments(
        output_dir="results/tmp_eval",
        per_device_eval_batch_size=args.batch_size,
        report_to=[],
    )
    trainer = Trainer(model=model, args=training_args, compute_metrics=compute_metrics)

    metrics = trainer.evaluate(eval_dataset=tokenized[eval_split])

    result = {
        "dataset": args.dataset,
        "method": args.method,
        "checkpoint": args.checkpoint,
        "split": eval_split,
        "metrics": metrics,
    }

    if args.output_file:
        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
