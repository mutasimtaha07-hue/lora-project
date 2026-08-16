"""Training entry point for the LoRA / full-fine-tuning experiments.

Example usage (not run by this implementation step):

    python -m src.train --dataset sst2 --method lora --rank 8 --seed 42
    python -m src.train --dataset sst2 --method full --seed 42

Every run's hyperparameters and resulting metrics are written to a JSON
summary under --results-dir so the rank x seed sweep can be aggregated later
without re-parsing training logs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

from src.data import DATASET_REGISTRY, load_raw_dataset, tokenize_dataset
from src.metrics import make_compute_metrics_fn
from src.model import MODEL_NAME, LoRASettings, build_model
from src.utils import (
    count_total_parameters,
    count_trainable_parameters,
    peak_gpu_memory_bytes,
    reset_peak_gpu_memory_stats,
    set_seed,
)


class EpochTimingCallback(TrainerCallback):
    """Records the wall-clock duration of each training epoch.

    `Trainer` doesn't expose per-epoch timing on its own — `trainer.train()`
    only gives you a single total duration. PROPOSAL.md's efficiency metrics
    require "training time per epoch", so this callback timestamps every
    epoch boundary and accumulates the per-epoch durations.
    """

    def __init__(self):
        self.epoch_times_seconds: list[float] = []
        self._epoch_start: float | None = None

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._epoch_start = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        if self._epoch_start is not None:
            self.epoch_times_seconds.append(time.time() - self._epoch_start)
            self._epoch_start = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train RoBERTa-base with full fine-tuning or LoRA."
    )
    parser.add_argument("--dataset", choices=list(DATASET_REGISTRY.keys()), default="sst2")
    parser.add_argument("--method", choices=["full", "lora"], default="lora")
    parser.add_argument("--model-name", default=MODEL_NAME)

    # LoRA hyperparameters (ignored when --method full).
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank r")
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)

    # Reproducibility + optimization.
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate. Defaults to 2e-4 for LoRA, 2e-5 for full fine-tuning.",
    )
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--weight-decay", type=float, default=0.01)

    # Smoke-testing: deterministically cap split sizes. Both default to None
    # (= use the full split), so the real experiments are unaffected unless
    # these flags are explicitly passed.
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="If set, deterministically use only the first N training examples "
        "(smoke testing). Default: use the full training split.",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="If set, deterministically use only the first N evaluation examples "
        "(smoke testing). Default: use the full evaluation split.",
    )

    # Paths.
    parser.add_argument("--output-dir", default=None, help="Checkpoint dir override")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--results-dir", default="results/raw")
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Path to a saved Trainer checkpoint dir to resume training from "
        "(model, optimizer, scheduler, and RNG state). Default: None, i.e. "
        "start a fresh run — existing behavior is unchanged unless this is passed.",
    )

    parser.add_argument("--fp16", action="store_true", default=False)

    return parser.parse_args(argv)


def default_lr(method: str) -> float:
    return 2e-4 if method == "lora" else 2e-5


def build_run_name(args: argparse.Namespace) -> str:
    if args.method == "lora":
        return f"{args.dataset}_lora_r{args.rank}_seed{args.seed}"
    return f"{args.dataset}_full_seed{args.seed}"


def limit_split(raw, split_name: str, max_samples: int | None, label: str):
    """Deterministically keep only the first `max_samples` examples of
    `raw[split_name]` (no shuffling/random sampling — preserves reproducibility).

    Returns `(raw, actual_count)`. If `max_samples` is None the split is left
    untouched and `actual_count` is just its existing size. If `max_samples`
    exceeds the split's size, it's clamped to the available size (with a
    warning) rather than raising, and the clamped count is what gets recorded.
    """
    available = len(raw[split_name])
    if max_samples is None:
        return raw, available
    if max_samples > available:
        print(
            f"Warning: --max-{label}-samples={max_samples} exceeds the "
            f"available {available} examples in split '{split_name}'; "
            f"using all {available} instead."
        )
    n = min(max_samples, available)
    raw[split_name] = raw[split_name].select(range(n))
    return raw, n


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    set_seed(args.seed)
    reset_peak_gpu_memory_stats()

    spec = DATASET_REGISTRY[args.dataset]
    run_name = build_run_name(args)
    output_dir = args.output_dir or str(Path(args.checkpoints_dir) / args.dataset / run_name)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    raw = load_raw_dataset(args.dataset)

    raw, train_samples_used = limit_split(
        raw, spec.train_split, args.max_train_samples, "train"
    )
    raw, eval_samples_used = limit_split(
        raw, spec.eval_split, args.max_eval_samples, "eval"
    )

    tokenized = tokenize_dataset(raw, tokenizer, args.dataset, max_length=args.max_length)

    lora_settings = LoRASettings(
        r=args.rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout
    )
    model = build_model(args.method, spec.num_labels, args.model_name, lora_settings)

    trainable_params = count_trainable_parameters(model)
    total_params = count_total_parameters(model)
    lr = args.lr if args.lr is not None else default_lr(args.method)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to=[],
        fp16=args.fp16,
    )

    compute_metrics = make_compute_metrics_fn(average=spec.metric_average)
    epoch_timer = EpochTimingCallback()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized[spec.train_split],
        eval_dataset=tokenized[spec.eval_split],
        compute_metrics=compute_metrics,
        callbacks=[epoch_timer],
    )

    start_time = time.time()
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    train_time_seconds = time.time() - start_time

    epoch_times_seconds = epoch_timer.epoch_times_seconds
    avg_epoch_time_seconds = (
        sum(epoch_times_seconds) / len(epoch_times_seconds) if epoch_times_seconds else None
    )

    eval_metrics = trainer.evaluate()

    summary = {
        "run_name": run_name,
        "dataset": args.dataset,
        "method": args.method,
        "model_name": args.model_name,
        "rank": args.rank if args.method == "lora" else None,
        "lora_alpha": args.lora_alpha if args.method == "lora" else None,
        "lora_dropout": args.lora_dropout if args.method == "lora" else None,
        "target_modules": lora_settings.target_modules if args.method == "lora" else None,
        "seed": args.seed,
        "learning_rate": lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "weight_decay": args.weight_decay,
        "max_length": args.max_length,
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        "train_samples_used": train_samples_used,
        "eval_samples_used": eval_samples_used,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "train_time_seconds": train_time_seconds,
        "epoch_times_seconds": epoch_times_seconds,
        "avg_epoch_time_seconds": avg_epoch_time_seconds,
        "eval_metrics": eval_metrics,
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes(),
    }

    summary_path = results_dir / f"{run_name}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
