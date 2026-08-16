# Reproducing LoRA: Cross-Dataset Rank Sensitivity

> A controlled 30-run study of LoRA rank sensitivity across SST-2 and AG News using
> RoBERTa-base, with 5 ranks × 3 seeds per dataset and full-fine-tuning baselines.

`30 LoRA runs` · `2 datasets` · `5 ranks × 3 seeds` · `RoBERTa-base` · `PEFT / LoRA`

## Overview

This repository reproduces the core fine-tuning methodology of **LoRA: Low-Rank
Adaptation of Large Language Models** (Hu et al., ICLR 2022) on RoBERTa-base, and extends
it with a cross-dataset rank-sensitivity study. LoRA freezes a pretrained model's weights
and injects small trainable low-rank matrices into selected layers, dramatically reducing
the number of trainable parameters needed to fine-tune a model for a downstream task.

We fine-tune RoBERTa-base with LoRA adapters on the query and value attention projections
across five ranks (r ∈ {1, 2, 4, 8, 16}) and three random seeds per rank, on two
classification tasks of different complexity — **SST-2** (binary sentiment) and **AG
News** (four-class topic classification) — and compare against a full-fine-tuning
baseline for each dataset. The full experimental record (30 LoRA runs, 2 full-FT
baselines, one preserved failed run), analysis code, figures, and technical report are
included in this repository.

### Key Takeaway

The experiment does not support the hypothesis that the more complex four-class AG News
task requires substantially higher LoRA rank. Across both datasets, increasing rank from
1 to 16 produced only marginal gains while LoRA trained less than 1% of the parameters
used by full fine-tuning.

## Research Question

> How sensitive is downstream classification performance to LoRA rank, and does task
> complexity affect the rank required for strong performance?

## Hypothesis

The original project proposal stated the following hypothesis, treated throughout as
something to test empirically rather than assume:

> "r ≤ 4 should be sufficient for SST-2, while AG News may require a higher rank to match
> full fine-tuning performance."

See [`report/final_report.md`](report/final_report.md) (Section 10) for how this
hypothesis was evaluated against the collected results.

## Experimental Setup

| | |
|---|---|
| Backbone | RoBERTa-base |
| Datasets | SST-2 (binary sentiment), AG News (4-class topic) |
| LoRA ranks | 1, 2, 4, 8, 16 |
| Seeds | 42, 123, 999 |
| LoRA target modules | `query`, `value` attention projections |
| LoRA alpha / dropout | 16 / 0.1 |
| Learning rate (LoRA / full-FT) | 2e-4 / 2e-5 |
| Batch size / eval batch size | 32 / 64 |
| Epochs | 3 |
| Max sequence length | 128 |
| Full fine-tuning baselines | one valid run per dataset, seed 42 |

## Main Results

Mean validation accuracy (LoRA: mean across 3 seeds; Full FT: single seed, 42):

| Dataset | r=1 | r=2 | r=4 | r=8 | r=16 | Full FT |
|---|---|---|---|---|---|---|
| SST-2 | 0.9354 | 0.9346 | 0.9362 | 0.9365 | 0.9373 | 0.9438 |
| AG News | 0.9408 | 0.9421 | 0.9428 | 0.9426 | 0.9433 | 0.9542 |

**Conclusions:**
- **r=16 is the best mean rank on both datasets**, but the gap from r=1 is small: **+0.19
  percentage points** on SST-2 and **+0.25 percentage points** on AG News.
- **AG News does not show evidence of requiring substantially higher rank** than SST-2 —
  its rank-to-rank spread and per-seed variance are comparable to, or tighter than,
  SST-2's (see [`report/final_report.md`](report/final_report.md), Sections 7–10, for the
  full analysis and the reasoning behind this conclusion).
- **LoRA uses less than 1% of full fine-tuning's trainable parameters** at every rank
  tested, while reaching within roughly 0.7–1.4 accuracy points of full fine-tuning on
  both datasets.

These are descriptive findings from a three-seed experimental matrix — see
[Limitations](#limitations) below and the report's own limitations section for how much
weight to place on them.

## Parameter Efficiency

| Configuration | Trainable parameters | % of full fine-tuning |
|---|---|---|
| LoRA r=1 | 630,532 | 0.503% |
| LoRA r=16 | 1,183,492 | 0.941% |
| Full fine-tuning | 124,648,708 | 100% |

(Figures above use AG News's exact parameter counts; SST-2's are within 0.25% of these —
see the report for the full breakdown by dataset.)

## Figures

All figures are generated from the stored experiment results by
[`experiments/generate_report_figures.py`](experiments/generate_report_figures.py) and
are included in this repository (not just regenerable — see [Reproducibility](#reproducibility)):

- [`figures/accuracy_vs_rank.png`](figures/accuracy_vs_rank.png) — mean accuracy vs. LoRA rank, both datasets
- [`figures/f1_vs_rank.png`](figures/f1_vs_rank.png) — mean F1 vs. LoRA rank, both datasets
- [`figures/lora_vs_full_ft.png`](figures/lora_vs_full_ft.png) — LoRA (r=1..16) vs. full fine-tuning
- [`figures/trainable_params_vs_rank.png`](figures/trainable_params_vs_rank.png) — trainable parameters vs. rank (log scale)
- [`figures/parameter_efficiency.png`](figures/parameter_efficiency.png) — accuracy vs. % of full-FT parameters trained
- [`figures/rank_sensitivity.png`](figures/rank_sensitivity.png) — accuracy change relative to r=1

## Reproducibility

### 1. Install dependencies

```bash
python -m venv .venv
```

Activate it (`.venv\Scripts\activate` on Windows, `source .venv/bin/activate` on
Linux/macOS), then install PyTorch separately from PyTorch's own index — see the comment
block at the top of [`requirements.txt`](requirements.txt) for why a plain
`torch==2.11.0+cu128` pin can't live in `requirements.txt` itself — followed by everything
else:

```bash
pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### 2. Activate the environment

Already covered above — make sure `.venv` is activated before running any of the commands
below.

### 3. Run training commands

Every run is invoked through [`src/train.py`](src/train.py)'s CLI. These are the exact
command shapes used to produce every result in `results/raw/` (confirmed against
`src/train.py`'s actual `argparse` definitions, not hypothetical):

**SST-2, LoRA** (rank and seed swept across the 5×3 matrix):

```bash
python -m src.train --dataset sst2 --method lora --rank 8 --seed 42 \
    --model-name roberta-base --epochs 3 --batch-size 32 --eval-batch-size 64 \
    --lr 2e-4 --weight-decay 0.01 --max-length 128 \
    --checkpoints-dir checkpoints --results-dir results/raw
```

**SST-2, full fine-tuning baseline:**

```bash
python -m src.train --dataset sst2 --method full --seed 42 \
    --model-name roberta-base --epochs 3 --batch-size 32 --eval-batch-size 64 \
    --lr 2e-5 --weight-decay 0.01 --max-length 128 \
    --checkpoints-dir checkpoints --results-dir results/raw
```

**AG News, LoRA** (same rank/seed sweep):

```bash
python -m src.train --dataset ag_news --method lora --rank 8 --seed 42 \
    --model-name roberta-base --epochs 3 --batch-size 32 --eval-batch-size 64 \
    --lr 2e-4 --weight-decay 0.01 --max-length 128 \
    --checkpoints-dir checkpoints --results-dir results/raw
```

**AG News, full fine-tuning baseline** (must use `--lr 2e-5`, *not* `2e-4` — see
[Experimental Lessons](#experimental-lessons)):

```bash
python -m src.train --dataset ag_news --method full --seed 42 \
    --model-name roberta-base --epochs 3 --batch-size 32 --eval-batch-size 64 \
    --lr 2e-5 --weight-decay 0.01 --max-length 128 \
    --checkpoints-dir checkpoints --results-dir results/raw
```

To reproduce the full sweep, repeat the LoRA commands for `--rank` in `{1, 2, 4, 8, 16}`
and `--seed` in `{42, 123, 999}` (15 runs per dataset). `src/train.py` also supports
`--max-train-samples`/`--max-eval-samples` (for quick smoke tests on a data subset) and
`--resume-from-checkpoint <path>` (to resume an interrupted run from a saved checkpoint) —
run `python -m src.train --help` for the complete, current flag list.

### 4. Inspect result JSONs

Each run writes one JSON summary to `results/raw/{run_name}.json`, containing the full
effective hyperparameters, trainable/total parameter counts, per-epoch and total training
time, peak GPU memory, and final evaluation metrics. **`results/raw/` is tracked in this
repository** — it contains the 33 recorded result JSONs from the actual experiment runs
(see [Project Structure](#project-structure)) and is the primary record of what was
measured. Re-running the training commands above is optional for reproduction; the
repository already contains the recorded results.

### 5. Run the analysis / figure-generation script

The figure/table generation script reads the stored, tracked JSONs already in
`results/raw/` — no training run is required — and regenerates the figures and summary
tables with:

```bash
python -m experiments.generate_report_figures
```

This reads only `results/raw/*.json` and writes to `figures/` and `results/tables/` — it
does not retrain anything.

## Project Structure

```
lora-project/
├── PROPOSAL.md                 # original project proposal
├── README.md                   # this file
├── requirements.txt            # pinned Python dependencies
├── configs/                    # documentation-only hyperparameter reference
│   ├── lora.yaml
│   ├── full_finetuning.yaml
│   └── experiments.yaml
├── data/
│   └── README.md               # datasets are fetched on demand via HF `datasets`
├── src/
│   ├── data.py                  # dataset registry (SST-2, AG News) + tokenization
│   ├── model.py                  # RoBERTa-base builders: full-FT and LoRA (Q/V) via PEFT
│   ├── metrics.py                # accuracy / F1 / precision / recall
│   ├── train.py                   # CLI training entry point
│   ├── evaluate.py               # CLI evaluation entry point for a saved checkpoint
│   └── utils.py                   # seeding, parameter counting, GPU memory helpers
├── experiments/
│   └── generate_report_figures.py   # regenerates figures/ + results/tables/ from results/raw/
├── results/
│   ├── raw/                    # 33 tracked per-run result JSONs — primary recorded outputs
│   └── tables/                 # curated summary CSV/Markdown tables (tracked)
├── figures/                    # generated PNG figures (tracked)
├── checkpoints/                # per-run model/adapter checkpoints (large, not tracked)
└── report/
    └── final_report.md         # full technical report
```

## Experimental Lessons

- **A full-fine-tuning learning-rate misconfiguration caused a training collapse.** An
  initial AG News full-fine-tuning attempt used `lr=2e-4` (the LoRA learning rate) instead
  of a rate appropriate for updating all 124.6M parameters, and the model collapsed to
  predicting a single constant class. This failed run is preserved on disk (not deleted or
  overwritten) as documented evidence and is **not** used as a valid baseline anywhere in
  this repository. The corrected baseline uses **`lr=2e-5`**, matching SST-2's recipe, and
  trained normally.
- **The root cause was a documentation/execution gap:** `configs/*.yaml` document intended
  hyperparameters but are **not loaded by any code** — every actual training run gets its
  parameters from CLI arguments passed to `src/train.py`. The `2e-5` learning rate had
  always been documented correctly in `configs/full_finetuning.yaml`; the failed run simply
  didn't use it. See [`report/final_report.md`](report/final_report.md) (Section 12) for
  the full discussion.
- **Several runs experienced genuine host-machine sleep/suspend events** during training —
  the largest observed gap was approximately **8 hours 17 minutes** in one AG News run.
  Every such anomaly was independently verified against checkpoint file timestamps and
  confirmed to be a real wall-clock gap, not a bug; all affected runs completed with valid
  metrics and correct parameter counts. See the report's reproducibility notes for the
  full list of affected runs.

## Limitations

- Only **three random seeds** per rank/dataset were used for the LoRA sweep.
- Only **one full-fine-tuning seed** per dataset — the full-FT reference values have no
  associated seed variance.
- Only **two classification datasets** (SST-2, AG News) were evaluated.
- Only **RoBERTa-base** was tested as the backbone model.
- Several runs' recorded training-time fields were affected by host-machine
  sleep/suspend events (see above) and should not be used for timing comparisons.

See [`report/final_report.md`](report/final_report.md) (Section 13) for the complete
limitations discussion.

## Citation

If you use or build on this reproduction, please cite the original LoRA paper:

> Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W.
> (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. International Conference
> on Learning Representations (ICLR 2022). https://arxiv.org/abs/2106.09685
