# Reproducing LoRA: Cross-Dataset Rank Sensitivity of Large Language Models

---

## Abstract

This report reproduces the core methodology of Low-Rank Adaptation (LoRA) fine-tuning
(Hu et al., 2022) on RoBERTa-base and extends it with a cross-dataset rank-sensitivity
study. We fine-tune RoBERTa-base with LoRA adapters (applied to the query and value
attention projections) across five ranks — r ∈ {1, 2, 4, 8, 16} — and three random seeds
per rank, on two text classification datasets of different complexity: SST-2 (binary
sentiment classification) and AG News (four-class topic classification). This yields a
30-run experimental matrix (15 SST-2 + 15 AG News LoRA runs), compared against a
full-fine-tuning baseline for each dataset. The main result is that LoRA is highly
parameter-efficient on both datasets — r=1 trains only 0.50% of full fine-tuning's
parameters while reaching within roughly 0.7–1.4 accuracy points of it — and that
increasing rank from 1 to 16 produces only marginal accuracy gains (+0.19 percentage
points on SST-2, +0.25 on AG News). The proposal's original hypothesis, that the more
complex four-class AG News task would require substantially higher LoRA rank than the
binary SST-2 task, is **not supported** by these results: AG News's rank-to-rank spread
and seed variance are comparable to, or tighter than, SST-2's. We report this finding
with the caveat that it is based on three seeds per configuration and single-seed
full-fine-tuning baselines.

---

## 1. Introduction

Fine-tuning large pretrained language models on downstream tasks traditionally updates
every parameter of the model. For models the size of RoBERTa-base (~125M parameters)
this is computationally manageable, but the same full-fine-tuning approach becomes
increasingly expensive as model scale grows, and it requires storing a full,
separately-fine-tuned copy of the model for every downstream task. This has motivated a
family of **parameter-efficient fine-tuning (PEFT)** methods, which freeze most or all of
the pretrained weights and train only a small number of additional parameters per task.

**Low-Rank Adaptation (LoRA)** (Hu et al., 2022) is one such method. It freezes the
pretrained weight matrices and injects trainable low-rank update matrices into selected
layers — commonly the attention projections. The rank of these update matrices, r, is a
central hyperparameter: a smaller r means fewer trainable parameters and less memory
overhead, but potentially less capacity to adapt the model. Understanding how sensitive
downstream performance actually is to this rank choice — and whether that sensitivity
depends on the difficulty of the downstream task — has direct practical implications for
how aggressively rank can be reduced in real deployments.

This project reproduces LoRA's core fine-tuning methodology on SST-2, the setting closest
to the original paper's GLUE experiments, and extends the evaluation to AG News, a
four-class topic-classification task, to ask whether task complexity changes the rank
required for LoRA to perform well. We do not claim to reproduce the original paper's exact
numbers point-for-point — a like RoBERTa-base backbone, dataset, and training recipe are
used, but exact hyperparameter choices and the resulting deltas from full fine-tuning
should be expected to differ somewhat from the paper's published GLUE results.

**Research question:** How sensitive is downstream classification performance to LoRA
rank, and does task complexity affect the rank required for strong performance?

---

## 2. Research Hypothesis

The project proposal stated the following empirical hypothesis, to be tested rather than
assumed:

> "A low rank (r ≤ 4) should be sufficient for binary classification such as SST-2, while
> the more complex four-class AG News task may require a higher rank to match full
> fine-tuning performance."

This was framed from the outset as a hypothesis to be tested experimentally, not as an
expected or assumed conclusion. Sections 7–10 of this report evaluate it directly against
the collected data.

---

## 3. Background: LoRA

LoRA modifies a pretrained weight matrix `W ∈ R^{d×k}` by adding a learned low-rank update
rather than updating `W` directly:

```
W' = W + ΔW
ΔW = B·A
```

where `B ∈ R^{d×r}`, `A ∈ R^{r×k}`, and the rank `r ≪ min(d, k)`. During fine-tuning, the
original weights `W` are **frozen** — they receive no gradient updates — and only `A` and
`B` are trained. At inference time, `ΔW = B·A` can be added back into `W` directly, so LoRA
introduces no extra inference latency.

Because `A` and `B` are low-rank, the number of trainable parameters they introduce is
`r·(d + k)`, which is small relative to the `d·k` parameters of the full matrix `W` when
`r` is small. In this project's setup, LoRA adapters are applied only to the **query** and
**value** projection matrices of each self-attention layer, and each adapter pair is
scaled by `lora_alpha / r` (following common LoRA practice) with `lora_alpha = 16` and a
dropout of `0.1` applied to the adapter branch. The pretrained RoBERTa-base backbone
remains frozen throughout; the only trainable parameters are the LoRA `A`/`B` matrices
for the targeted Q/V projections, plus the (small) classification head required for each
downstream task.

This report reproduces the mechanism above and its practical parameter-efficiency
consequences; it does not attempt to restate the original paper's derivations or
theoretical analysis. For the full method and its motivation, see the original paper:

> Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W.
> (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022.
> https://arxiv.org/abs/2106.09685

---

## 4. Experimental Setup

### Model

**RoBERTa-base** (~125M parameters), used as the frozen backbone for LoRA runs and as the
fully-trainable model for the full fine-tuning baselines.

### Datasets

**SST-2** — Stanford Sentiment Treebank, binary sentiment classification, distributed via
the GLUE benchmark. Evaluation is performed on the GLUE validation split (the labeled
held-out set; GLUE's SST-2 test split is unlabeled and unsuitable for evaluation).

**AG News** — four-class news topic classification (World, Sports, Business,
Sci/Tech). Evaluation is performed on the dataset's standard test split, used here as the
validation set for model selection and reporting.

### LoRA configuration

| Setting | Value |
|---|---|
| Ranks | 1, 2, 4, 8, 16 |
| Seeds | 42, 123, 999 |
| Target modules | query, value projections |
| LoRA alpha | 16 |
| LoRA dropout | 0.1 |
| Learning rate | 2e-4 |
| Batch size / eval batch size | 32 / 64 |
| Epochs | 3 |
| Max sequence length | 128 |
| Weight decay | 0.01 |

### Baselines: full fine-tuning

| Dataset | Seed | Learning rate | Accuracy | F1 |
|---|---|---|---|---|
| SST-2 | 42 | 2e-5 | 0.9438 | 0.9454 |
| AG News | 42 | 2e-5 | 0.9542 | 0.9542 |

Both full-fine-tuning baselines use a single seed (42) — see Limitations (Section 13) for
the implications of this.

An initial AG News full-fine-tuning attempt at learning rate `2e-4` (matching the LoRA
learning rate rather than a full-fine-tuning-appropriate rate) **collapsed** during
training and is **not** used as a valid baseline anywhere in this report. It is preserved
on disk as documented experimental failure evidence and discussed in Section 12.

---

## 5. Experimental Matrix

| Dataset | Method | Ranks | Seeds | Number of runs |
|---|---|---|---|---|
| SST-2 | LoRA | 1, 2, 4, 8, 16 | 42, 123, 999 | 15 |
| SST-2 | Full fine-tuning | — | 42 | 1 |
| AG News | LoRA | 1, 2, 4, 8, 16 | 42, 123, 999 | 15 |
| AG News | Full fine-tuning (valid) | — | 42 | 1 |
| AG News | Full fine-tuning (failed, preserved as evidence, not a baseline) | — | 42 | 1 |

**30 LoRA runs total** (15 per dataset, 5 ranks × 3 seeds each), plus one valid
full-fine-tuning baseline per dataset, plus the one preserved failed AG News run.

---

## 6. Results

All values below are taken directly from `results/tables/summary_tables.md` (and the
underlying `table1`–`table4` CSVs), which are computed from the raw per-run result JSONs
in `results/raw/`.

### Table 1 — AG News LoRA results (mean ± std across 3 seeds)

| Rank | Accuracy | F1 | Trainable params | Trainable % |
|---|---|---|---|---|
| 1 | 0.9408 ± 0.0009 | 0.9407 ± 0.0009 | 630,532 | 0.503% |
| 2 | 0.9421 ± 0.0005 | 0.9421 ± 0.0005 | 667,396 | 0.533% |
| 4 | 0.9428 ± 0.0006 | 0.9427 ± 0.0006 | 741,124 | 0.591% |
| 8 | 0.9426 ± 0.0003 | 0.9426 ± 0.0003 | 888,580 | 0.708% |
| 16 | 0.9433 ± 0.0003 | 0.9432 ± 0.0003 | 1,183,492 | 0.941% |

### Table 2 — SST-2 LoRA results (mean ± std across 3 seeds)

| Rank | Accuracy | F1 | Trainable params | Trainable % |
|---|---|---|---|---|
| 1 | 0.9354 ± 0.0029 | 0.9370 ± 0.0031 | 628,994 | 0.502% |
| 2 | 0.9346 ± 0.0041 | 0.9359 ± 0.0041 | 665,858 | 0.531% |
| 4 | 0.9362 ± 0.0026 | 0.9377 ± 0.0024 | 739,586 | 0.590% |
| 8 | 0.9365 ± 0.0048 | 0.9379 ± 0.0053 | 887,042 | 0.707% |
| 16 | 0.9373 ± 0.0029 | 0.9387 ± 0.0031 | 1,181,954 | 0.939% |

### Table 3 — Cross-dataset comparison

| Rank | SST-2 accuracy | AG News accuracy | SST-2 std | AG News std |
|---|---|---|---|---|
| 1 | 0.9354 | 0.9408 | 0.0029 | 0.0009 |
| 2 | 0.9346 | 0.9421 | 0.0041 | 0.0005 |
| 4 | 0.9362 | 0.9428 | 0.0026 | 0.0006 |
| 8 | 0.9365 | 0.9426 | 0.0048 | 0.0003 |
| 16 | 0.9373 | 0.9433 | 0.0029 | 0.0003 |

### Table 4 — Full fine-tuning comparison

| Dataset | LoRA rank | Accuracy | Gap to full FT | Trainable params | Trainable % |
|---|---|---|---|---|---|
| SST-2 | 1 | 0.9354 | 0.0084 | 628,994 | 0.502% |
| SST-2 | 2 | 0.9346 | 0.0092 | 665,858 | 0.531% |
| SST-2 | 4 | 0.9362 | 0.0076 | 739,586 | 0.590% |
| SST-2 | 8 | 0.9365 | 0.0073 | 887,042 | 0.707% |
| SST-2 | 16 | 0.9373 | 0.0065 | 1,181,954 | 0.939% |
| AG News | 1 | 0.9408 | 0.0134 | 630,532 | 0.503% |
| AG News | 2 | 0.9421 | 0.0121 | 667,396 | 0.533% |
| AG News | 4 | 0.9428 | 0.0114 | 741,124 | 0.591% |
| AG News | 8 | 0.9426 | 0.0116 | 888,580 | 0.708% |
| AG News | 16 | 0.9433 | 0.0109 | 1,183,492 | 0.941% |

### Figures

The following figures (generated by `experiments/generate_report_figures.py` from the raw
result JSONs — not regenerated for this report) are referenced here and should be viewed
alongside the tables above:

- `figures/accuracy_vs_rank.png` — mean accuracy vs. LoRA rank, both datasets, with std
  error bars.
- `figures/f1_vs_rank.png` — mean F1 vs. LoRA rank, both datasets, with std error bars.
- `figures/lora_vs_full_ft.png` — grouped bar chart comparing all five LoRA ranks against
  the (visually distinct, hatched) full-fine-tuning reference bars, explicitly labeled as
  single-seed with no error bar.
- `figures/trainable_params_vs_rank.png` — trainable parameter count vs. rank, log-scaled
  Y-axis, with the full-fine-tuning parameter count shown as a horizontal reference line
  roughly two orders of magnitude above the LoRA curve.
- `figures/parameter_efficiency.png` — mean accuracy vs. trainable parameters (as % of
  full fine-tuning, log-scaled X-axis), with full-fine-tuning reference points at 100%.
- `figures/rank_sensitivity.png` — change in mean accuracy relative to r=1, for both
  datasets, on a deliberately narrow Y-axis range to make the small magnitude of the rank
  effect visually obvious.

---

## 7. Rank Sensitivity

This is the central empirical question of the study: how much does LoRA rank actually
matter for downstream accuracy?

**SST-2:** mean accuracy rises from **0.9354** at r=1 to **0.9373** at r=16, a change of
**Δ = +0.0019** (0.19 percentage points). The relationship across ranks is **not
monotonic** — r=2 (0.9346) is actually slightly *lower* than r=1, before accuracy recovers
and edges upward at r=4, r=8, and r=16 (Figure `accuracy_vs_rank.png`, `rank_sensitivity.png`).

**AG News:** mean accuracy rises from **0.9408** at r=1 to **0.9433** at r=16, a change of
**Δ = +0.0025** (0.25 percentage points). Unlike SST-2, this trend is monotonically
non-decreasing from r=1 through r=4, with a small dip at r=8 (0.9426) before recovering to
its maximum at r=16.

Both deltas are small in absolute terms, and in both cases the magnitude of the rank
effect is comparable to — or smaller than — the seed-to-seed standard deviation observed
at individual ranks (e.g., SST-2's std at r=8 is 0.0048, nearly as large as the entire
r=1-to-r=16 change). **The observed rank effect is small relative to seed-to-seed
variability.** This does not mean rank has *no* effect — the direction of the trend is
consistently non-negative on both datasets — but the practical magnitude of that effect,
at least across the ranks tested here, is modest.

---

## 8. Parameter Efficiency

LoRA's central practical appeal is that it can approach full-fine-tuning performance while
training only a small fraction of the model's parameters. The counts below apply to both
datasets essentially identically — the LoRA adapter parameter count itself is
dataset-independent, and the only cross-dataset difference is a small classifier-head size
difference (SST-2 has 2 output classes, AG News has 4), which changes total trainable
parameters by well under 1%. AG News figures are used below as the reference values.

| Configuration | Trainable parameters | % of full fine-tuning |
|---|---|---|
| LoRA r=1 | 630,532 | 0.503% |
| LoRA r=16 | 1,183,492 | 0.941% |
| Full fine-tuning | 124,648,708 | 100% |

Even at the smallest rank tested (r=1), LoRA trains under **0.51%** of the parameters full
fine-tuning would update, while landing within roughly 0.7–1.4 accuracy points of the
full-fine-tuning baseline across both datasets (Section 6, Table 4). Increasing rank
16-fold, from r=1 to r=16, still keeps trainable parameters under 1% of the full model
(0.941%). Figure `parameter_efficiency.png` visualizes this directly: both datasets' LoRA
curves sit in the far-left, low-parameter region of the plot, clustered well below and to
the left of their respective full-fine-tuning reference points, with a comparatively flat
accuracy response across nearly two orders of magnitude of parameter-count variation on
the LoRA side alone.

**Practical implication:** for both a simple binary task and a moderately more complex
four-class task, a very small fraction of the pretrained model's parameters is sufficient
to recover the large majority of full fine-tuning's downstream performance.

---

## 9. Cross-Dataset Analysis

Comparing SST-2 and AG News directly surfaces two related but distinct findings that
should not be conflated.

**AG News reaches higher absolute accuracy than SST-2** at every rank tested (e.g., r=1:
0.9408 vs. 0.9354; r=16: 0.9433 vs. 0.9373) — this reflects the two tasks' different
intrinsic difficulty and dataset characteristics, not a claim about rank sensitivity.

**AG News has a larger LoRA-to-full-FT gap than SST-2, at every rank** (Table 4): AG
News's gap ranges from 0.0109 to 0.0134 (1.1–1.4 accuracy points), while SST-2's ranges
from 0.0065 to 0.0092 (0.65–0.92 points). AG News's full-fine-tuning ceiling (0.9542) is
higher than SST-2's (0.9438), and LoRA closes proportionally less of that larger ceiling.

**However, AG News does not show stronger rank sensitivity than SST-2.** Its
rank-to-rank accuracy spread (max − min across ranks: 0.9433 − 0.9408 = 0.0025) is
comparable to SST-2's (0.9373 − 0.9346 = 0.0027), and its **per-rank seed variance is
actually tighter** than SST-2's: AG News's std values range from 0.0003 to 0.0009, while
SST-2's range from 0.0024 to 0.0053 (Table 3). If the four-class task genuinely required
more rank capacity to adapt well, this would be expected to show up as a *steeper*
rank-to-rank accuracy climb on AG News than on SST-2 — instead, both curves are similarly
flat, and AG News's is, if anything, more stable across seeds.

**These two observations — "distance to the full-fine-tuning ceiling" and "rank
sensitivity" — are conceptually separate and should not be conflated.** AG News being
further from its own full-FT ceiling at a given rank is a statement about how much of that
particular dataset's achievable performance LoRA recovers; it says nothing by itself about
whether increasing rank would close that gap. The evidence here suggests it would not, at
least not substantially within the tested rank range: the gap-to-full-FT for AG News
barely narrows across the five ranks (0.0134 → 0.0109), a change on the same modest scale
as SST-2's own gap narrowing (0.0084 → 0.0065).

---

## 10. Hypothesis Evaluation

The proposal's hypothesis (Section 2) was that AG News, as the more complex four-class
task, would require a higher LoRA rank than SST-2 to match full fine-tuning performance.

**Conclusion: the observed results do not support the hypothesis that the more complex
4-class AG News task requires substantially higher LoRA rank.**

The evidence for this conclusion:

- **r=16 is the best mean rank on both datasets** — but this alone does not indicate a
  strong rank requirement; it merely identifies the top of a fairly flat curve.
- **r=1 already captures most of the achievable performance** on both datasets. It is
  within 0.19 points of SST-2's own r=16 result and within 0.25 points of AG News's r=16
  result — in both cases a small fraction of the corresponding gap to full fine-tuning.
- **The total r=1 → r=16 improvement is small on both datasets**: +0.19 percentage points
  on SST-2, +0.25 percentage points on AG News. If AG News genuinely needed higher rank,
  this delta would be expected to be markedly larger than SST-2's — instead, it is only
  modestly larger (0.25 vs. 0.19 points), and both deltas are of the same small order of
  magnitude.
- **AG News does not show stronger rank sensitivity** by the more direct measures examined
  in Section 9 — its rank-to-rank spread and per-seed variance are comparable to, or
  tighter than, SST-2's.

This is stated as an empirical, cautious finding, **not** as a mathematical disproof of the
hypothesis. The experiments were run with **three seeds per rank per dataset (n=3)**,
which is sufficient to observe that the rank effect is small relative to seed-to-seed
noise on this data, but is not a large enough sample to rule out a real, small effect that
these experiments were underpowered to detect, nor to extrapolate confidently to ranks
outside the 1–16 range tested, other model scales, or other task types.

---

## 11. Reproducibility

### Repository structure

```
lora-project/
├── PROPOSAL.md
├── requirements.txt
├── configs/                  # lora.yaml, full_finetuning.yaml, experiments.yaml
│                              #   (reference/documentation — see Section 12)
├── data/README.md            # datasets fetched on demand via Hugging Face `datasets`
├── src/
│   ├── data.py                # dataset registry (SST-2, AG News) + tokenization
│   ├── model.py                # RoBERTa-base builders: full-FT and LoRA (Q/V) via PEFT
│   ├── metrics.py              # accuracy / F1 / precision / recall
│   ├── train.py                 # CLI training entry point (Trainer-based)
│   ├── evaluate.py             # CLI evaluation entry point for a saved checkpoint/adapter
│   └── utils.py                 # seeding, parameter counting, GPU memory helpers
├── experiments/
│   └── generate_report_figures.py   # generates figures/ and results/tables/ from results/raw/
├── results/
│   ├── raw/                   # one JSON per run (hyperparameters + metrics + timing)
│   ├── processed/, tables/    # aggregated CSV/Markdown summary tables
├── figures/                   # generated PNG figures
├── checkpoints/{sst2,ag_news}/  # per-run model/adapter checkpoints
└── report/final_report.md     # this report
```

### Python environment

Python 3.11.9, in a project-local `.venv`. Exact package versions are pinned in
[`requirements.txt`](../requirements.txt), including `torch==2.11.0+cu128` (installed
separately from PyTorch's CUDA 12.8 index — see the comment block at the top of
`requirements.txt` for why), `transformers==5.15.0`, `peft==0.20.0`, `datasets==5.0.1`,
`accelerate==1.14.0`, `scikit-learn==1.9.0`, `evaluate==0.4.6`, and (added for this
documentation phase) `matplotlib==3.11.1`.

### Experiment configuration and seeds

Every experiment is invoked via `src/train.py`'s CLI (see Section 4 for the exact
hyperparameters used). Seeds `{42, 123, 999}` are set once per run via `set_seed()`
(covering Python's `random`, NumPy, and PyTorch CPU/CUDA RNGs) before any model or data is
constructed, ensuring each of the 3 seeds per rank/dataset is a fully independent run.

### Checkpoints and result JSONs

Each run writes:
- A per-epoch model/adapter checkpoint directory under `checkpoints/{dataset}/{run_name}/checkpoint-{step}/`
  (LoRA runs save adapter-only weights via `adapter_model.safetensors`; full-fine-tuning
  runs save the complete model).
- A single summary JSON under `results/raw/{run_name}.json`, containing the full effective
  hyperparameter configuration (including `model_name`, `eval_batch_size`, `weight_decay`,
  and `target_modules` — fields added mid-project once their absence was noticed, see
  Section 12), trainable/total parameter counts, per-epoch and total training time, peak
  GPU memory, and final evaluation metrics.

### Figure generation

All figures and summary tables in this report are generated by
[`experiments/generate_report_figures.py`](../experiments/generate_report_figures.py),
which reads only the JSON files under `results/raw/` and writes only to `figures/` and
`results/tables/` — it does not retrain anything and does not modify any existing
experiment output. Re-running it (`python -m experiments.generate_report_figures`) will
regenerate identical figures/tables from the same stored results.

---

## 12. Experimental Issues and Lessons Learned

### Learning-rate failure (AG News full fine-tuning)

The valid full-fine-tuning configuration for both datasets uses learning rate **2e-5**. An
initial attempt at fine-tuning AG News with the full model used learning rate **2e-4** —
the same rate used for the LoRA runs — rather than a full-fine-tuning-appropriate rate.
This run **collapsed**: it converged to predicting a single constant class for every
example, producing accuracy exactly at the random-guess level for AG News's balanced
four-class split (0.2500) and a correspondingly degenerate F1 (0.1000). Training loss
stayed flat near `ln(4) ≈ 1.386` — the entropy of a uniform four-way guess — across all
three epochs, rather than decreasing.

This failed run is **preserved on disk** (its result JSON and checkpoints were never
deleted or overwritten) as documented evidence, and is explicitly **not** used as a
baseline anywhere in this report. The corrected run, using `lr=2e-5` and otherwise
identical settings, trained normally (monotonically decreasing loss, steadily improving
accuracy) and is the AG News full-fine-tuning baseline used throughout Sections 6–10.

**Root cause and reproducibility lesson:** the project's `configs/lora.yaml` and
`configs/full_finetuning.yaml` files were **documentation-only** — they record the
intended hyperparameters for reference, but `src/train.py` never reads them. All actual
training parameters are supplied via CLI arguments at invocation time. The
`full_finetuning.yaml` config had, in fact, always specified the correct `2e-5` learning
rate; the failed run's `2e-4` was a manual deviation from that documented value that
nothing in the codebase caught before the run was launched. This is a concrete illustration
of the risk of documentation-only configuration: a config file that isn't the enforced
source of truth can silently drift from what's actually executed. A natural follow-up
(noted again in Section 15) is to have `train.py` load its defaults directly from these
YAML files, so a documented hyperparameter cannot diverge from an executed one without an
explicit, deliberate override.

### Timing anomalies

Several training runs, across both datasets, exhibited large single-epoch wall-clock time
spikes — periods where the recorded epoch duration was several times longer than that same
run's other epochs, or than comparable runs at the same rank/dataset. The most extreme case
was an AG News LoRA run (r=16, seed=123) whose second epoch took approximately **8 hours
17 minutes**, versus roughly 15–18 minutes for its other two epochs and for comparable
runs.

Every anomaly was independently verified against **checkpoint file modification
timestamps**: the elapsed wall-clock time between successive saved checkpoints matched the
anomalous duration recorded in the run's own epoch-timing log, confirming the gap was a
genuine wall-clock event rather than a bug in the timing instrumentation. The pattern is
consistent with the host machine entering sleep or suspend during those windows — the
training process was not killed, and **training resumed normally afterward, with metrics
remaining valid** (correct parameter counts, healthy accuracy/F1, sensible loss curves) in
every affected run.

These anomalies **should not be interpreted as reflecting model-computation scaling** —
they do not indicate that a given rank or dataset is computationally more expensive per
epoch. For this reason, no timing-vs-rank figure is included in the main report; the raw
per-run timing fields remain in the result JSONs and the full list of affected runs is
documented in `results/tables/summary_tables.md`'s reproducibility notes.

---

## 13. Limitations

1. **Only three seeds per rank** (42, 123, 999) were used for the LoRA sweep on each
   dataset — sufficient to estimate a mean and standard deviation, but a small sample for
   detecting subtle rank effects.
2. **Only one full-fine-tuning seed per dataset** (42) — the full-FT reference points/lines
   in every figure and table carry no seed-based error bar, unlike the LoRA results.
3. **Only two classification datasets** were evaluated (SST-2, AG News); both are
   English-language text classification tasks of a similar general type (sentence/short-
   document classification), differing mainly in class count.
4. **Only one backbone model** (RoBERTa-base, ~125M parameters) was tested; rank
   sensitivity could differ at other model scales or architectures.
5. **Rank range limited to {1, 2, 4, 8, 16}** — larger ranks, or fractional/intermediate
   ranks, were not tested.
6. **Wall-clock timing measurements were affected by host-machine suspension events** in
   several runs (Section 12); reported `train_time_seconds` and `epoch_times_seconds`
   values in the affected result JSONs are not representative of true computational cost
   and were excluded from this report's figures.
7. **No statistical significance testing beyond descriptive mean/std** was performed —
   claims in Sections 7–10 are based on comparing means and standard deviations across
   three-seed samples, not on formal hypothesis tests (e.g., no t-tests, confidence
   intervals, or multiple-comparison correction were computed).

---

## 14. Conclusion

LoRA achieves strong parameter efficiency on both SST-2 and AG News: at rank 1, it trains
under 0.51% of the parameters full fine-tuning would update, while reaching within roughly
0.7–1.4 accuracy points of the full-fine-tuning baseline on both datasets. Increasing rank
from 1 to 16 produces only marginal further gains — +0.19 percentage points on SST-2 and
+0.25 percentage points on AG News — and this improvement is small relative to the
run-to-run variability already present at a single rank. The central hypothesis motivating
the cross-dataset extension of this project — that the more complex four-class AG News task
would require substantially higher LoRA rank than the binary SST-2 task — is **not
supported** by these experiments: AG News's rank-sensitivity (rank-to-rank spread and seed
variance) is comparable to, or tighter than, SST-2's, even though AG News does show a
consistently larger absolute gap to its own full-fine-tuning ceiling. These findings are
reported as descriptive results from a three-seed experimental matrix, not as a
statistically definitive resolution of the underlying question.

---

## 15. Future Work

The following extensions could sharpen or stress-test the findings above. None of them are
presented as necessary to validate the current results — they would extend the scope of
what has been tested, not correct a known deficiency in what has already been measured.

- **More random seeds** per rank/dataset, to narrow the standard-error estimates on the
  already-small observed rank effect.
- **Multiple full-fine-tuning seeds** per dataset, so the full-FT reference points carry
  their own error bars, matching the rigor already applied to the LoRA sweep.
- **Additional datasets**, spanning a wider range of class counts, domains, and task
  difficulty, to test whether the "task complexity doesn't require higher rank" finding
  generalizes beyond SST-2 and AG News.
- **Larger backbone models**, to test whether rank sensitivity changes with model scale.
- **Broader LoRA rank ranges**, including ranks below 1 (via structured sparsification) or
  above 16, and non-power-of-two ranks.
- **Task types beyond classification** (e.g., generation, extractive QA, sequence
  labeling), where the low-rank update may need to capture different kinds of task
  adaptation.
- **Controlled hardware benchmarking**, run on dedicated infrastructure without background
  processes or sleep/suspend risk, to produce a reliable timing-vs-rank comparison — the
  anomalies documented in Section 12 currently preclude drawing any such conclusion from
  this project's data.
- **Enforcing YAML configs directly from code**, so that `configs/lora.yaml` and
  `configs/full_finetuning.yaml` become the actual source of truth for training runs rather
  than documentation that can silently drift from what is executed (Section 12).

---

## References

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W.
(2022). *LoRA: Low-Rank Adaptation of Large Language Models*. International Conference on
Learning Representations (ICLR 2022). https://arxiv.org/abs/2106.09685
