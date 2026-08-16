# LoRA Rank Sweep — Summary Tables

All figures computed directly from the result JSONs in `results/raw/`. LoRA statistics are mean ± sample standard deviation across seeds {42, 123, 999} (n=3).


## Table 1 — AG News

| Rank | Accuracy (mean ± std) | F1 (mean ± std) | Trainable params | Trainable % |
|---|---|---|---|---|
| 1 | 0.9408 ± 0.0009 | 0.9407 ± 0.0009 | 630,532 | 0.503% |
| 2 | 0.9421 ± 0.0005 | 0.9421 ± 0.0005 | 667,396 | 0.533% |
| 4 | 0.9428 ± 0.0006 | 0.9427 ± 0.0006 | 741,124 | 0.591% |
| 8 | 0.9426 ± 0.0003 | 0.9426 ± 0.0003 | 888,580 | 0.708% |
| 16 | 0.9433 ± 0.0003 | 0.9432 ± 0.0003 | 1,183,492 | 0.941% |

## Table 2 — SST-2

| Rank | Accuracy (mean ± std) | F1 (mean ± std) | Trainable params | Trainable % |
|---|---|---|---|---|
| 1 | 0.9354 ± 0.0029 | 0.9370 ± 0.0031 | 628,994 | 0.502% |
| 2 | 0.9346 ± 0.0041 | 0.9359 ± 0.0041 | 665,858 | 0.531% |
| 4 | 0.9362 ± 0.0026 | 0.9377 ± 0.0024 | 739,586 | 0.590% |
| 8 | 0.9365 ± 0.0048 | 0.9379 ± 0.0053 | 887,042 | 0.707% |
| 16 | 0.9373 ± 0.0029 | 0.9387 ± 0.0031 | 1,181,954 | 0.939% |

## Table 3 — Cross-Dataset Comparison

| Rank | SST-2 accuracy | AG News accuracy | SST-2 std | AG News std |
|---|---|---|---|---|
| 1 | 0.9354 | 0.9408 | 0.0029 | 0.0009 |
| 2 | 0.9346 | 0.9421 | 0.0041 | 0.0005 |
| 4 | 0.9362 | 0.9428 | 0.0026 | 0.0006 |
| 8 | 0.9365 | 0.9426 | 0.0048 | 0.0003 |
| 16 | 0.9373 | 0.9433 | 0.0029 | 0.0003 |

## Table 4 — Full Fine-Tuning Comparison

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

Full-FT references (single seed 42 each, no error bar): SST-2 accuracy=0.9438, F1=0.9454; AG News (valid, lr=2e-5) accuracy=0.9542, F1=0.9542.


## Reproducibility Notes

- **AG News full fine-tuning, lr=2e-4 (preserved as historical failure evidence only, NOT used as a baseline anywhere above):** collapsed to constant-class prediction (accuracy=0.2500, F1=0.1000 — exactly the values expected from always predicting one class on a balanced 4-class set). Root cause: learning rate too high for full-parameter fine-tuning. Corrected run used lr=2e-5, matching SST-2's full-FT recipe, and is the only AG News full-FT baseline used in this report.
- **Timing anomalies:** several LoRA runs (AG News r=4/seed=42, r=4/seed=123, r=4/seed=999, r=16/seed=123; and multiple SST-2 runs) showed single-epoch wall-clock spikes ranging from ~1.4× normal up to ~29× normal (~8h17m in the most extreme case, AG News r=16/seed=123). Every anomaly was independently checkpoint-verified (file modification timestamps match the recorded epoch durations) as genuine wall-clock gaps consistent with host-machine sleep/suspend events — not computation bugs. All affected runs completed with valid, healthy metrics and exactly-correct parameter counts. `train_time_seconds` / `epoch_times_seconds` in the affected result JSONs should not be interpreted as computational complexity and are excluded from the figures in this report (no timing-vs-rank figure was generated, per the decision to avoid a misleading plot).
- **Full fine-tuning baselines are single-seed (seed 42 only)** for both datasets — unlike the LoRA sweep, there is no seed-based error bar on the full-FT reference points/lines in these figures and tables.