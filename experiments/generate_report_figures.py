"""Generate publication-quality figures and summary tables for the final
technical report, from the completed SST-2 / AG News LoRA rank-sweep +
full-fine-tuning results already on disk.

Strictly read-only with respect to `results/raw/` and `checkpoints/` — this
script only *reads* result JSONs and *writes* new files under `figures/`
and `results/tables/`. It does not retrain anything and does not modify or
delete any existing experiment output.

Usage:
    python -m experiments.generate_report_figures
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless, no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

RESULTS_DIR = Path("results/raw")
TABLES_DIR = Path("results/tables")
FIGURES_DIR = Path("figures")

RANKS = [1, 2, 4, 8, 16]
SEEDS = [42, 123, 999]

# Consistent color coding across every figure.
COLOR_SST2 = "#1f77b4"      # blue
COLOR_AGNEWS = "#d62728"    # red
COLOR_FULLFT = "#2ca02c"    # green — reserved for full fine-tuning references
COLOR_FULLFT_FAILED = "#7f7f7f"  # gray — historical failure evidence only


# --------------------------------------------------------------------------
# Data loading (read-only)
# --------------------------------------------------------------------------

def load_sweep(dataset: str) -> dict[tuple[int, int], dict]:
    data = {}
    for r in RANKS:
        for s in SEEDS:
            fp = RESULTS_DIR / f"{dataset}_lora_r{r}_seed{s}.json"
            data[(r, s)] = json.loads(fp.read_text())
    return data


def stats_by_rank(data: dict[tuple[int, int], dict]) -> dict[int, dict]:
    out = {}
    for r in RANKS:
        runs = [data[(r, s)] for s in SEEDS]
        acc = [x["eval_metrics"]["eval_accuracy"] for x in runs]
        f1 = [x["eval_metrics"]["eval_f1"] for x in runs]
        prec = [x["eval_metrics"]["eval_precision"] for x in runs]
        rec = [x["eval_metrics"]["eval_recall"] for x in runs]
        trainable = runs[0]["trainable_params"]
        total = runs[0]["total_params"]
        out[r] = dict(
            acc_mean=statistics.mean(acc), acc_std=statistics.stdev(acc),
            f1_mean=statistics.mean(f1), f1_std=statistics.stdev(f1),
            prec_mean=statistics.mean(prec), prec_std=statistics.stdev(prec),
            rec_mean=statistics.mean(rec), rec_std=statistics.stdev(rec),
            trainable_params=trainable, total_params=total,
            trainable_pct=100 * trainable / total,
        )
    return out


sst2 = load_sweep("sst2")
agnews = load_sweep("ag_news")
sst2_stats = stats_by_rank(sst2)
agnews_stats = stats_by_rank(agnews)

sst2_ft = json.loads((RESULTS_DIR / "sst2_full_seed42.json").read_text())
agnews_ft = json.loads((RESULTS_DIR / "ag_news_lr_fix" / "ag_news_full_seed42.json").read_text())
# Preserved historical failure evidence only — NOT used as a valid baseline anywhere below.
agnews_ft_failed = json.loads((RESULTS_DIR / "ag_news_full_seed42.json").read_text())

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Figure 1 — Accuracy vs LoRA Rank
# --------------------------------------------------------------------------

def fig_accuracy_vs_rank():
    fig, ax = plt.subplots(figsize=(7, 5))
    x = RANKS

    sst2_means = [sst2_stats[r]["acc_mean"] for r in RANKS]
    sst2_stds = [sst2_stats[r]["acc_std"] for r in RANKS]
    agnews_means = [agnews_stats[r]["acc_mean"] for r in RANKS]
    agnews_stds = [agnews_stats[r]["acc_std"] for r in RANKS]

    ax.errorbar(x, sst2_means, yerr=sst2_stds, marker="o", capsize=4,
                color=COLOR_SST2, label="SST-2 (mean ± std, n=3 seeds)", linewidth=2)
    ax.errorbar(x, agnews_means, yerr=agnews_stds, marker="s", capsize=4,
                color=COLOR_AGNEWS, label="AG News (mean ± std, n=3 seeds)", linewidth=2)

    ax.set_xscale("log", base=2)
    ax.set_xticks(RANKS)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("LoRA rank (r)")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Accuracy vs. LoRA Rank\n(RoBERTa-base, Q/V target modules)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "accuracy_vs_rank.png", dpi=180)
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 2 — F1 vs LoRA Rank
# --------------------------------------------------------------------------

def fig_f1_vs_rank():
    fig, ax = plt.subplots(figsize=(7, 5))
    x = RANKS

    sst2_means = [sst2_stats[r]["f1_mean"] for r in RANKS]
    sst2_stds = [sst2_stats[r]["f1_std"] for r in RANKS]
    agnews_means = [agnews_stats[r]["f1_mean"] for r in RANKS]
    agnews_stds = [agnews_stats[r]["f1_std"] for r in RANKS]

    ax.errorbar(x, sst2_means, yerr=sst2_stds, marker="o", capsize=4,
                color=COLOR_SST2, label="SST-2 (mean ± std, n=3 seeds)", linewidth=2)
    ax.errorbar(x, agnews_means, yerr=agnews_stds, marker="s", capsize=4,
                color=COLOR_AGNEWS, label="AG News (mean ± std, n=3 seeds)", linewidth=2)

    ax.set_xscale("log", base=2)
    ax.set_xticks(RANKS)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("LoRA rank (r)")
    ax.set_ylabel("Validation F1 (binary for SST-2, macro for AG News)")
    ax.set_title("F1 vs. LoRA Rank\n(RoBERTa-base, Q/V target modules)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "f1_vs_rank.png", dpi=180)
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 3 — LoRA vs Full Fine-Tuning
# --------------------------------------------------------------------------

def fig_lora_vs_full_ft():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    categories = [f"r={r}" for r in RANKS] + ["Full FT"]
    n_cat = len(categories)
    bar_width = 0.35
    x = range(n_cat)

    sst2_vals = [sst2_stats[r]["acc_mean"] for r in RANKS] + [sst2_ft["eval_metrics"]["eval_accuracy"]]
    sst2_errs = [sst2_stats[r]["acc_std"] for r in RANKS] + [0]
    agnews_vals = [agnews_stats[r]["acc_mean"] for r in RANKS] + [agnews_ft["eval_metrics"]["eval_accuracy"]]
    agnews_errs = [agnews_stats[r]["acc_std"] for r in RANKS] + [0]

    x_sst2 = [i - bar_width / 2 for i in x]
    x_agnews = [i + bar_width / 2 for i in x]

    # LoRA bars (first 5 categories) in normal dataset colors; Full-FT bar (last
    # category) rendered with a hatch pattern + distinct edge to set it apart.
    bars_sst2 = ax.bar(x_sst2, sst2_vals, bar_width, yerr=sst2_errs, capsize=3,
                        color=[COLOR_SST2] * 5 + ["white"],
                        edgecolor=COLOR_SST2, linewidth=1.5,
                        hatch=[None] * 5 + ["///"],
                        label="SST-2")
    bars_agnews = ax.bar(x_agnews, agnews_vals, bar_width, yerr=agnews_errs, capsize=3,
                          color=[COLOR_AGNEWS] * 5 + ["white"],
                          edgecolor=COLOR_AGNEWS, linewidth=1.5,
                          hatch=[None] * 5 + ["///"],
                          label="AG News")

    ax.set_xticks(list(x))
    ax.set_xticklabels(categories)
    ax.set_ylabel("Validation accuracy")
    ax.set_ylim(0.9, 0.97)
    ax.set_title("LoRA (r=1..16) vs. Full Fine-Tuning\nHatched bars = Full FT, single seed (42) — no error bar")
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)

    ax.axvline(x=n_cat - 1.5, color="black", linestyle=":", alpha=0.5, linewidth=1)
    ax.annotate("Full FT\n(single seed,\nno std dev)",
                xy=(n_cat - 1, min(sst2_vals[-1], agnews_vals[-1]) - 0.005),
                xytext=(n_cat - 1, 0.905),
                ha="center", fontsize=8, color="black",
                arrowprops=dict(arrowstyle="->", alpha=0.6))

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "lora_vs_full_ft.png", dpi=180)
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 4 — Trainable Parameters vs Rank (log Y-axis)
# --------------------------------------------------------------------------

def fig_trainable_params_vs_rank():
    fig, ax = plt.subplots(figsize=(7, 5))
    x = RANKS

    sst2_params = [sst2_stats[r]["trainable_params"] for r in RANKS]
    agnews_params = [agnews_stats[r]["trainable_params"] for r in RANKS]

    ax.plot(x, sst2_params, marker="o", color=COLOR_SST2, linewidth=2,
            label="SST-2 LoRA trainable params")
    ax.plot(x, agnews_params, marker="s", color=COLOR_AGNEWS, linewidth=2,
            linestyle="--", label="AG News LoRA trainable params")

    full_ft_params = sst2_ft["trainable_params"]  # 124,647,170 (AG News: 124,648,708 — nearly identical)
    ax.axhline(y=full_ft_params, color=COLOR_FULLFT, linestyle="-.", linewidth=2,
               label=f"Full fine-tuning: {full_ft_params:,} params (both datasets, ~equal)")

    ax.set_xscale("log", base=2)
    ax.set_xticks(RANKS)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_yscale("log")
    ax.set_xlabel("LoRA rank (r)")
    ax.set_ylabel("Trainable parameters (log scale)")
    ax.set_title("Trainable Parameters vs. Rank\n(Y-axis is logarithmic — full-FT is ~100–200× larger than any LoRA rank)")
    ax.legend(loc="center right", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    for r, p in zip(RANKS, agnews_params):
        ax.annotate(f"{p:,}", xy=(r, p), xytext=(0, -12), textcoords="offset points",
                    ha="center", fontsize=7, color=COLOR_AGNEWS)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "trainable_params_vs_rank.png", dpi=180)
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 5 — Parameter Efficiency vs Accuracy
# --------------------------------------------------------------------------

def fig_parameter_efficiency():
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    sst2_pct = [sst2_stats[r]["trainable_pct"] for r in RANKS]
    sst2_acc = [sst2_stats[r]["acc_mean"] for r in RANKS]
    agnews_pct = [agnews_stats[r]["trainable_pct"] for r in RANKS]
    agnews_acc = [agnews_stats[r]["acc_mean"] for r in RANKS]

    ax.plot(sst2_pct, sst2_acc, marker="o", color=COLOR_SST2, linewidth=2,
            label="SST-2 LoRA (r=1..16)")
    ax.plot(agnews_pct, agnews_acc, marker="s", color=COLOR_AGNEWS, linewidth=2,
            label="AG News LoRA (r=1..16)")

    for r, px, py in zip(RANKS, sst2_pct, sst2_acc):
        ax.annotate(f"r={r}", xy=(px, py), xytext=(4, 4), textcoords="offset points", fontsize=7)
    for r, px, py in zip(RANKS, agnews_pct, agnews_acc):
        ax.annotate(f"r={r}", xy=(px, py), xytext=(4, -10), textcoords="offset points", fontsize=7)

    # Full-FT reference points at 100% trainable params.
    ax.scatter([100], [sst2_ft["eval_metrics"]["eval_accuracy"]], marker="*", s=220,
               color=COLOR_FULLFT, edgecolor="black", zorder=5,
               label="SST-2 Full FT (100%, single seed)")
    ax.scatter([100], [agnews_ft["eval_metrics"]["eval_accuracy"]], marker="P", s=180,
               color=COLOR_FULLFT, edgecolor="black", zorder=5,
               label="AG News Full FT (100%, single seed)")

    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (% of full fine-tuning, log scale)")
    ax.set_ylabel("Mean validation accuracy")
    ax.set_title("Parameter Efficiency: Accuracy vs. % of Full-FT Parameters Trained")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "parameter_efficiency.png", dpi=180)
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 6 — Rank Sensitivity Comparison (change relative to r=1)
# --------------------------------------------------------------------------

def fig_rank_sensitivity():
    fig, ax = plt.subplots(figsize=(7, 5))
    x = RANKS

    sst2_delta = [sst2_stats[r]["acc_mean"] - sst2_stats[1]["acc_mean"] for r in RANKS]
    agnews_delta = [agnews_stats[r]["acc_mean"] - agnews_stats[1]["acc_mean"] for r in RANKS]

    ax.plot(x, sst2_delta, marker="o", color=COLOR_SST2, linewidth=2, label="SST-2")
    ax.plot(x, agnews_delta, marker="s", color=COLOR_AGNEWS, linewidth=2, label="AG News")
    ax.axhline(y=0, color="black", linewidth=1, alpha=0.6)

    ax.set_xscale("log", base=2)
    ax.set_xticks(RANKS)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_ylabel("Δ mean accuracy vs. r=1")
    ax.set_xlabel("LoRA rank (r)")
    ax.set_ylim(-0.006, 0.006)
    ax.set_title("Rank Sensitivity: Accuracy Change Relative to r=1\n(note the very small y-axis range — the rank effect is small)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "rank_sensitivity.png", dpi=180)
    plt.close(fig)


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------

def write_csv(path: Path, header: list[str], rows: list[list]):
    # Explicit UTF-8: on Windows, open()'s default encoding follows the
    # system codepage, which mangles non-ASCII characters like "±" into
    # replacement characters. All tables use "±" in the mean±std columns.
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def generate_tables():
    # Table 1 — AG News
    t1_rows = [
        [r, f"{agnews_stats[r]['acc_mean']:.4f} ± {agnews_stats[r]['acc_std']:.4f}",
         f"{agnews_stats[r]['f1_mean']:.4f} ± {agnews_stats[r]['f1_std']:.4f}",
         agnews_stats[r]["trainable_params"], f"{agnews_stats[r]['trainable_pct']:.3f}%"]
        for r in RANKS
    ]
    write_csv(TABLES_DIR / "table1_agnews_summary.csv",
              ["rank", "accuracy_mean_std", "f1_mean_std", "trainable_params", "trainable_pct"], t1_rows)

    # Table 2 — SST-2
    t2_rows = [
        [r, f"{sst2_stats[r]['acc_mean']:.4f} ± {sst2_stats[r]['acc_std']:.4f}",
         f"{sst2_stats[r]['f1_mean']:.4f} ± {sst2_stats[r]['f1_std']:.4f}",
         sst2_stats[r]["trainable_params"], f"{sst2_stats[r]['trainable_pct']:.3f}%"]
        for r in RANKS
    ]
    write_csv(TABLES_DIR / "table2_sst2_summary.csv",
              ["rank", "accuracy_mean_std", "f1_mean_std", "trainable_params", "trainable_pct"], t2_rows)

    # Table 3 — Cross-dataset comparison
    t3_rows = [
        [r, f"{sst2_stats[r]['acc_mean']:.4f}", f"{agnews_stats[r]['acc_mean']:.4f}",
         f"{sst2_stats[r]['acc_std']:.4f}", f"{agnews_stats[r]['acc_std']:.4f}"]
        for r in RANKS
    ]
    write_csv(TABLES_DIR / "table3_cross_dataset_comparison.csv",
              ["rank", "sst2_accuracy_mean", "agnews_accuracy_mean", "sst2_std", "agnews_std"], t3_rows)

    # Table 4 — Full fine-tuning comparison
    t4_rows = []
    for r in RANKS:
        gap = sst2_ft["eval_metrics"]["eval_accuracy"] - sst2_stats[r]["acc_mean"]
        t4_rows.append(["SST-2", r, f"{sst2_stats[r]['acc_mean']:.4f}", f"{gap:.4f}",
                         sst2_stats[r]["trainable_params"], f"{sst2_stats[r]['trainable_pct']:.3f}%"])
    for r in RANKS:
        gap = agnews_ft["eval_metrics"]["eval_accuracy"] - agnews_stats[r]["acc_mean"]
        t4_rows.append(["AG News", r, f"{agnews_stats[r]['acc_mean']:.4f}", f"{gap:.4f}",
                         agnews_stats[r]["trainable_params"], f"{agnews_stats[r]['trainable_pct']:.3f}%"])
    write_csv(TABLES_DIR / "table4_full_ft_comparison.csv",
              ["dataset", "lora_rank", "accuracy", "gap_to_full_ft", "trainable_params", "trainable_pct"], t4_rows)

    # Combined Markdown report with all 4 tables + reproducibility notes.
    md = []
    md.append("# LoRA Rank Sweep — Summary Tables\n")
    md.append("All figures computed directly from the result JSONs in `results/raw/`. "
              "LoRA statistics are mean ± sample standard deviation across seeds {42, 123, 999} (n=3).\n")

    md.append("\n## Table 1 — AG News\n")
    md.append("| Rank | Accuracy (mean ± std) | F1 (mean ± std) | Trainable params | Trainable % |")
    md.append("|---|---|---|---|---|")
    for r in RANKS:
        d = agnews_stats[r]
        md.append(f"| {r} | {d['acc_mean']:.4f} ± {d['acc_std']:.4f} | {d['f1_mean']:.4f} ± {d['f1_std']:.4f} | "
                   f"{d['trainable_params']:,} | {d['trainable_pct']:.3f}% |")

    md.append("\n## Table 2 — SST-2\n")
    md.append("| Rank | Accuracy (mean ± std) | F1 (mean ± std) | Trainable params | Trainable % |")
    md.append("|---|---|---|---|---|")
    for r in RANKS:
        d = sst2_stats[r]
        md.append(f"| {r} | {d['acc_mean']:.4f} ± {d['acc_std']:.4f} | {d['f1_mean']:.4f} ± {d['f1_std']:.4f} | "
                   f"{d['trainable_params']:,} | {d['trainable_pct']:.3f}% |")

    md.append("\n## Table 3 — Cross-Dataset Comparison\n")
    md.append("| Rank | SST-2 accuracy | AG News accuracy | SST-2 std | AG News std |")
    md.append("|---|---|---|---|---|")
    for r in RANKS:
        md.append(f"| {r} | {sst2_stats[r]['acc_mean']:.4f} | {agnews_stats[r]['acc_mean']:.4f} | "
                   f"{sst2_stats[r]['acc_std']:.4f} | {agnews_stats[r]['acc_std']:.4f} |")

    md.append("\n## Table 4 — Full Fine-Tuning Comparison\n")
    md.append("| Dataset | LoRA rank | Accuracy | Gap to full FT | Trainable params | Trainable % |")
    md.append("|---|---|---|---|---|---|")
    for row in t4_rows:
        dataset, r, acc, gap, params, pct = row
        md.append(f"| {dataset} | {r} | {acc} | {gap} | {int(params):,} | {pct} |")

    md.append(f"\nFull-FT references (single seed 42 each, no error bar): "
              f"SST-2 accuracy={sst2_ft['eval_metrics']['eval_accuracy']:.4f}, "
              f"F1={sst2_ft['eval_metrics']['eval_f1']:.4f}; "
              f"AG News (valid, lr=2e-5) accuracy={agnews_ft['eval_metrics']['eval_accuracy']:.4f}, "
              f"F1={agnews_ft['eval_metrics']['eval_f1']:.4f}.\n")

    md.append("\n## Reproducibility Notes\n")
    md.append(
        "- **AG News full fine-tuning, lr=2e-4 (preserved as historical failure evidence only, "
        "NOT used as a baseline anywhere above):** collapsed to constant-class prediction "
        f"(accuracy={agnews_ft_failed['eval_metrics']['eval_accuracy']:.4f}, "
        f"F1={agnews_ft_failed['eval_metrics']['eval_f1']:.4f} — exactly the values expected from "
        "always predicting one class on a balanced 4-class set). Root cause: learning rate too high "
        "for full-parameter fine-tuning. Corrected run used lr=2e-5, matching SST-2's full-FT recipe, "
        "and is the only AG News full-FT baseline used in this report."
    )
    md.append(
        "- **Timing anomalies:** several LoRA runs (AG News r=4/seed=42, r=4/seed=123, r=4/seed=999, "
        "r=16/seed=123; and multiple SST-2 runs) showed single-epoch wall-clock spikes ranging from "
        "~1.4× normal up to ~29× normal (~8h17m in the most extreme case, AG News r=16/seed=123). "
        "Every anomaly was independently checkpoint-verified (file modification timestamps match the "
        "recorded epoch durations) as genuine wall-clock gaps consistent with host-machine sleep/suspend "
        "events — not computation bugs. All affected runs completed with valid, healthy metrics and "
        "exactly-correct parameter counts. `train_time_seconds` / `epoch_times_seconds` in the affected "
        "result JSONs should not be interpreted as computational complexity and are excluded from the "
        "figures in this report (no timing-vs-rank figure was generated, per the decision to avoid a "
        "misleading plot)."
    )
    md.append(
        "- **Full fine-tuning baselines are single-seed (seed 42 only)** for both datasets — unlike the "
        "LoRA sweep, there is no seed-based error bar on the full-FT reference points/lines in these "
        "figures and tables."
    )

    (TABLES_DIR / "summary_tables.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    fig_accuracy_vs_rank()
    fig_f1_vs_rank()
    fig_lora_vs_full_ft()
    fig_trainable_params_vs_rank()
    fig_parameter_efficiency()
    fig_rank_sensitivity()
    generate_tables()
    print("Done. Figures written to figures/, tables written to results/tables/.")
