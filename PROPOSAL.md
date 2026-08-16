# Reproducing LoRA: Low-Rank Adaptation of Large Language Models

### Cross-Dataset Rank-Sensitivity Study

> **LebNet Tech Fellows 2026 — Option 1: Reproduce an Academic Paper + Extend It**

---

## 📌 Overview

This project reproduces the core methodology of:

> **Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," ICLR 2022**

and extends the evaluation with a **cross-dataset rank-sensitivity study**.

The project investigates whether very low-rank LoRA adapters are sufficient across different classification tasks, or whether task complexity affects the rank required to approach full fine-tuning performance.

### Main Objectives

1. Reproduce the core LoRA methodology.
2. Reproduce the SST-2 classification experiment.
3. Compare LoRA against full fine-tuning.
4. Evaluate LoRA across multiple ranks.
5. Extend the evaluation to AG News.
6. Study how rank affects performance across datasets.
7. Analyze the accuracy–efficiency trade-off.

---

## 🎯 Research Question

> **How does the LoRA rank affect the performance and efficiency of parameter-efficient fine-tuning across classification tasks with different levels of complexity?**

### Hypothesis

A low rank (`r ≤ 4`) should be sufficient for binary classification such as SST-2, while the more complex four-class AG News task may require a higher rank to match full fine-tuning performance.

This hypothesis will be tested experimentally rather than assumed.

---

## 🧠 Background

Traditional fine-tuning updates **all parameters** of a pretrained language model.

Although effective, this approach can be computationally expensive and memory-intensive. It also requires storing a separate full-sized model checkpoint for every downstream task.

LoRA addresses this problem by:

* Freezing the pretrained model weights.
* Injecting trainable low-rank matrices into selected Transformer layers.
* Training only these small matrices.
* Keeping the original pretrained weights unchanged.

The goal is to achieve performance comparable to full fine-tuning while substantially reducing the number of trainable parameters and memory requirements.

---

## 🔬 Methodology

The reproduction uses:

* **Backbone:** RoBERTa-base
* **Framework:** PyTorch
* **Libraries:** Hugging Face Transformers + PEFT
* **LoRA target modules:** Query and Value projections
* **Baseline:** Full fine-tuning
* **Reproduction dataset:** SST-2
* **Extension dataset:** AG News

The pretrained RoBERTa-base backbone remains frozen during LoRA training, while only the LoRA parameters are optimized.

---

## 📊 Experimental Design

### Rank Sweep

The following LoRA ranks will be evaluated:

```text
r ∈ {1, 2, 4, 8, 16}
```

Each configuration will be repeated using three random seeds:

```text
42
123
999
```

Therefore:

```text
5 ranks × 3 seeds = 15 runs per dataset
```

and:

```text
2 datasets × 15 runs = 30 LoRA experiments
```

The results will be reported as:

```text
mean ± standard deviation
```

This allows the experiment to distinguish systematic rank effects from run-to-run randomness.

---

## 📚 Datasets

| Dataset     | Task                     | Classes | Role               |
| ----------- | ------------------------ | ------: | ------------------ |
| **SST-2**   | Sentiment Classification |       2 | Paper reproduction |
| **AG News** | Topic Classification     |       4 | Extension          |

### SST-2

The Stanford Sentiment Treebank is a binary sentiment-classification benchmark distributed through the GLUE benchmark.

**Purpose:** reproduce the paper's parameter-efficient fine-tuning setup.

Dataset:

https://huggingface.co/datasets/glue

### AG News

AG News is a four-class news topic-classification dataset.

**Purpose:** test whether LoRA's rank requirements change when moving from binary sentiment classification to a four-class classification problem.

Dataset:

https://huggingface.co/datasets/ag_news

---

## ⚔️ Baselines

The project compares two approaches.

### 1. Full Fine-Tuning

All pretrained model parameters are trainable.

```text
RoBERTa-base
      ↓
All parameters trainable
      ↓
Classification task
```

### 2. LoRA

The pretrained backbone is frozen and only low-rank adaptation parameters are trained.

```text
RoBERTa-base (Frozen)
        +
LoRA adapters
        ↓
Classification task
```

---

## 📈 Evaluation Metrics

The experiments will evaluate both **task quality** and **computational efficiency**.

### Task Quality

* Accuracy
* F1-score
* Precision
* Recall

### Efficiency

* Trainable parameter count
* Peak GPU memory
* Training time per epoch

---

## 📋 Results

Results will be reported after the experiments are completed.

### SST-2

| Method           | Rank | Accuracy |  F1 | Trainable Params | GPU Memory | Time/Epoch |
| ---------------- | ---: | -------: | --: | ---------------: | ---------: | ---------: |
| Full Fine-Tuning |    — |      TBD | TBD |              TBD |        TBD |        TBD |
| LoRA             |    1 |      TBD | TBD |              TBD |        TBD |        TBD |
| LoRA             |    2 |      TBD | TBD |              TBD |        TBD |        TBD |
| LoRA             |    4 |      TBD | TBD |              TBD |        TBD |        TBD |
| LoRA             |    8 |      TBD | TBD |              TBD |        TBD |        TBD |
| LoRA             |   16 |      TBD | TBD |              TBD |        TBD |        TBD |

### AG News

| Method           | Rank | Accuracy |  F1 | Trainable Params | GPU Memory | Time/Epoch |
| ---------------- | ---: | -------: | --: | ---------------: | ---------: | ---------: |
| Full Fine-Tuning |    — |      TBD | TBD |              TBD |        TBD |        TBD |
| LoRA             |    1 |      TBD | TBD |              TBD |        TBD |        TBD |
| LoRA             |    2 |      TBD | TBD |              TBD |        TBD |        TBD |
| LoRA             |    4 |      TBD | TBD |              TBD |        TBD |        TBD |
| LoRA             |    8 |      TBD | TBD |              TBD |        TBD |        TBD |
| LoRA             |   16 |      TBD | TBD |              TBD |        TBD |        TBD |

---

## 📉 Planned Visualizations

The repository will contain the following figures:

### Accuracy vs. Rank

```text
Accuracy
   ↑
   │
   │             ●
   │         ●
   │      ●
   │   ●
   │ ●
   └──────────────────→ Rank
      1  2  4  8  16
```

### F1 vs. Rank

Performance as a function of LoRA rank.

### GPU Memory vs. Rank

Comparison of peak GPU memory consumption across ranks.

### Trainable Parameters vs. Rank

Shows how the number of trainable parameters changes with rank.

### Training Time vs. Rank

Measures the computational cost associated with increasing LoRA rank.

---

## 🗂️ Project Structure

```text
lora-reproduction/
│
├── README.md
├── requirements.txt
├── LICENSE
│
├── configs/
│   ├── lora.yaml
│   ├── full_finetuning.yaml
│   └── experiments.yaml
│
├── data/
│   └── README.md
│
├── src/
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── metrics.py
│   └── utils.py
│
├── experiments/
│   ├── run_sst2.py
│   ├── run_agnews.py
│   ├── rank_sweep.py
│   └── full_finetuning.py
│
├── results/
│   ├── raw/
│   ├── processed/
│   └── tables/
│
├── figures/
│   ├── accuracy_vs_rank.png
│   ├── f1_vs_rank.png
│   ├── memory_vs_rank.png
│   ├── parameters_vs_rank.png
│   └── training_time_vs_rank.png
│
├── checkpoints/
│   ├── sst2/
│   └── agnews/
│
└── report/
    └── technical_report.pdf
```

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/mutasimtaha07-hue/lora-project.git
cd lora-project
```

### Create a virtual environment

```bash
python -m venv .venv
```

### Activate it

#### Windows

```bash
.venv\Scripts\activate
```

#### Linux / macOS

```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Experiments

### SST-2 Reproduction

```bash
python experiments/run_sst2.py
```

### AG News Extension

```bash
python experiments/run_agnews.py
```

### Full Fine-Tuning Baseline

```bash
python experiments/full_finetuning.py
```

### Rank Sweep

```bash
python experiments/rank_sweep.py
```

The rank sweep evaluates:

```text
r = 1
r = 2
r = 4
r = 8
r = 16
```

across:

```text
seed = 42
seed = 123
seed = 999
```

---

## 🔁 Reproducibility

To make the experiments reproducible, the project will:

* Fix random seeds.
* Record the exact model configuration.
* Record LoRA hyperparameters.
* Pin library versions.
* Save experiment configurations.
* Store raw experimental results.
* Report mean ± standard deviation.
* Document hardware and software environments.

---

## ✅ Success Criteria

The project will be considered successful if it achieves the following:

### Reproduction

The SST-2 experiment should recover performance close to the reported target while substantially reducing the number of trainable parameters.

### Rank Sensitivity

The rank sweep should reveal how performance changes across:

```text
r = 1, 2, 4, 8, 16
```

and identify where performance begins to saturate.

### Efficiency

LoRA should demonstrate a substantial reduction in trainable parameters and GPU memory compared with full fine-tuning.

### Research Extension

The AG News experiment should provide evidence about whether the rank required for effective adaptation changes across tasks.

---

## ⚠️ Limitations & Risks

### Limited GPU Resources

If the available hardware cannot support the complete experiment:

1. Prioritize SST-2 reproduction.
2. Complete the most informative ranks.
3. Prioritize `r = 1, 4, 16`.
4. Use a smaller backbone only if necessary while preserving the methodology.

### Library / Version Differences

Differences between library versions may affect reproducibility.

To mitigate this:

* Pin dependency versions.
* Record the exact configuration.
* Document the execution environment.

### Time Constraints

The project is designed for completion within a short period.

The priority order is:

```text
SST-2 reproduction
        ↓
Validated pipeline
        ↓
AG News extension
        ↓
Rank sweep
        ↓
Full comparison
        ↓
Analysis & report
```

---

## 🧪 Optional Experiments

If sufficient time remains, the project may include:

* Adapter comparison
* LoRA alpha ablation
* LoRA dropout ablation

These experiments are considered **stretch goals** and are not required for the core project.

---

## 🗓️ Timeline

| Date                | Task                                                 |
| ------------------- | ---------------------------------------------------- |
| **Jul 26 – Jul 31** | Study paper, review implementation, prepare datasets |
| **Aug 1 – Aug 5**   | Build pipeline and reproduce SST-2 baseline          |
| **Aug 6 – Aug 10**  | Run rank sweep: `r = 1, 2, 4, 8, 16` × 3 seeds       |
| **Aug 11 – Aug 14** | Full fine-tuning vs. LoRA comparison                 |
| **Aug 15 – Aug 16** | Aggregate results and generate figures               |
| **Aug 17 – Aug 18** | Technical report + GitHub finalization               |
| **Aug 19**          | Demo video + submission                              |

---

## 📦 Expected Deliverables

* [ ] Reproducible Python implementation
* [ ] SST-2 LoRA reproduction
* [ ] AG News extension
* [ ] Full fine-tuning baseline
* [ ] Rank sweep
* [ ] Mean ± standard deviation results
* [ ] LoRA checkpoints
* [ ] Accuracy/F1 plots
* [ ] GPU-memory analysis
* [ ] Trainable-parameter analysis
* [ ] Training-time analysis
* [ ] Technical report
* [ ] GitHub repository
* [ ] 3-minute demonstration video

---

## 📚 References

### Original Paper

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W.

**LoRA: Low-Rank Adaptation of Large Language Models.**

ICLR 2022.

Paper:

https://arxiv.org/abs/2106.09685

### Official Implementation

Microsoft LoRA:

https://github.com/microsoft/LoRA

### Hugging Face

Transformers:

https://huggingface.co/docs/transformers

PEFT:

https://huggingface.co/docs/peft

### Datasets

GLUE / SST-2:

https://huggingface.co/datasets/glue

AG News:

https://huggingface.co/datasets/ag_news

---

## 📝 Citation

If you use this repository or build upon this reproduction, please cite the original LoRA paper:

```bibtex
@inproceedings{hu2022lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J. and Shen, Yelong and Wallis, Phillip and
          Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and
          Wang, Lu and Chen, Weizhu},
  booktitle={International Conference on Learning Representations},
  year={2022}
}
```

---

## 👤 Author

**Mutasem Usama Taha**

LebNet Tech Fellows 2026

**Project:** Reproducing LoRA with a Cross-Dataset Rank-Sensitivity Study

**Project Type:** Option 1 — Reproduce an Academic Paper + Extend It

---

## ⭐ Project Status

```text
🟡 In Progress
```

The repository is currently under development. Experimental results will be added as the reproduction and extension experiments are completed.

---
