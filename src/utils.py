"""Reproducibility helpers and small shared utilities."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed every RNG we rely on so a run is reproducible given the same seed.

    Covers Python's `random`, NumPy, and PyTorch (CPU + all CUDA devices).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Number of parameters with requires_grad=True (what LoRA is meant to shrink)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model: torch.nn.Module) -> int:
    """Total parameter count, trainable or not (the full backbone + heads/adapters)."""
    return sum(p.numel() for p in model.parameters())


def get_device() -> torch.device:
    """Best available device: CUDA if PyTorch can see a GPU, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reset_peak_gpu_memory_stats() -> None:
    """Call before a training run so peak-memory measurements start from zero."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_gpu_memory_bytes() -> int | None:
    """Peak CUDA memory allocated since the last reset, or None if no GPU is present."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated()
    return None
