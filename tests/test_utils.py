"""Tests for src/utils.py.

Covers parameter counting on a tiny hand-built `nn.Module`, within-process
seed reproducibility, and the GPU-memory helpers. None of this requires a
GPU: the seed tests only check that the *same process*, given the same seed
twice, draws the same sequence of values -- they make no claim about
cross-machine or cross-GPU bitwise determinism, which PyTorch does not
guarantee even with all seeds fixed.
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn

from src.utils import (
    count_total_parameters,
    count_trainable_parameters,
    get_device,
    peak_gpu_memory_bytes,
    reset_peak_gpu_memory_stats,
    set_seed,
)


# --- Parameter counting ------------------------------------------------------


class _TinyModel(nn.Module):
    """A toy module with a known, hand-computable parameter count."""

    def __init__(self):
        super().__init__()
        self.trainable_layer = nn.Linear(4, 2)  # 4*2 + 2 = 10 params
        self.frozen_layer = nn.Linear(2, 1)  # 2*1 + 1 = 3 params

    def forward(self, x):
        return self.trainable_layer(x)


def _build_tiny_model_with_one_frozen_layer() -> _TinyModel:
    model = _TinyModel()
    for p in model.frozen_layer.parameters():
        p.requires_grad = False
    return model


def test_count_trainable_and_total_parameters():
    model = _build_tiny_model_with_one_frozen_layer()

    assert count_total_parameters(model) == 13  # 10 + 3
    assert count_trainable_parameters(model) == 10  # frozen_layer excluded


def test_count_trainable_parameters_is_zero_when_everything_is_frozen():
    model = _build_tiny_model_with_one_frozen_layer()
    for p in model.parameters():
        p.requires_grad = False

    assert count_trainable_parameters(model) == 0
    assert count_total_parameters(model) == 13  # total is unaffected by requires_grad


# --- Seed handling (within-process reproducibility only) --------------------


def _draw_values():
    """One value from each RNG `set_seed` claims to control."""
    return (
        random.random(),
        float(np.random.rand()),
        torch.rand(3).tolist(),
    )


def test_set_seed_is_reproducible_within_the_same_process():
    set_seed(42)
    first_draw = _draw_values()

    set_seed(42)
    second_draw = _draw_values()

    assert first_draw == second_draw


def test_set_seed_different_seeds_produce_different_draws():
    set_seed(42)
    draw_a = _draw_values()

    set_seed(123)
    draw_b = _draw_values()

    assert draw_a != draw_b


# --- Device / GPU-memory helpers --------------------------------------------


def test_get_device_matches_actual_cuda_availability():
    device = get_device()
    assert isinstance(device, torch.device)
    expected_type = "cuda" if torch.cuda.is_available() else "cpu"
    assert device.type == expected_type


def test_gpu_memory_helpers_do_not_raise_and_return_sensible_types():
    # No assertion is made about the *value* of peak memory usage, and none
    # about determinism across runs/hardware -- only that these helpers are
    # safe to call and return a type consistent with GPU availability.
    reset_peak_gpu_memory_stats()  # must not raise, whether or not a GPU is present
    result = peak_gpu_memory_bytes()

    if torch.cuda.is_available():
        assert isinstance(result, int)
        assert result >= 0
    else:
        assert result is None
