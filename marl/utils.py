"""Utilities shared across the MAPPO modules."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Seed numpy, torch (CPU + CUDA), and Python's random module."""
    import random as _random

    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Fraction of variance in ``y_true`` explained by ``y_pred``."""
    var_y = torch.var(y_true)
    if var_y == 0:
        return float("nan")
    return float(1.0 - torch.var(y_true - y_pred) / var_y)


def flatten_iterable(it: Iterable[float]) -> np.ndarray:
    return np.asarray(list(it), dtype=np.float32)


class PopArt(torch.nn.Module):
    """PopArt value normalisation (Hessel et al., 2016).

    Maintains a running mean/std of value targets and rescales the
    *unnormalised* value head outputs. The MAPPO critic uses this to
    keep targets in a stable range without clipping the policy's
    gradient.
    """

    def __init__(self, beta: float = 5e-4, epsilon: float = 1e-5) -> None:
        super().__init__()
        self.beta = beta
        self.epsilon = epsilon
        self.register_buffer("mean", torch.zeros(1))
        self.register_buffer("mean_sq", torch.ones(1))
        self.register_buffer("debias", torch.zeros(1))

    @property
    def std(self) -> torch.Tensor:
        var = self.mean_sq - self.mean ** 2
        var = torch.nan_to_num(var, nan=self.epsilon, posinf=1.0, neginf=self.epsilon)
        return torch.sqrt(torch.clamp(var, min=self.epsilon))

    def update(self, x: torch.Tensor) -> None:
        with torch.no_grad():
            # Self-heal if state was corrupted (e.g. inf loaded from an old
            # checkpoint with the legacy double-normalised loss). An inf EMA
            # cannot decay back to a finite value, so reset rather than carry
            # the poison forward.
            if not (torch.isfinite(self.mean).all() and torch.isfinite(self.mean_sq).all()):
                self.mean.zero_()
                self.mean_sq.fill_(1.0)
                self.debias.zero_()
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            batch_mean = x.mean()
            batch_mean_sq = (x ** 2).mean()
            if not (torch.isfinite(batch_mean) and torch.isfinite(batch_mean_sq)):
                return
            self.debias.mul_(1 - self.beta).add_(self.beta * 1.0)
            self.mean.mul_(1 - self.beta).add_(self.beta * batch_mean)
            self.mean_sq.mul_(1 - self.beta).add_(self.beta * batch_mean_sq)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if float(self.debias) < 1e-6:
            return x
        return (x - self.mean) / self.std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if float(self.debias) < 1e-6:
            return x
        return x * self.std + self.mean
