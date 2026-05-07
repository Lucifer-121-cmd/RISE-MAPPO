"""Per-robot sparse variational GP for hazard / occupancy modelling.

This is the Phase-2 STEP 3 module. Each robot maintains a small
GPyTorch SVGP that maps 2-D positions to scalar hazard values:

    h(x) = GP(0, k_RBF(x, x'))    with Gaussian likelihood

The number of inducing points is fixed (default 50) so the per-robot
posterior summary that gets shared with the BCM fuser
(:class:`gp.distributed_gp.DistributedGP`) has bounded bandwidth
``O(n_inducing)`` regardless of how many observations the robot has
ingested. ``predict`` is fast (vectorised over query points);
``update`` retrains the variational parameters for a small, fixed
number of epochs to keep tick latency bounded.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch


@dataclass
class LocalGPConfig:
    """Configuration for :class:`LocalGP`."""

    n_inducing: int = 50
    kernel_lengthscale: float = 1.0
    kernel_variance: float = 1.0
    noise_variance: float = 0.05
    learning_rate: float = 0.05
    n_epochs: int = 30
    batch_size: int = 256
    max_train_points: int = 1024
    device: str = "cpu"
    world_size: float = 10.0


def _build_svgp_modules(cfg: LocalGPConfig) -> Tuple[torch.nn.Module, "torch.nn.Module"]:
    """Construct a GPyTorch SVGP model + Gaussian likelihood lazily."""
    import gpytorch
    from gpytorch.models import ApproximateGP
    from gpytorch.variational import (
        CholeskyVariationalDistribution,
        VariationalStrategy,
    )

    inducing = torch.linspace(0.0, cfg.world_size, int(np.sqrt(cfg.n_inducing)) + 1)
    grid = torch.stack(torch.meshgrid(inducing, inducing, indexing="ij"), dim=-1).reshape(-1, 2)
    Z = grid[: cfg.n_inducing].to(cfg.device)

    class _SVGP(ApproximateGP):
        def __init__(self, inducing_points: torch.Tensor) -> None:
            var_dist = CholeskyVariationalDistribution(inducing_points.shape[0])
            var_strat = VariationalStrategy(
                self, inducing_points, var_dist, learn_inducing_locations=True,
            )
            super().__init__(var_strat)
            self.mean_module = gpytorch.means.ConstantMean()
            base_kernel = gpytorch.kernels.RBFKernel()
            base_kernel.lengthscale = cfg.kernel_lengthscale
            self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
            self.covar_module.outputscale = cfg.kernel_variance

        def forward(self, x: torch.Tensor):
            return gpytorch.distributions.MultivariateNormal(
                self.mean_module(x), self.covar_module(x),
            )

    model = _SVGP(Z).to(cfg.device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(cfg.device)
    likelihood.noise = cfg.noise_variance
    return model, likelihood


class LocalGP:
    """Sparse variational GP wrapping GPyTorch.

    The model is created lazily on first :meth:`update` call so that
    constructing many :class:`LocalGP` instances (one per robot) is
    cheap when the GP is unused.
    """

    def __init__(self, config: Optional[LocalGPConfig] = None) -> None:
        self.cfg = config or LocalGPConfig()
        self._device = torch.device(self.cfg.device)
        self._model = None
        self._likelihood = None
        self._x_train: Optional[torch.Tensor] = None
        self._y_train: Optional[torch.Tensor] = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def _ensure_built(self) -> None:
        if self._model is None:
            self._model, self._likelihood = _build_svgp_modules(self.cfg)

    def reset(self) -> None:
        """Drop training data and re-init the model on next update."""
        self._model = None
        self._likelihood = None
        self._x_train = None
        self._y_train = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Training / inference
    # ------------------------------------------------------------------
    def update(self, x_train: np.ndarray, y_train: np.ndarray) -> float:
        """Append observations and retrain via variational ELBO.

        Parameters
        ----------
        x_train : (n, 2) array
            Positions visited.
        y_train : (n,) array
            Observed hazard values at those positions.

        Returns
        -------
        float
            Final ELBO loss after the limited training pass.
        """
        import gpytorch

        x = torch.as_tensor(np.asarray(x_train, dtype=np.float32),
                            dtype=torch.float32, device=self._device).reshape(-1, 2)
        y = torch.as_tensor(np.asarray(y_train, dtype=np.float32),
                            dtype=torch.float32, device=self._device).reshape(-1)
        if x.shape[0] != y.shape[0]:
            raise ValueError("x_train and y_train length mismatch")
        if x.shape[0] == 0:
            return float("nan")
        if self._x_train is None:
            self._x_train, self._y_train = x, y
        else:
            self._x_train = torch.cat([self._x_train, x], dim=0)
            self._y_train = torch.cat([self._y_train, y], dim=0)
        if self._x_train.shape[0] > self.cfg.max_train_points:
            keep = self._x_train.shape[0] - self.cfg.max_train_points
            self._x_train = self._x_train[keep:]
            self._y_train = self._y_train[keep:]
        self._ensure_built()
        self._model.train()
        self._likelihood.train()
        optim = torch.optim.Adam(
            list(self._model.parameters()) + list(self._likelihood.parameters()),
            lr=self.cfg.learning_rate,
        )
        mll = gpytorch.mlls.VariationalELBO(
            self._likelihood, self._model, num_data=self._x_train.shape[0],
        )
        n = self._x_train.shape[0]
        bs = min(self.cfg.batch_size, n)
        last = float("nan")
        for _ in range(self.cfg.n_epochs):
            idx = torch.randperm(n, device=self._device)[:bs]
            xb, yb = self._x_train[idx], self._y_train[idx]
            optim.zero_grad(set_to_none=True)
            output = self._model(xb)
            loss = -mll(output, yb)
            loss.backward()
            optim.step()
            last = float(loss.detach())
        self._fitted = True
        return last

    @torch.no_grad()
    def predict(self, x_query: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict ``(mean, variance)`` at ``x_query`` of shape (n, 2)."""
        import gpytorch

        x = torch.as_tensor(np.asarray(x_query, dtype=np.float32),
                            dtype=torch.float32, device=self._device).reshape(-1, 2)
        if not self._fitted:
            mean = np.zeros(x.shape[0], dtype=np.float32)
            var = np.full(x.shape[0], self.cfg.kernel_variance, dtype=np.float32)
            return mean, var
        self._model.eval()
        self._likelihood.eval()
        with gpytorch.settings.fast_pred_var():
            f_dist = self._model(x)
            y_dist = self._likelihood(f_dist)
        return (
            y_dist.mean.detach().cpu().numpy().astype(np.float32),
            y_dist.variance.detach().cpu().numpy().astype(np.float32),
        )

    def uncertainty_at(self, positions: np.ndarray) -> np.ndarray:
        """Return ``sigma`` (sqrt variance) at ``positions`` (n, 2)."""
        _, var = self.predict(positions)
        return np.sqrt(np.clip(var, 0.0, None))

    # ------------------------------------------------------------------
    # Posterior summary for BCM fusion
    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_posterior_params(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(inducing_pts, mean, variance)`` summarising the local GP.

        These are the quantities exchanged between robots for BCM
        fusion. Bandwidth scales with ``n_inducing``, not data size.
        """
        if not self._fitted:
            n = self.cfg.n_inducing
            return (
                np.zeros((n, 2), dtype=np.float32),
                np.zeros(n, dtype=np.float32),
                np.full(n, self.cfg.kernel_variance, dtype=np.float32),
            )
        var_strat = self._model.variational_strategy
        Z = var_strat.inducing_points.detach().cpu().numpy().astype(np.float32)
        mean, var = self.predict(Z)
        return Z, mean.astype(np.float32), var.astype(np.float32)

    @property
    def is_fitted(self) -> bool:
        return self._fitted
