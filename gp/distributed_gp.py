"""Distributed GP with Bayesian Committee Machine fusion (Phase 2).

Each robot owns a :class:`gp.local_gp.LocalGP`; this class fuses their
posteriors at any query point with the BCM rule

    1/σ²_global = Σ_i 1/σ²_i − (n − 1)/σ²_prior
    μ_global    = σ²_global · Σ_i μ_i / σ²_i

A Phase-1-compatible decay grid is also kept so the existing
:class:`envs.multi_robot_search_env.MultiRobotSearchEnv` (and its tests)
keep working when no real GP observations are streamed yet. The class
therefore exposes both:

* legacy methods used by the env / Phase-1 tests:
    ``reset()``, ``update(robot_positions)``, ``uncertainty_grid()``,
    ``uncertainty_patch(...)``, ``information_gain()``,
    ``cvar_risk(robot_pos)``.
* Phase-2 methods used by the upgraded env wiring:
    ``update_robot(robot_id, positions, values)``, ``fuse()``,
    ``predict_global(x_query)``, ``information_gain_at(positions)``,
    ``cvar_risk_at(positions, alpha)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import norm

from gp.local_gp import LocalGP, LocalGPConfig


_EPS = 1e-9


@dataclass
class DistributedGPConfig:
    """Configuration for :class:`DistributedGP`."""

    world_size: float = 10.0
    resolution: float = 0.1
    prior_sigma: float = 1.0
    sigma_floor: float = 0.05
    update_radius: float = 1.0
    decay: float = 0.4
    cvar_alpha: float = 0.05
    local_gp: LocalGPConfig = field(default_factory=LocalGPConfig)


def gaussian_cvar_coefficient(alpha: float) -> float:
    """Return ``c(α) = φ(Φ⁻¹(α)) / (1 − α)`` for Gaussian CVaR.

    Convention: for ``L ~ N(μ, σ²)`` and α ∈ (0, 1),
    ``CVaR_α(L) = μ + σ · c(α)``. Higher α ⇒ larger ``c(α)`` ⇒ more
    conservative tail. (See author's GP-CVaR-MPC paper.)
    """
    a = float(alpha)
    if a <= 0.0:
        return 0.0
    if a >= 1.0:
        raise ValueError("alpha must be < 1.0")
    z = norm.ppf(a)
    return float(norm.pdf(z) / (1.0 - a))


class DistributedGP:
    """Phase-2 distributed GP with BCM fusion + Phase-1 decay fallback."""

    def __init__(
        self,
        cfg: Optional[DistributedGPConfig] = None,
        num_robots: int = 0,
    ) -> None:
        self.cfg = cfg or DistributedGPConfig()
        self.grid_size = int(round(self.cfg.world_size / self.cfg.resolution))
        # Phase-1 decay grid (legacy / fallback when no GP data).
        self.sigma = np.full(
            (self.grid_size, self.grid_size),
            self.cfg.prior_sigma,
            dtype=np.float32,
        )
        self._last_total = float(self.sigma.sum())
        # Phase-2 local GPs.
        self.num_robots = int(num_robots)
        self.local_gps: List[LocalGP] = [
            LocalGP(self.cfg.local_gp) for _ in range(self.num_robots)
        ]
        self._fused_cache_grid: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset both Phase-1 grid and Phase-2 local GPs."""
        self.sigma.fill(self.cfg.prior_sigma)
        self._last_total = float(self.sigma.sum())
        for gp in self.local_gps:
            gp.reset()
        self._fused_cache_grid = None

    # ------------------------------------------------------------------
    # Phase-1 legacy interface (kept for env + existing tests)
    # ------------------------------------------------------------------
    def update(self, robot_positions: Sequence[Sequence[float]]) -> None:
        """Decay the legacy sigma grid near robot positions."""
        g = self.grid_size
        res = self.cfg.resolution
        for (x, y) in robot_positions:
            ix = int(np.clip(x / res, 0, g - 1))
            iy = int(np.clip(y / res, 0, g - 1))
            r_cells = int(np.ceil(self.cfg.update_radius / res))
            x_lo, x_hi = max(0, ix - r_cells), min(g, ix + r_cells + 1)
            y_lo, y_hi = max(0, iy - r_cells), min(g, iy + r_cells + 1)
            for i in range(x_lo, x_hi):
                for j in range(y_lo, y_hi):
                    cx = (i + 0.5) * res
                    cy = (j + 0.5) * res
                    d2 = (cx - x) ** 2 + (cy - y) ** 2
                    if d2 > self.cfg.update_radius ** 2:
                        continue
                    decay = self.cfg.decay * np.exp(-d2 / (0.5 * self.cfg.update_radius ** 2))
                    new = self.sigma[j, i] * (1.0 - decay)
                    self.sigma[j, i] = max(self.cfg.sigma_floor, new)
        self._fused_cache_grid = None

    def uncertainty_grid(
        self,
        world_size: Optional[float] = None,
        resolution: Optional[float] = None,
    ) -> np.ndarray:
        """Return the active sigma grid.

        If at least one local GP is fitted, returns the BCM-fused sigma
        rasterised onto the grid (Phase 2). Otherwise returns the
        Phase-1 decay grid. ``world_size`` / ``resolution`` are
        accepted for API compatibility but the configured grid is
        always used.
        """
        if any(gp.is_fitted for gp in self.local_gps):
            if self._fused_cache_grid is None:
                self._fused_cache_grid = self._compute_fused_sigma_grid()
            return self._fused_cache_grid
        return self.sigma

    def uncertainty_patch(
        self,
        robot_pos: Sequence[float],
        patch_size: int,
        cell_size: Optional[float] = None,
    ) -> np.ndarray:
        """Sample an ego-centric sigma patch from the active grid."""
        cell = cell_size if cell_size is not None else self.cfg.resolution
        g = self.grid_size
        sigma = self.uncertainty_grid()
        x, y = float(robot_pos[0]), float(robot_pos[1])
        half = patch_size // 2
        offsets = (np.arange(patch_size) - half) * cell
        gx = x + offsets[None, :]
        gy = y + offsets[:, None]
        ix = np.clip((gx / self.cfg.resolution).astype(int), 0, g - 1)
        iy = np.clip((gy / self.cfg.resolution).astype(int), 0, g - 1)
        valid = (gx >= 0) & (gx <= self.cfg.world_size) & (gy >= 0) & (gy <= self.cfg.world_size)
        return np.where(valid, sigma[iy, ix], self.cfg.prior_sigma).astype(np.float32)

    def information_gain(self) -> float:
        """Reduction in summed sigma since the last call."""
        total = float(self.uncertainty_grid().sum())
        gain = max(0.0, self._last_total - total)
        self._last_total = total
        return gain

    def cvar_risk(self, robot_pos: Sequence[float]) -> float:
        """Phase-1 CVaR surrogate at one position.

        When local GPs are fitted, returns ``μ + σ · c(α)`` from the
        fused posterior; otherwise the local-mean of the decay grid.
        """
        if any(gp.is_fitted for gp in self.local_gps):
            risk = self.cvar_risk_at(np.asarray([robot_pos], dtype=float),
                                     alpha=self.cfg.cvar_alpha)
            return float(risk[0])
        patch = self.uncertainty_patch(robot_pos, patch_size=5)
        return float(patch.mean())

    # ------------------------------------------------------------------
    # Phase-2 BCM fusion interface
    # ------------------------------------------------------------------
    def update_robot(
        self,
        robot_id: int,
        positions: np.ndarray,
        values: np.ndarray,
    ) -> None:
        """Push observations into a single robot's local GP."""
        if not 0 <= robot_id < self.num_robots:
            raise IndexError(f"robot_id {robot_id} out of range [0, {self.num_robots})")
        self.local_gps[robot_id].update(positions, values)
        self._fused_cache_grid = None

    def fuse(self, others: Optional[Sequence["DistributedGP"]] = None) -> None:
        """Recompute the fused-grid cache.

        ``others`` is accepted for API symmetry with the Phase-1
        signature but ignored: the BCM fuse over per-robot
        :class:`LocalGP` posteriors happens internally.
        """
        self._fused_cache_grid = self._compute_fused_sigma_grid()

    def predict_global(
        self,
        x_query: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """BCM-fused ``(mean, variance)`` at ``x_query`` of shape (n, 2)."""
        x = np.asarray(x_query, dtype=float).reshape(-1, 2)
        prior_var = self.cfg.prior_sigma ** 2
        contributing: List[Tuple[np.ndarray, np.ndarray]] = []
        for gp in self.local_gps:
            if not gp.is_fitted:
                continue
            m, v = gp.predict(x.astype(np.float32))
            v = np.clip(v, _EPS, None)
            contributing.append((m.astype(np.float64), v.astype(np.float64)))
        if not contributing:
            return (
                np.zeros(x.shape[0], dtype=np.float32),
                np.full(x.shape[0], prior_var, dtype=np.float32),
            )
        n_active = len(contributing)
        precision = np.zeros(x.shape[0], dtype=np.float64)
        weighted_mean = np.zeros(x.shape[0], dtype=np.float64)
        for m, v in contributing:
            precision += 1.0 / v
            weighted_mean += m / v
        bcm_precision = precision - (n_active - 1) / prior_var
        bcm_precision = np.clip(bcm_precision, 1.0 / (prior_var * 10.0), None)
        var_global = 1.0 / bcm_precision
        mean_global = var_global * weighted_mean
        return mean_global.astype(np.float32), var_global.astype(np.float32)

    def information_gain_at(self, positions: np.ndarray) -> np.ndarray:
        """Mutual information per query: 0.5 · log(σ²_prior / σ²_post)."""
        prior_var = self.cfg.prior_sigma ** 2
        _, var = self.predict_global(positions)
        var = np.clip(var, _EPS, None)
        return (0.5 * np.log(prior_var / var)).astype(np.float32)

    def cvar_risk_at(
        self,
        positions: np.ndarray,
        alpha: float = 0.05,
    ) -> np.ndarray:
        """CVaR_α of hazard at each query: ``μ + σ · c(α)``."""
        c = gaussian_cvar_coefficient(float(alpha))
        mean, var = self.predict_global(positions)
        sigma = np.sqrt(np.clip(var, 0.0, None))
        return (mean + sigma * c).astype(np.float32)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _compute_fused_sigma_grid(self) -> np.ndarray:
        g = self.grid_size
        res = self.cfg.resolution
        xs = (np.arange(g) + 0.5) * res
        ys = (np.arange(g) + 0.5) * res
        xx, yy = np.meshgrid(xs, ys, indexing="xy")
        flat = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)
        _, var = self.predict_global(flat)
        sigma = np.sqrt(np.clip(var, 0.0, None)).reshape(g, g).astype(np.float32)
        return np.maximum(sigma, self.cfg.sigma_floor)

    # ------------------------------------------------------------------
    # Phase-1 placeholder hooks kept for backward compat
    # ------------------------------------------------------------------
    def predict(
        self,
        positions: Sequence[Sequence[float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Backward-compatible variant of :meth:`predict_global`."""
        return self.predict_global(np.asarray(positions, dtype=float).reshape(-1, 2))
