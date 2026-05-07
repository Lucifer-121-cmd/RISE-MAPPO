"""Sanity tests for Phase-1 + Phase-2 distributed GP.

Phase 1 keeps the decay-grid surrogate; Phase 2 adds local sparse GPs
and BCM fusion. Both are exercised here.
"""
from __future__ import annotations

import numpy as np
import pytest

from gp.distributed_gp import DistributedGP, DistributedGPConfig, gaussian_cvar_coefficient
from gp.local_gp import LocalGP, LocalGPConfig


def test_initial_uncertainty_is_prior() -> None:
    gp = DistributedGP(DistributedGPConfig(prior_sigma=0.7))
    grid = gp.uncertainty_grid()
    assert np.allclose(grid, 0.7)


def test_observation_decreases_local_uncertainty() -> None:
    gp = DistributedGP(DistributedGPConfig())
    pre = gp.uncertainty_grid().sum()
    gp.update([(5.0, 5.0)])
    post = gp.uncertainty_grid().sum()
    assert post < pre


def test_information_gain_is_nonnegative_and_resets() -> None:
    gp = DistributedGP(DistributedGPConfig())
    gp.update([(5.0, 5.0)])
    g1 = gp.information_gain()
    g2 = gp.information_gain()
    assert g1 > 0.0
    assert g2 == 0.0  # second call sees no further reduction


def test_uncertainty_patch_shape() -> None:
    gp = DistributedGP(DistributedGPConfig())
    patch = gp.uncertainty_patch((2.0, 2.0), patch_size=16)
    assert patch.shape == (16, 16)


def test_cvar_risk_is_finite() -> None:
    gp = DistributedGP(DistributedGPConfig())
    risk = gp.cvar_risk((4.0, 4.0))
    assert np.isfinite(risk)
    assert risk >= 0.0


# ---------------------------------------------------------------------
# Phase-2 STEP 3: LocalGP
# ---------------------------------------------------------------------
def _hazard_fn(xy: np.ndarray) -> np.ndarray:
    """Reference scalar field for GP regression tests."""
    return np.exp(-((xy[:, 0] - 5.0) ** 2 + (xy[:, 1] - 5.0) ** 2) / 4.0)


def test_local_gp_variance_low_near_training_high_far() -> None:
    pytest.importorskip("gpytorch")
    rng = np.random.default_rng(0)
    xs = rng.uniform(4.0, 6.0, size=(40, 2)).astype(np.float32)
    ys = _hazard_fn(xs).astype(np.float32)
    gp = LocalGP(LocalGPConfig(n_inducing=25, n_epochs=40))
    gp.update(xs, ys)
    near = np.array([[5.0, 5.0]], dtype=np.float32)
    far = np.array([[0.5, 0.5]], dtype=np.float32)
    _, v_near = gp.predict(near)
    _, v_far = gp.predict(far)
    assert float(v_far[0]) > float(v_near[0])


def test_local_gp_more_data_decreases_variance() -> None:
    pytest.importorskip("gpytorch")
    rng = np.random.default_rng(1)
    xs1 = rng.uniform(0.0, 10.0, size=(20, 2)).astype(np.float32)
    gp = LocalGP(LocalGPConfig(n_inducing=20, n_epochs=20))
    gp.update(xs1, _hazard_fn(xs1).astype(np.float32))
    grid = np.array([[3.0, 3.0], [7.0, 7.0]], dtype=np.float32)
    _, v_before = gp.predict(grid)
    xs2 = rng.uniform(0.0, 10.0, size=(60, 2)).astype(np.float32)
    gp.update(xs2, _hazard_fn(xs2).astype(np.float32))
    _, v_after = gp.predict(grid)
    assert v_after.mean() <= v_before.mean()


def test_local_gp_posterior_params_shapes() -> None:
    pytest.importorskip("gpytorch")
    gp = LocalGP(LocalGPConfig(n_inducing=16, n_epochs=10))
    xs = np.random.default_rng(2).uniform(0, 10, size=(30, 2)).astype(np.float32)
    gp.update(xs, _hazard_fn(xs).astype(np.float32))
    Z, m, v = gp.get_posterior_params()
    assert Z.shape == (16, 2)
    assert m.shape == (16,) and v.shape == (16,)
    assert np.all(v >= 0.0)


# ---------------------------------------------------------------------
# Phase-2 STEP 4: DistributedGP with BCM fusion
# ---------------------------------------------------------------------
def test_bcm_fusion_lowers_variance() -> None:
    pytest.importorskip("gpytorch")
    cfg = DistributedGPConfig(
        local_gp=LocalGPConfig(n_inducing=20, n_epochs=15),
    )
    dgp = DistributedGP(cfg, num_robots=3)
    rng = np.random.default_rng(3)
    # Each robot observes one quadrant.
    quadrants = [(0.0, 5.0, 0.0, 5.0), (5.0, 10.0, 0.0, 5.0), (2.5, 7.5, 5.0, 10.0)]
    for i, (x0, x1, y0, y1) in enumerate(quadrants):
        xs = np.column_stack([
            rng.uniform(x0, x1, 30),
            rng.uniform(y0, y1, 30),
        ]).astype(np.float32)
        dgp.update_robot(i, xs, _hazard_fn(xs).astype(np.float32))
    dgp.fuse()
    grid = np.array([[3.0, 3.0], [7.0, 3.0], [5.0, 7.0]], dtype=np.float32)
    _, v_global = dgp.predict_global(grid)
    _, v_solo = dgp.local_gps[0].predict(grid)
    # Fused variance must be ≤ a single robot's variance at every query.
    assert np.all(v_global <= v_solo + 1e-5)


def test_information_gain_at_decreases_with_data() -> None:
    pytest.importorskip("gpytorch")
    cfg = DistributedGPConfig(local_gp=LocalGPConfig(n_inducing=15, n_epochs=15))
    dgp = DistributedGP(cfg, num_robots=2)
    grid = np.array([[5.0, 5.0]], dtype=np.float32)
    rng = np.random.default_rng(4)
    xs = rng.uniform(4.0, 6.0, size=(40, 2)).astype(np.float32)
    dgp.update_robot(0, xs, _hazard_fn(xs).astype(np.float32))
    ig_visited = float(dgp.information_gain_at(grid)[0])
    far_grid = np.array([[0.2, 0.2]], dtype=np.float32)
    ig_unvisited = float(dgp.information_gain_at(far_grid)[0])
    # Mutual info should be larger near training data than far away.
    assert ig_visited > ig_unvisited - 1e-3


def test_cvar_risk_at_grows_in_hazard() -> None:
    pytest.importorskip("gpytorch")
    cfg = DistributedGPConfig(local_gp=LocalGPConfig(n_inducing=15, n_epochs=15))
    dgp = DistributedGP(cfg, num_robots=2)
    rng = np.random.default_rng(5)
    xs = rng.uniform(0.0, 10.0, size=(80, 2)).astype(np.float32)
    dgp.update_robot(0, xs, _hazard_fn(xs).astype(np.float32))
    safe = np.array([[0.5, 0.5]], dtype=np.float32)
    haz = np.array([[5.0, 5.0]], dtype=np.float32)
    cvar_safe = float(dgp.cvar_risk_at(safe, alpha=0.95)[0])
    cvar_haz = float(dgp.cvar_risk_at(haz, alpha=0.95)[0])
    assert cvar_haz > cvar_safe


def test_uncertainty_grid_shape_after_fuse() -> None:
    pytest.importorskip("gpytorch")
    cfg = DistributedGPConfig(world_size=10.0, resolution=0.5,
                              local_gp=LocalGPConfig(n_inducing=10, n_epochs=10))
    dgp = DistributedGP(cfg, num_robots=2)
    rng = np.random.default_rng(6)
    xs = rng.uniform(0.0, 10.0, size=(20, 2)).astype(np.float32)
    dgp.update_robot(0, xs, _hazard_fn(xs).astype(np.float32))
    dgp.fuse()
    g = dgp.uncertainty_grid()
    expected = int(round(cfg.world_size / cfg.resolution))
    assert g.shape == (expected, expected)


def test_gaussian_cvar_coefficient_matches_known_value() -> None:
    # c(0.95) ≈ 2.0627 from norm tables.
    val = gaussian_cvar_coefficient(0.95)
    assert val == pytest.approx(2.0627, abs=1e-3)
    assert gaussian_cvar_coefficient(0.0) == 0.0
    with pytest.raises(ValueError):
        gaussian_cvar_coefficient(1.0)
