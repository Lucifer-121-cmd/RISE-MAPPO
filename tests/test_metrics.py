"""Unit tests for analysis.metrics."""
from __future__ import annotations

import numpy as np
import pytest

from analysis.metrics import (
    EpisodeData,
    METRIC_KEYS,
    collision_rate,
    compute_all_metrics,
    coverage_over_time,
    coverage_rate,
    detection_success_rate,
    energy_efficiency,
    exploration_overlap,
    lyapunov_stability,
    mean_cvar_risk,
    time_to_full_detection,
)


def _make_episode(
    T: int = 5,
    N: int = 3,
    G: int = 10,
    targets: int = 5,
    world_size: float = 10.0,
) -> EpisodeData:
    rng = np.random.default_rng(0)
    ep = EpisodeData(num_robots=N, num_targets=targets, max_steps=T, world_size=world_size)
    for t in range(T):
        ep.robot_positions.append(rng.uniform(0.0, world_size, size=(N, 3)).astype(np.float32))
        ep.coverage_maps.append(np.zeros((G, G), dtype=bool))
        ep.gp_uncertainty.append(np.ones((G, G), dtype=np.float32))
        ep.cvar_values.append(np.zeros(N, dtype=np.float32))
        ep.lyapunov_values.append(np.full(N, 1.0 - t * 0.1, dtype=np.float32))
        ep.energy_consumed.append(np.full(N, float(t + 1), dtype=np.float32))
        ep.collisions.append(np.zeros(N, dtype=np.float32))
        ep.detections.append(0)
        ep.rewards.append(0.0)
        ep.mpc_solve_times.append(np.zeros(N, dtype=np.float32))
    return ep


def test_coverage_rate_full_and_half():
    ep = _make_episode()
    ep.coverage_maps[-1][:] = True
    assert coverage_rate(ep) == pytest.approx(1.0)
    half = np.zeros_like(ep.coverage_maps[-1])
    half[: half.shape[0] // 2, :] = True
    ep.coverage_maps[-1] = half
    assert coverage_rate(ep) == pytest.approx(0.5)


def test_coverage_over_time_shape():
    ep = _make_episode(T=7)
    arr = coverage_over_time(ep)
    assert arr.shape == (7,)


def test_detection_success_fraction():
    ep = _make_episode(targets=5)
    ep.total_targets_found = 3
    assert detection_success_rate(ep) == pytest.approx(0.6)


def test_time_to_full_detection_found_and_not_found():
    ep = _make_episode(T=5, targets=2)
    ep.detections = [0, 1, 0, 1, 0]      # cumulative reaches 2 at index 3
    assert time_to_full_detection(ep) == 3.0
    ep.detections = [0, 0, 0, 0, 0]
    assert time_to_full_detection(ep) == float(ep.max_steps)


def test_collision_rate_zero():
    ep = _make_episode()
    assert collision_rate(ep) == 0.0
    ep.collisions[2][1] = 1.0
    ep.collisions[3][0] = 1.0
    assert collision_rate(ep) == 2.0


def test_energy_efficiency_positive():
    ep = _make_episode()
    ep.coverage_maps[-1][:] = True
    val = energy_efficiency(ep)
    assert val > 0.0


def test_mean_cvar_risk_zero():
    ep = _make_episode()
    assert mean_cvar_risk(ep) == pytest.approx(0.0)


def test_lyapunov_monotonic_decreasing():
    ep = _make_episode(T=5)
    # Already decreasing in _make_episode → all monotonic.
    res = lyapunov_stability(ep)
    assert res["monotonic_fraction"] == pytest.approx(1.0)
    assert res["mean_decay_rate"] < 0.0


def test_lyapunov_violation_detected():
    ep = _make_episode(T=5)
    for t in range(5):
        ep.lyapunov_values[t][:] = float(t)  # monotonic INCREASING
    res = lyapunov_stability(ep)
    assert res["monotonic_fraction"] < 0.5
    assert res["max_violation"] > 0.0


def test_exploration_overlap_full_and_none():
    # All robots at the same cell → all visited cells overlap.
    ep = _make_episode(T=3, N=2, G=10, world_size=10.0)
    for t in range(3):
        ep.robot_positions[t][:] = np.array([5.0, 5.0, 0.0], dtype=np.float32)
    assert exploration_overlap(ep) == pytest.approx(1.0)
    # Robots at well-separated cells → no overlap.
    ep2 = _make_episode(T=3, N=2, G=10, world_size=10.0)
    for t in range(3):
        ep2.robot_positions[t][0] = np.array([1.0, 1.0, 0.0], dtype=np.float32)
        ep2.robot_positions[t][1] = np.array([9.0, 9.0, 0.0], dtype=np.float32)
    assert exploration_overlap(ep2) == pytest.approx(0.0)


def test_compute_all_metrics_keys_present():
    ep = _make_episode()
    out = compute_all_metrics(ep)
    for k in METRIC_KEYS:
        assert k in out
        assert isinstance(out[k], float)
