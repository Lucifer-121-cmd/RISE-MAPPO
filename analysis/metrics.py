"""Evaluation metrics for multi-robot cooperative search.

All metrics operate on episode-level data collected during evaluation.
Each function takes an :class:`EpisodeData` container and returns a
scalar (or small dict) describing some quality of the trajectory.

The grid-to-world mapping mirrors the env: ``ix = clip(x / resolution)``
with ``grid_size = world_size / resolution``. The mapping is exposed
through :meth:`EpisodeData.world_size` so metrics are agnostic to the
scenario.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class EpisodeData:
    """Container for one evaluation episode's data.

    Per-timestep lists are appended once per env step. Episode-level
    scalars are set at episode end.
    """
    # Per-timestep data (lists of length T).
    robot_positions: List[np.ndarray] = field(default_factory=list)     # [(N, 3)]
    coverage_maps: List[np.ndarray] = field(default_factory=list)       # [(G, G)]
    gp_uncertainty: List[np.ndarray] = field(default_factory=list)      # [(G, G)]
    cvar_values: List[np.ndarray] = field(default_factory=list)         # [(N,)]
    lyapunov_values: List[np.ndarray] = field(default_factory=list)     # [(N,)]
    energy_consumed: List[np.ndarray] = field(default_factory=list)     # [(N,)] cumulative
    collisions: List[np.ndarray] = field(default_factory=list)          # [(N,)] flag
    detections: List[int] = field(default_factory=list)                 # new targets per step
    rewards: List[float] = field(default_factory=list)
    mpc_solve_times: List[np.ndarray] = field(default_factory=list)     # [(N,)]

    # Episode-level scalars.
    num_robots: int = 0
    num_targets: int = 0
    max_steps: int = 0
    total_targets_found: int = 0
    world_size: float = 10.0


def coverage_rate(data: EpisodeData) -> float:
    """Fraction of explorable grid covered by episode end."""
    if not data.coverage_maps:
        return 0.0
    return float(np.mean(data.coverage_maps[-1]))


def coverage_over_time(data: EpisodeData) -> np.ndarray:
    """Per-step coverage fraction (for plotting curves)."""
    if not data.coverage_maps:
        return np.zeros(0, dtype=np.float32)
    return np.array([float(np.mean(m)) for m in data.coverage_maps], dtype=np.float32)


def detection_success_rate(data: EpisodeData) -> float:
    """Fraction of targets found within episode budget."""
    return float(data.total_targets_found) / max(data.num_targets, 1)


def time_to_full_detection(data: EpisodeData) -> float:
    """First timestep at which all targets are detected (else max_steps)."""
    if not data.detections:
        return float(data.max_steps)
    cumulative = np.cumsum(data.detections)
    hits = np.where(cumulative >= data.num_targets)[0]
    if len(hits) > 0:
        return float(hits[0])
    return float(data.max_steps)


def collision_rate(data: EpisodeData) -> float:
    """Total collision events across episode (lower is better)."""
    if not data.collisions:
        return 0.0
    return float(sum(float(np.sum(c)) for c in data.collisions))


def energy_efficiency(data: EpisodeData) -> float:
    """Coverage per unit total energy consumed (higher is better)."""
    if not data.energy_consumed:
        return 0.0
    final_cov = coverage_rate(data)
    total_energy = float(np.sum(data.energy_consumed[-1]))
    if total_energy < 1e-6:
        return 0.0
    return final_cov / total_energy


def mean_cvar_risk(data: EpisodeData) -> float:
    """Mean per-robot per-step CVaR risk (lower is safer)."""
    if not data.cvar_values:
        return 0.0
    return float(np.mean(np.concatenate([c.ravel() for c in data.cvar_values])))


def exploration_overlap(data: EpisodeData) -> float:
    """Fraction of visited cells touched by >1 robot (lower is better coord)."""
    if not data.coverage_maps or not data.robot_positions:
        return 0.0
    grid_size = data.coverage_maps[0].shape[0]
    world_size = float(data.world_size)
    # Per-robot set of cells visited across the episode.
    visited = [set() for _ in range(data.num_robots)]
    for positions in data.robot_positions:
        for i in range(data.num_robots):
            gx = int(np.clip(positions[i, 0] / world_size * grid_size, 0, grid_size - 1))
            gy = int(np.clip(positions[i, 1] / world_size * grid_size, 0, grid_size - 1))
            visited[i].add((gx, gy))
    counts = np.zeros((grid_size, grid_size), dtype=np.int32)
    for cells in visited:
        for (gx, gy) in cells:
            counts[gx, gy] += 1
    any_visited = counts > 0
    overlapping = counts > 1
    n_visited = int(any_visited.sum())
    if n_visited == 0:
        return 0.0
    return float(overlapping.sum()) / float(n_visited)


def lyapunov_stability(data: EpisodeData) -> Dict[str, float]:
    """Lyapunov-function monotonic-decrease diagnostic.

    Returns mean across robots of:
        - monotonic_fraction : fraction of steps with V(t+1) <= V(t).
        - max_violation      : largest dV across the episode.
        - mean_decay_rate    : mean dV (negative = decreasing).
    """
    out = {"monotonic_fraction": [], "max_violation": [], "mean_decay_rate": []}
    if not data.lyapunov_values:
        return {k: 0.0 for k in out}
    for i in range(data.num_robots):
        v_series = np.array([lv[i] for lv in data.lyapunov_values], dtype=np.float64)
        if v_series.size < 2:
            out["monotonic_fraction"].append(1.0)
            out["max_violation"].append(0.0)
            out["mean_decay_rate"].append(0.0)
            continue
        dv = np.diff(v_series)
        out["monotonic_fraction"].append(float(np.mean(dv <= 1e-6)))
        out["max_violation"].append(float(np.max(dv)))
        out["mean_decay_rate"].append(float(np.mean(dv)))
    return {k: float(np.mean(v)) for k, v in out.items()}


def compute_all_metrics(data: EpisodeData) -> Dict[str, float]:
    """Flatten all metrics into one dict (one row per episode)."""
    lyap = lyapunov_stability(data)
    return {
        "coverage_rate": coverage_rate(data),
        "detection_success": detection_success_rate(data),
        "time_to_detection": time_to_full_detection(data),
        "collision_rate": collision_rate(data),
        "energy_efficiency": energy_efficiency(data),
        "mean_cvar_risk": mean_cvar_risk(data),
        "exploration_overlap": exploration_overlap(data),
        "lyapunov_monotonic": lyap["monotonic_fraction"],
        "lyapunov_max_violation": lyap["max_violation"],
        "lyapunov_mean_decay": lyap["mean_decay_rate"],
    }


METRIC_KEYS = (
    "coverage_rate",
    "detection_success",
    "time_to_detection",
    "collision_rate",
    "energy_efficiency",
    "mean_cvar_risk",
    "exploration_overlap",
    "lyapunov_monotonic",
    "lyapunov_max_violation",
    "lyapunov_mean_decay",
)
