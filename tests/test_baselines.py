"""Baselines correctness tests (lightweight; runs on CPU)."""
from __future__ import annotations

import numpy as np
import pytest

from baselines import (
    BASELINE_REGISTRY,
    BasePolicy,
    NearestFrontierPolicy,
    RandomPolicy,
    VoronoiPartitionPolicy,
)


K = 25
SUBGOAL_GRID = 5


def _make_state(
    N: int = 3,
    G: int = 20,
    world_size: float = 10.0,
    visited_left_half: bool = True,
) -> dict:
    coverage = np.zeros((G, G), dtype=bool)
    if visited_left_half:
        coverage[:, : G // 2] = True
    # Robots on the LEFT (in visited side) so the frontier is to the right.
    robot_states = np.zeros((N, 3), dtype=np.float32)
    robot_states[:, 0] = np.linspace(1.0, 2.0, N)
    robot_states[:, 1] = np.linspace(2.0, 4.0, N)
    return {
        "coverage_map": coverage,
        "robot_states": robot_states,
        "world_size": float(world_size),
    }


def test_random_policy_actions_in_range():
    pol = RandomPolicy(num_actions=K, seed=0)
    pol.reset(num_robots=3)
    a = pol.get_actions(observations=None, global_state=_make_state())
    assert a.shape == (3,)
    assert a.dtype == np.int64
    assert np.all(a >= 0) and np.all(a < K)


def test_nearest_frontier_picks_right_side():
    pol = NearestFrontierPolicy(num_actions=K, subgoal_grid=SUBGOAL_GRID, subgoal_spacing=1.0)
    pol.reset(num_robots=3)
    state = _make_state(visited_left_half=True)
    a = pol.get_actions(observations=None, global_state=state)
    # All actions should have col >= center (move RIGHT, toward unvisited).
    for ai in a:
        col = int(ai) % SUBGOAL_GRID
        assert col >= SUBGOAL_GRID // 2, f"action {ai} not moving toward right frontier"


def test_voronoi_partition_disjoint_regions():
    pol = VoronoiPartitionPolicy(num_actions=K, subgoal_grid=SUBGOAL_GRID, subgoal_spacing=1.0)
    pol.reset(num_robots=3)
    state = _make_state(N=3, visited_left_half=False)  # nothing visited yet
    a = pol.get_actions(observations=None, global_state=state)
    assert a.shape == (3,)
    assert np.all(a >= 0) and np.all(a < K)


def test_all_baselines_in_registry_satisfy_interface():
    for key, cls in BASELINE_REGISTRY.items():
        pol = cls(num_actions=K) if key == "random" else cls(num_actions=K)
        assert isinstance(pol, BasePolicy)
        assert isinstance(pol.name, str)
        pol.reset(num_robots=2)
        a = pol.get_actions(
            observations=None,
            global_state=_make_state(N=2, visited_left_half=True),
        )
        assert a.shape == (2,)
        assert a.dtype == np.int64


def test_nearest_frontier_no_frontier_returns_center():
    pol = NearestFrontierPolicy(num_actions=K, subgoal_grid=SUBGOAL_GRID, subgoal_spacing=1.0)
    pol.reset(num_robots=2)
    state = _make_state(N=2)
    state["coverage_map"][:] = True  # fully covered → no frontier
    a = pol.get_actions(observations=None, global_state=state)
    center = SUBGOAL_GRID * SUBGOAL_GRID // 2
    assert np.all(a == center)
