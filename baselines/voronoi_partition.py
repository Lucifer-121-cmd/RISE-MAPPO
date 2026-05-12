"""Voronoi-partitioned exploration.

Each grid cell is assigned to its closest robot. Each robot then heads
toward the nearest *unvisited* cell inside its own region. If a robot
has no unvisited cells in its region, it falls back to the global
nearest unvisited cell.
"""
from __future__ import annotations

import numpy as np

from baselines.base_policy import BasePolicy
from baselines.nearest_frontier import NearestFrontierPolicy


class VoronoiPartitionPolicy(BasePolicy):
    """Partition the grid by nearest robot; greedy local exploration."""

    def __init__(
        self,
        num_actions: int,
        subgoal_grid: int = 5,
        subgoal_spacing: float = 1.0,
    ) -> None:
        self._num_actions = int(num_actions)
        # Reuse the frontier policy's subgoal-conversion helper.
        self._helper = NearestFrontierPolicy(
            num_actions=num_actions,
            subgoal_grid=subgoal_grid,
            subgoal_spacing=subgoal_spacing,
        )
        self._K = self._helper._K
        self._num_robots = 0

    def reset(self, num_robots: int) -> None:
        self._num_robots = int(num_robots)
        self._helper.reset(num_robots)

    @property
    def name(self) -> str:
        return "Voronoi Partition"

    def get_actions(self, observations, global_state) -> np.ndarray:
        coverage = global_state["coverage_map"]
        robot_states = global_state["robot_states"]
        world_size = float(global_state["world_size"])
        G = coverage.shape[0]
        res = world_size / G
        unvisited = ~coverage.astype(bool)
        actions = np.full(self._num_robots, self._K * self._K // 2, dtype=np.int64)
        unv_idx = np.argwhere(unvisited)
        if unv_idx.size == 0:
            return actions
        # World coords of every unvisited cell centre.
        unv_xy = (unv_idx + 0.5) * res
        unv_xy = unv_xy[:, ::-1]
        robot_xy = robot_states[:, :2]
        # Assign each unvisited cell to its closest robot (Voronoi).
        d2 = np.sum(
            (unv_xy[:, None, :] - robot_xy[None, :, :]) ** 2,
            axis=2,
        )
        owner = np.argmin(d2, axis=1)
        for i in range(self._num_robots):
            mine = unv_xy[owner == i]
            if mine.size == 0:
                # Empty region (rare) — fall back to global nearest unvisited.
                d2_i = np.sum((unv_xy - robot_xy[i][None, :]) ** 2, axis=1)
                target = unv_xy[int(np.argmin(d2_i))]
            else:
                d2_i = np.sum((mine - robot_xy[i][None, :]) ** 2, axis=1)
                target = mine[int(np.argmin(d2_i))]
            actions[i] = self._helper._target_to_action(robot_xy[i], target)
        return actions
