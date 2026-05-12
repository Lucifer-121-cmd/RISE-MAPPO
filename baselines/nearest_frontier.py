"""Nearest-frontier exploration (Yamauchi 1997).

Each robot picks the closest cell on the visibility frontier — a cell
that is *unvisited* yet adjacent to a *visited* cell — and converts
that target into the closest subgoal in the env's ego-centric 5×5 grid.
"""
from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from baselines.base_policy import BasePolicy


def _frontier_cells(coverage: np.ndarray) -> np.ndarray:
    """Return ``(M, 2)`` array of (ix, iy) frontier cells (unvisited & adj. to visited)."""
    visited = coverage.astype(bool)
    unvisited = ~visited
    # Shift visited mask in 4-connected directions; any True neighbour ⇒ frontier.
    adj = np.zeros_like(visited)
    adj[1:, :] |= visited[:-1, :]
    adj[:-1, :] |= visited[1:, :]
    adj[:, 1:] |= visited[:, :-1]
    adj[:, :-1] |= visited[:, 1:]
    frontier = unvisited & adj
    return np.argwhere(frontier)  # rows of (iy, ix) given numpy row/col convention


class NearestFrontierPolicy(BasePolicy):
    """Greedy per-robot nearest-frontier exploration."""

    def __init__(
        self,
        num_actions: int,
        subgoal_grid: int = 5,
        subgoal_spacing: float = 1.0,
    ) -> None:
        self._num_actions = int(num_actions)
        self._K = int(subgoal_grid)
        self._spacing = float(subgoal_spacing)
        self._half = self._K // 2
        self._num_robots = 0

    def reset(self, num_robots: int) -> None:
        self._num_robots = int(num_robots)

    @property
    def name(self) -> str:
        return "Nearest Frontier"

    def _target_to_action(
        self,
        robot_xy: np.ndarray,
        target_xy: np.ndarray,
    ) -> int:
        rx = (target_xy[0] - robot_xy[0]) / self._spacing
        ry = (target_xy[1] - robot_xy[1]) / self._spacing
        col = int(np.clip(round(rx), -self._half, self._half) + self._half)
        row = int(np.clip(round(ry), -self._half, self._half) + self._half)
        return int(row * self._K + col)

    def get_actions(self, observations, global_state) -> np.ndarray:
        coverage = global_state["coverage_map"]
        robot_states = global_state["robot_states"]  # (N, 3)
        world_size = float(global_state["world_size"])
        G = coverage.shape[0]
        res = world_size / G
        frontier = _frontier_cells(coverage)
        actions = np.full(self._num_robots, self._K * self._K // 2, dtype=np.int64)
        if frontier.size == 0:
            return actions
        # Cell-centre world coordinates of each frontier cell.
        front_xy = (frontier + 0.5) * res
        front_xy = front_xy[:, ::-1]  # numpy (iy, ix) → (x, y)
        for i in range(self._num_robots):
            robot_xy = robot_states[i, :2]
            d2 = np.sum((front_xy - robot_xy[None, :]) ** 2, axis=1)
            nearest = front_xy[int(np.argmin(d2))]
            actions[i] = self._target_to_action(robot_xy, nearest)
        return actions
