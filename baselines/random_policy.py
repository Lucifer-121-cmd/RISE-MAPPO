"""Uniform random subgoal selection. Lower-bound baseline."""
from __future__ import annotations

import numpy as np

from baselines.base_policy import BasePolicy


class RandomPolicy(BasePolicy):
    """Sample subgoal indices uniformly over ``[0, num_actions)``."""

    def __init__(self, num_actions: int, seed: int = 0) -> None:
        self._num_actions = int(num_actions)
        self._rng = np.random.default_rng(seed)
        self._num_robots = 0

    def reset(self, num_robots: int) -> None:
        self._num_robots = int(num_robots)

    def get_actions(self, observations, global_state) -> np.ndarray:
        return self._rng.integers(0, self._num_actions, size=self._num_robots, dtype=np.int64)

    @property
    def name(self) -> str:
        return "Random"
