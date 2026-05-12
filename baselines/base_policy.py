"""Abstract base class for all evaluation policies (trained + baselines).

A :class:`BasePolicy` consumes either raw env observations (baselines)
or torch-friendly tensors (trained networks). The evaluation loop
always calls :meth:`get_actions` once per env step.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BasePolicy(ABC):
    """Uniform policy interface for evaluation."""

    @abstractmethod
    def reset(self, num_robots: int) -> None:
        """Reset per-episode state. Called at episode start."""

    @abstractmethod
    def get_actions(
        self,
        observations: Any,
        global_state: Any,
    ) -> np.ndarray:
        """Return one discrete action per robot.

        Parameters
        ----------
        observations : Any
            Per-agent observation dict ``{agent_id: {key: ndarray}}`` as
            returned by :class:`MultiRobotSearchEnv`.
        global_state : Any
            Centralised state dict (used by trained policies, ignored
            by baselines).

        Returns
        -------
        actions : np.ndarray, shape ``(N,)``, dtype int64
            Subgoal index per robot in ``[0, num_actions)``.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier (used for logging + plotting)."""
