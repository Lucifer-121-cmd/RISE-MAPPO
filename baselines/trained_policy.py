"""Trained RISE-MAPPO actor wrapper conforming to :class:`BasePolicy`.

Loads only the actor sub-module from a checkpoint — the critic is not
needed for evaluation. Forward passes are wrapped in ``torch.no_grad``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import torch
import yaml

from baselines.base_policy import BasePolicy
from marl.mappo.actor import Actor, ActorConfig


_LOG = logging.getLogger("paper3.eval.trained")
_PATCH_SIZE = 48


def _load_yaml(path: Union[str, Path]) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


class TrainedPolicy(BasePolicy):
    """Load a saved actor and run deterministic / stochastic rollouts."""

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        config_path: Union[str, Path],
        device: str = "cpu",
        deterministic: bool = True,
        name: str = "",
    ) -> None:
        ckpt_path = Path(checkpoint_path)
        cfg_path = Path(config_path)
        raw = _load_yaml(cfg_path)
        env_raw = raw.get("env", {})
        num_robots = int(env_raw.get("num_robots", 3))
        # K = subgoal_grid ** 2; subgoal_grid hard-coded to 5 in env.
        num_actions = 25
        self._device = torch.device(device)
        self._deterministic = bool(deterministic)
        self._num_robots = num_robots
        self._num_actions = num_actions
        self._actor = Actor(ActorConfig(
            patch_size=_PATCH_SIZE,
            num_actions=num_actions,
            num_robots=num_robots,
        ))
        state = torch.load(str(ckpt_path), map_location=self._device, weights_only=False)
        if "actor" not in state:
            raise KeyError(
                f"Checkpoint {ckpt_path} missing 'actor' key (found {list(state.keys())})"
            )
        self._actor.load_state_dict(state["actor"])
        self._actor.to(self._device).eval()
        self._update = int(state.get("update", -1))
        if name:
            self._name = name
        else:
            self._name = f"RISE-MAPPO (upd{self._update})" if self._update >= 0 else "RISE-MAPPO"
        _LOG.info(
            "loaded trained policy from %s (update=%d, name=%s)",
            ckpt_path, self._update, self._name,
        )

    def reset(self, num_robots: int) -> None:
        if num_robots != self._num_robots:
            _LOG.warning(
                "trained policy was built for %d robots but env has %d; "
                "the actor will run agent-by-agent and may degrade.",
                self._num_robots, num_robots,
            )
        self._active_robots = int(num_robots)

    @torch.no_grad()
    def get_actions(self, observations: Dict[str, Dict[str, np.ndarray]], global_state: Any) -> np.ndarray:
        agents = list(observations.keys())
        batch: Dict[str, torch.Tensor] = {}
        for key in observations[agents[0]]:
            stacked = np.stack([observations[a][key] for a in agents], axis=0)
            batch[key] = torch.from_numpy(stacked).float().to(self._device)
        action, _, _ = self._actor.get_action(batch, deterministic=self._deterministic)
        return action.detach().cpu().numpy().astype(np.int64)

    @property
    def name(self) -> str:
        return self._name


class AblationPolicy(TrainedPolicy):
    """Trained-policy wrapper with an explicit override name (for ablations)."""

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        config_path: Union[str, Path],
        name: str,
        **kwargs,
    ) -> None:
        super().__init__(checkpoint_path=checkpoint_path, config_path=config_path, name=name, **kwargs)
