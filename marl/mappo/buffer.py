"""Rollout buffer for MAPPO with GAE."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Mapping, Tuple

import numpy as np
import torch


@dataclass
class BufferConfig:
    rollout_length: int
    num_envs: int
    num_agents: int
    gamma: float = 0.99
    gae_lambda: float = 0.95
    device: str = "cpu"


class RolloutBuffer:
    """Stores (T, E, A, ...) rollouts and computes GAE returns.

    Notes
    -----
    Each agent shares observations only at the local level; the global
    state is shared across all agents per env per step. ``rewards`` are
    team-shared (same for every agent) but stored per-agent for shape
    consistency with the per-agent value of the *centralised* critic
    (which we evaluate once per step and broadcast).
    """

    def __init__(
        self,
        cfg: BufferConfig,
        local_obs_shapes: Mapping[str, Tuple[int, ...]],
        global_state_shapes: Mapping[str, Tuple[int, ...]],
    ) -> None:
        self.cfg = cfg
        self._local_keys = list(local_obs_shapes.keys())
        self._global_keys = list(global_state_shapes.keys())
        T, E, A = cfg.rollout_length, cfg.num_envs, cfg.num_agents
        self.local_obs: Dict[str, np.ndarray] = {
            k: np.zeros((T, E, A) + tuple(shape), dtype=np.float32)
            for k, shape in local_obs_shapes.items()
        }
        self.global_state: Dict[str, np.ndarray] = {
            k: np.zeros((T, E) + tuple(shape), dtype=np.float32)
            for k, shape in global_state_shapes.items()
        }
        self.actions = np.zeros((T, E, A), dtype=np.int64)
        self.log_probs = np.zeros((T, E, A), dtype=np.float32)
        self.rewards = np.zeros((T, E), dtype=np.float32)
        self.dones = np.zeros((T, E), dtype=np.float32)
        self.values = np.zeros((T, E), dtype=np.float32)
        self.advantages = np.zeros((T, E), dtype=np.float32)
        self.returns = np.zeros((T, E), dtype=np.float32)

    # ------------------------------------------------------------------
    def insert(
        self,
        t: int,
        local_obs: Mapping[str, np.ndarray],
        global_state: Mapping[str, np.ndarray],
        actions: np.ndarray,
        log_probs: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
    ) -> None:
        for k in self._local_keys:
            self.local_obs[k][t] = local_obs[k]
        for k in self._global_keys:
            self.global_state[k][t] = global_state[k]
        self.actions[t] = actions
        self.log_probs[t] = log_probs
        self.rewards[t] = rewards
        self.dones[t] = dones
        self.values[t] = values

    # ------------------------------------------------------------------
    def compute_returns_and_advantages(
        self,
        last_values: np.ndarray,
        last_dones: np.ndarray,
    ) -> None:
        """GAE with γ-λ; bootstraps from ``last_values`` at horizon."""
        T = self.cfg.rollout_length
        gae = np.zeros(self.cfg.num_envs, dtype=np.float32)
        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 1.0 - last_dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_values = self.values[t + 1]
            delta = (
                self.rewards[t]
                + self.cfg.gamma * next_values * next_non_terminal
                - self.values[t]
            )
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae
        self.returns = self.advantages + self.values

    # ------------------------------------------------------------------
    def feed_forward_generator(
        self,
        num_mini_batch: int,
    ) -> Iterator[Dict[str, torch.Tensor]]:
        """Yield mini-batches over (T*E*A) flattened transitions.

        Image features and vectors are stacked across agents so the
        actor sees a flat batch of agent samples. The centralised
        critic still consumes the *per-env* global state, broadcast
        across agents.
        """
        T = self.cfg.rollout_length
        E = self.cfg.num_envs
        A = self.cfg.num_agents
        device = torch.device(self.cfg.device)
        # Flatten T*E*A for actor inputs.
        local_flat = {
            k: torch.from_numpy(arr.reshape((T * E * A,) + arr.shape[3:])).to(device)
            for k, arr in self.local_obs.items()
        }
        actions = torch.from_numpy(self.actions.reshape(T * E * A)).to(device)
        log_probs = torch.from_numpy(self.log_probs.reshape(T * E * A)).to(device)
        # Per-step global state broadcast across agents → (T*E*A, ...).
        global_flat: Dict[str, torch.Tensor] = {}
        for k, arr in self.global_state.items():
            broadcast = np.broadcast_to(arr[:, :, None], (T, E, A) + arr.shape[2:])
            shaped = broadcast.reshape((T * E * A,) + arr.shape[2:]).copy()
            global_flat[k] = torch.from_numpy(shaped).to(device)
        # Advantage / return are per-env per-step → broadcast across agents.
        adv = torch.from_numpy(np.broadcast_to(self.advantages[:, :, None], (T, E, A))
                               .reshape(T * E * A).copy()).to(device)
        ret = torch.from_numpy(np.broadcast_to(self.returns[:, :, None], (T, E, A))
                               .reshape(T * E * A).copy()).to(device)
        old_values = torch.from_numpy(np.broadcast_to(self.values[:, :, None], (T, E, A))
                                      .reshape(T * E * A).copy()).to(device)
        # Normalise advantages.
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        N = T * E * A
        idx = torch.randperm(N, device=device)
        bs = N // num_mini_batch
        for i in range(num_mini_batch):
            mb = idx[i * bs:(i + 1) * bs] if i < num_mini_batch - 1 else idx[i * bs:]
            yield {
                "local_obs": {k: v[mb] for k, v in local_flat.items()},
                "global_state": {k: v[mb] for k, v in global_flat.items()},
                "actions": actions[mb],
                "old_log_probs": log_probs[mb],
                "advantages": adv[mb],
                "returns": ret[mb],
                "old_values": old_values[mb],
            }
