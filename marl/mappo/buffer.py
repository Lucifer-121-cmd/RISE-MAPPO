"""Rollout buffer for MAPPO with GAE.

Phase-2.5 RISE-MAPPO additions (gated on :attr:`BufferConfig.use_rise`):

* ``risk_costs`` — per-step CVaR risk signal stored alongside the
  reward so the CVaR head can be trained against its own GAE return.
* ``values_cvar`` — second value head's predictions per step.
* ``agent_sigmas`` — per-agent GP sigma at each step, broadcast across
  agents at sampling time so the critic forward pass can re-apply
  GP-uncertainty attention during PPO updates.
* :meth:`compute_returns_and_advantages` runs a second GAE pass over
  ``risk_costs`` / ``values_cvar`` and stores ``advantages_risk`` and
  ``returns_risk``.

When ``use_rise=False`` the buffer is bit-for-bit equivalent to the
Phase-1 implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Mapping, Optional, Tuple

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
    use_rise: bool = False


class RolloutBuffer:
    """Stores (T, E, A, ...) rollouts and computes GAE returns."""

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
        if cfg.use_rise:
            self.risk_costs = np.zeros((T, E), dtype=np.float32)
            self.values_cvar = np.zeros((T, E), dtype=np.float32)
            self.advantages_risk = np.zeros((T, E), dtype=np.float32)
            self.returns_risk = np.zeros((T, E), dtype=np.float32)
            self.agent_sigmas = np.zeros((T, E, A), dtype=np.float32)

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
        risk_costs: Optional[np.ndarray] = None,
        values_cvar: Optional[np.ndarray] = None,
        agent_sigmas: Optional[np.ndarray] = None,
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
        if self.cfg.use_rise:
            if risk_costs is not None:
                self.risk_costs[t] = risk_costs
            if values_cvar is not None:
                self.values_cvar[t] = values_cvar
            if agent_sigmas is not None:
                self.agent_sigmas[t] = agent_sigmas

    # ------------------------------------------------------------------
    def _gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        last_values: np.ndarray,
        last_dones: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        T = self.cfg.rollout_length
        adv = np.zeros_like(rewards)
        gae = np.zeros(self.cfg.num_envs, dtype=np.float32)
        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 1.0 - last_dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_values = values[t + 1]
            delta = (
                rewards[t]
                + self.cfg.gamma * next_values * next_non_terminal
                - values[t]
            )
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * next_non_terminal * gae
            adv[t] = gae
        ret = adv + values
        return adv, ret

    def compute_returns_and_advantages(
        self,
        last_values: np.ndarray,
        last_dones: np.ndarray,
        last_values_cvar: Optional[np.ndarray] = None,
    ) -> None:
        """GAE with γ-λ; bootstraps from ``last_values`` at horizon.

        When :attr:`BufferConfig.use_rise` is enabled, also runs a
        second GAE pass over ``risk_costs`` against ``values_cvar``.
        """
        adv, ret = self._gae(self.rewards, self.values, last_values, last_dones)
        self.advantages = adv
        self.returns = ret
        if self.cfg.use_rise:
            lv_cvar = (
                last_values_cvar
                if last_values_cvar is not None
                else np.zeros_like(last_values)
            )
            adv_r, ret_r = self._gae(
                self.risk_costs, self.values_cvar, lv_cvar, last_dones,
            )
            self.advantages_risk = adv_r
            self.returns_risk = ret_r

    # ------------------------------------------------------------------
    def feed_forward_generator(
        self,
        num_mini_batch: int,
    ) -> Iterator[Dict[str, torch.Tensor]]:
        """Yield mini-batches over (T*E*A) flattened transitions."""
        T = self.cfg.rollout_length
        E = self.cfg.num_envs
        A = self.cfg.num_agents
        device = torch.device(self.cfg.device)
        local_flat = {
            k: torch.from_numpy(arr.reshape((T * E * A,) + arr.shape[3:])).to(device)
            for k, arr in self.local_obs.items()
        }
        actions = torch.from_numpy(self.actions.reshape(T * E * A)).to(device)
        log_probs = torch.from_numpy(self.log_probs.reshape(T * E * A)).to(device)
        global_flat: Dict[str, torch.Tensor] = {}
        for k, arr in self.global_state.items():
            broadcast = np.broadcast_to(arr[:, :, None], (T, E, A) + arr.shape[2:])
            shaped = broadcast.reshape((T * E * A,) + arr.shape[2:]).copy()
            global_flat[k] = torch.from_numpy(shaped).to(device)
        adv_raw = torch.from_numpy(np.broadcast_to(self.advantages[:, :, None], (T, E, A))
                                   .reshape(T * E * A).copy()).to(device)
        ret = torch.from_numpy(np.broadcast_to(self.returns[:, :, None], (T, E, A))
                               .reshape(T * E * A).copy()).to(device)
        old_values = torch.from_numpy(np.broadcast_to(self.values[:, :, None], (T, E, A))
                                      .reshape(T * E * A).copy()).to(device)
        # RISE extras (broadcast / flatten the same way).
        if self.cfg.use_rise:
            adv_risk_raw = torch.from_numpy(
                np.broadcast_to(self.advantages_risk[:, :, None], (T, E, A))
                .reshape(T * E * A).copy()
            ).to(device)
            ret_risk = torch.from_numpy(
                np.broadcast_to(self.returns_risk[:, :, None], (T, E, A))
                .reshape(T * E * A).copy()
            ).to(device)
            old_values_cvar = torch.from_numpy(
                np.broadcast_to(self.values_cvar[:, :, None], (T, E, A))
                .reshape(T * E * A).copy()
            ).to(device)
            # agent_sigmas already (T, E, A) — broadcast each agent-sample's
            # global-state with all-N sigmas: (T, E, A, A)[i,j,k,:] is the
            # full per-agent sigma vector for env j at step i.
            sigmas_b = np.broadcast_to(
                self.agent_sigmas[:, :, None, :], (T, E, A, A)
            ).reshape(T * E * A, A).copy()
            agent_sigmas = torch.from_numpy(sigmas_b).to(device)
        # Standard MAPPO path normalises the (single) mean advantage here.
        # Under RISE, the risk-adjusted combination is built and normalised
        # in the algorithm so both heads see consistent statistics.
        adv = (adv_raw - adv_raw.mean()) / (adv_raw.std() + 1e-8)
        N = T * E * A
        idx = torch.randperm(N, device=device)
        bs = N // num_mini_batch
        for i in range(num_mini_batch):
            mb = idx[i * bs:(i + 1) * bs] if i < num_mini_batch - 1 else idx[i * bs:]
            batch: Dict[str, object] = {
                "local_obs": {k: v[mb] for k, v in local_flat.items()},
                "global_state": {k: v[mb] for k, v in global_flat.items()},
                "actions": actions[mb],
                "old_log_probs": log_probs[mb],
                "advantages": adv[mb],
                "returns": ret[mb],
                "old_values": old_values[mb],
            }
            if self.cfg.use_rise:
                batch["advantages_mean_raw"] = adv_raw[mb]
                batch["advantages_risk_raw"] = adv_risk_raw[mb]
                batch["returns_risk"] = ret_risk[mb]
                batch["old_values_cvar"] = old_values_cvar[mb]
                batch["agent_sigmas"] = agent_sigmas[mb]
            yield batch
