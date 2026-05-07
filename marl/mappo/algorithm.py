"""MAPPO PPO update."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from marl.mappo.actor import Actor
from marl.mappo.buffer import RolloutBuffer
from marl.mappo.critic import Critic


@dataclass
class MAPPOConfig:
    actor_lr: float = 5.0e-4
    critic_lr: float = 5.0e-4
    ppo_epoch: int = 10
    clip_param: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 10.0
    num_mini_batch: int = 4
    use_popart: bool = True


class MAPPO:
    """Vanilla MAPPO with shared parameters across agents.

    *Parameter sharing* (one actor for all agents) is the standard
    MAPPO setup; agent identity is supplied via the ``agent_id``
    one-hot in the observation. The centralised critic has its own
    optimiser.
    """

    def __init__(
        self,
        actor: Actor,
        critic: Critic,
        config: MAPPOConfig,
        device: torch.device,
    ) -> None:
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.cfg = config
        self.device = device
        self.actor_optim = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
        self.critic_optim = torch.optim.Adam(critic.parameters(), lr=config.critic_lr)

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """Run :attr:`MAPPOConfig.ppo_epoch` PPO epochs over ``buffer``."""
        stats = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_frac": 0.0,
            "n_updates": 0,
        }
        for _epoch in range(self.cfg.ppo_epoch):
            for batch in buffer.feed_forward_generator(self.cfg.num_mini_batch):
                p_loss, v_loss, ent, kl, clip = self._update_step(batch)
                stats["policy_loss"] += float(p_loss)
                stats["value_loss"] += float(v_loss)
                stats["entropy"] += float(ent)
                stats["approx_kl"] += float(kl)
                stats["clip_frac"] += float(clip)
                stats["n_updates"] += 1
        n = max(stats["n_updates"], 1)
        for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac"):
            stats[k] /= n
        return stats

    # ------------------------------------------------------------------
    def _update_step(
        self,
        batch: Dict[str, object],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        local_obs = batch["local_obs"]
        global_state = batch["global_state"]
        actions = batch["actions"]
        old_log_probs = batch["old_log_probs"]
        advantages = batch["advantages"]
        returns = batch["returns"]
        old_values = batch["old_values"]
        # Policy.
        new_log_probs, entropy = self.actor.evaluate_action(local_obs, actions)
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_param, 1.0 + self.cfg.clip_param) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        ent_term = entropy.mean()
        actor_total = policy_loss - self.cfg.entropy_coef * ent_term
        self.actor_optim.zero_grad(set_to_none=True)
        actor_total.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
        self.actor_optim.step()
        # Critic with optional PopArt.
        new_values = self.critic(global_state)
        targets = returns
        if self.cfg.use_popart and self.critic.popart is not None:
            self.critic.popart.update(targets)
            norm_targets = self.critic.popart.normalize(targets)
            norm_old_values = self.critic.popart.normalize(old_values)
            norm_new_values = self.critic.popart.normalize(new_values)
            v_pred_clipped = norm_old_values + torch.clamp(
                norm_new_values - norm_old_values, -self.cfg.clip_param, self.cfg.clip_param
            )
            v_loss_unclipped = (norm_new_values - norm_targets).pow(2)
            v_loss_clipped = (v_pred_clipped - norm_targets).pow(2)
            value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
        else:
            v_pred_clipped = old_values + torch.clamp(
                new_values - old_values, -self.cfg.clip_param, self.cfg.clip_param
            )
            v_loss_unclipped = (new_values - targets).pow(2)
            v_loss_clipped = (v_pred_clipped - targets).pow(2)
            value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
        critic_total = self.cfg.value_loss_coef * value_loss
        self.critic_optim.zero_grad(set_to_none=True)
        critic_total.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
        self.critic_optim.step()
        with torch.no_grad():
            approx_kl = (old_log_probs - new_log_probs).mean()
            clip_frac = ((ratio - 1.0).abs() > self.cfg.clip_param).float().mean()
        return policy_loss.detach(), value_loss.detach(), ent_term.detach(), approx_kl.detach(), clip_frac.detach()
