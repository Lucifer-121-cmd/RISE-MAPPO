"""MAPPO PPO update.

Phase 2.5 RISE-MAPPO additions (gated on :attr:`MAPPOConfig.use_rise`):

* Risk-adjusted advantage ``A = A_mean − λ_risk · A_risk`` drives the
  PPO clipped policy loss.
* Two value losses, one per head, summed with a CVaR loss coefficient.
* Each head has its own PopArt normaliser; the actor is unchanged.

When ``use_rise=False`` the update is bit-for-bit equivalent to
Phase-1 MAPPO.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn

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
    # RISE-MAPPO settings.
    use_rise: bool = False
    lambda_risk: float = 0.1
    cvar_loss_coef: float = 0.5
    gp_attention_eta: float = 1.0
    gp_attention_heads: int = 1


class MAPPO:
    """Vanilla MAPPO with shared parameters across agents."""

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
        stats: Dict[str, float] = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_frac": 0.0,
            "n_updates": 0.0,
        }
        if self.cfg.use_rise:
            stats.update({
                "v_mean_loss": 0.0,
                "v_cvar_loss": 0.0,
                "risk_advantage_mean": 0.0,
                "risk_advantage_std": 0.0,
                "gp_attention_entropy": 0.0,
            })
        for _epoch in range(self.cfg.ppo_epoch):
            for batch in buffer.feed_forward_generator(self.cfg.num_mini_batch):
                metrics = self._update_step(batch)
                for k, v in metrics.items():
                    stats[k] = stats.get(k, 0.0) + float(v)
                stats["n_updates"] += 1
        n = max(stats["n_updates"], 1)
        for k in list(stats.keys()):
            if k == "n_updates":
                continue
            stats[k] /= n
        return stats

    # ------------------------------------------------------------------
    def _value_loss(
        self,
        new_values: torch.Tensor,
        old_values: torch.Tensor,
        targets: torch.Tensor,
        popart,
    ) -> torch.Tensor:
        """PPO clipped value loss with optional PopArt normalisation."""
        if self.cfg.use_popart and popart is not None:
            popart.update(targets)
            norm_targets = popart.normalize(targets)
            norm_old = popart.normalize(old_values)
            norm_new = popart.normalize(new_values)
            v_clipped = norm_old + torch.clamp(
                norm_new - norm_old, -self.cfg.clip_param, self.cfg.clip_param,
            )
            uncl = (norm_new - norm_targets).pow(2)
            cl = (v_clipped - norm_targets).pow(2)
        else:
            v_clipped = old_values + torch.clamp(
                new_values - old_values, -self.cfg.clip_param, self.cfg.clip_param,
            )
            uncl = (new_values - targets).pow(2)
            cl = (v_clipped - targets).pow(2)
        return 0.5 * torch.max(uncl, cl).mean()

    # ------------------------------------------------------------------
    def _update_step(self, batch: Dict[str, object]) -> Dict[str, torch.Tensor]:
        local_obs = batch["local_obs"]
        global_state = batch["global_state"]
        actions = batch["actions"]
        old_log_probs = batch["old_log_probs"]
        if self.cfg.use_rise:
            return self._rise_update_step(
                local_obs, global_state, actions, old_log_probs, batch,
            )
        # Phase-1 path (unchanged).
        advantages = batch["advantages"]
        returns = batch["returns"]
        old_values = batch["old_values"]
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
        new_values = self.critic(global_state)
        value_loss = self._value_loss(new_values, old_values, returns, self.critic.popart)
        critic_total = self.cfg.value_loss_coef * value_loss
        self.critic_optim.zero_grad(set_to_none=True)
        critic_total.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
        self.critic_optim.step()
        with torch.no_grad():
            approx_kl = (old_log_probs - new_log_probs).mean()
            clip_frac = ((ratio - 1.0).abs() > self.cfg.clip_param).float().mean()
        return {
            "policy_loss": policy_loss.detach(),
            "value_loss": value_loss.detach(),
            "entropy": ent_term.detach(),
            "approx_kl": approx_kl.detach(),
            "clip_frac": clip_frac.detach(),
        }

    # ------------------------------------------------------------------
    def _rise_update_step(
        self,
        local_obs,
        global_state,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        batch: Dict[str, object],
    ) -> Dict[str, torch.Tensor]:
        """RISE-MAPPO update with risk-adjusted advantage and dual-head loss."""
        adv_mean_raw = batch["advantages_mean_raw"]
        adv_risk_raw = batch["advantages_risk_raw"]
        returns_mean = batch["returns"]
        returns_risk = batch["returns_risk"]
        old_values_mean = batch["old_values"]
        old_values_cvar = batch["old_values_cvar"]
        agent_sigmas = batch["agent_sigmas"]
        # Risk-adjusted advantage (core RISE-MAPPO novelty).
        adv = adv_mean_raw - self.cfg.lambda_risk * adv_risk_raw
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        # Actor (PPO clipped).
        new_log_probs, entropy = self.actor.evaluate_action(local_obs, actions)
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_param, 1.0 + self.cfg.clip_param) * adv
        policy_loss = -torch.min(surr1, surr2).mean()
        ent_term = entropy.mean()
        actor_total = policy_loss - self.cfg.entropy_coef * ent_term
        self.actor_optim.zero_grad(set_to_none=True)
        actor_total.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
        self.actor_optim.step()
        # Critic (dual-head).
        v_mean_pred, v_cvar_pred = self.critic(global_state, agent_sigmas=agent_sigmas)
        v_mean_loss = self._value_loss(
            v_mean_pred, old_values_mean, returns_mean, self.critic.popart_mean,
        )
        v_cvar_loss = self._value_loss(
            v_cvar_pred, old_values_cvar, returns_risk, self.critic.popart_cvar,
        )
        value_loss = v_mean_loss + self.cfg.cvar_loss_coef * v_cvar_loss
        critic_total = self.cfg.value_loss_coef * value_loss
        self.critic_optim.zero_grad(set_to_none=True)
        critic_total.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
        self.critic_optim.step()
        with torch.no_grad():
            approx_kl = (old_log_probs - new_log_probs).mean()
            clip_frac = ((ratio - 1.0).abs() > self.cfg.clip_param).float().mean()
            risk_adv_mean = adv_risk_raw.mean()
            risk_adv_std = adv_risk_raw.std()
            # Diagnostic: attention entropy on the same batch (re-forward cheap).
            try:
                attn_entropy = self._attention_entropy(global_state, agent_sigmas)
            except Exception:
                attn_entropy = torch.zeros((), device=adv.device)
        return {
            "policy_loss": policy_loss.detach(),
            "value_loss": value_loss.detach(),
            "v_mean_loss": v_mean_loss.detach(),
            "v_cvar_loss": v_cvar_loss.detach(),
            "entropy": ent_term.detach(),
            "approx_kl": approx_kl.detach(),
            "clip_frac": clip_frac.detach(),
            "risk_advantage_mean": risk_adv_mean,
            "risk_advantage_std": risk_adv_std,
            "gp_attention_entropy": attn_entropy,
        }

    @torch.no_grad()
    def _attention_entropy(self, global_state, agent_sigmas: torch.Tensor) -> torch.Tensor:
        """Mean Shannon entropy of attention rows (lower = more focused)."""
        feats = self.critic._per_agent_features(global_state)
        _, weights = self.critic.gp_attention(feats, agent_sigmas)
        ent = -(weights * (weights.clamp_min(1e-12)).log()).sum(dim=-1).mean()
        return ent
