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

import logging
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn

from marl.mappo.actor import Actor
from marl.mappo.buffer import RolloutBuffer
from marl.mappo.critic import Critic


_LOG = logging.getLogger("paper3.mappo")


@dataclass
class MAPPOConfig:
    actor_lr: float = 5.0e-4
    critic_lr: float = 5.0e-4
    ppo_epoch: int = 10
    clip_param: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5  # FIXED: was 10.0, now matches default.yaml
    num_mini_batch: int = 4
    use_popart: bool = True
    # RISE-MAPPO settings.
    use_rise: bool = False
    lambda_risk: float = 0.05  # FIXED: was 0.1, now matches default.yaml
    cvar_loss_coef: float = 0.25  # FIXED: was 0.5, now matches default.yaml
    gp_attention_eta: float = 0.5  # FIXED: was 1.0, now matches default.yaml
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
        self._update_count = 0
        self.diag_print = False

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
        self._update_count += 1
        self._diag_emitted = False
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
        """PPO clipped value loss with optional PopArt normalisation.

        Standard PopArt convention: the critic head output (``new_values``)
        lives in *normalised* space; ``old_values`` and ``targets`` come from
        the buffer in *raw* (de-normalised) space and must be normalised
        before comparing.  The previous form also normalised ``new_values``,
        producing ``((new − target)/σ)²`` whose minimiser is ``new → target``
        in *raw* scale, but the rollout still de-normalised the head as
        ``head·σ + μ``. That inflated returns by σ, grew σ, and eventually
        overflowed ``mean_sq`` to FP +inf, freezing the loss at 0.
        """
        targets = torch.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)
        old_values = torch.nan_to_num(old_values, nan=0.0)
        if self.cfg.use_popart and popart is not None:
            popart.update(targets)
            norm_targets = popart.normalize(targets)
            norm_old = popart.normalize(old_values)
            v_clipped = norm_old + torch.clamp(
                new_values - norm_old, -self.cfg.clip_param, self.cfg.clip_param,
            )
            uncl = (new_values - norm_targets).pow(2)
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
        first_mb = self.diag_print and not getattr(self, "_diag_emitted", True)
        if first_mb:
            with torch.no_grad():
                pm = self.critic.popart_mean
                pc = self.critic.popart_cvar
                def _stat(t):
                    return (
                        float(t.min()), float(t.max()), float(t.mean()),
                        int(torch.isnan(t).sum()), int(torch.isinf(t).sum()),
                    )
                rmn = _stat(returns_mean)
                rrk = _stat(returns_risk)
                _LOG.debug(
                    f"[DIAG upd={self._update_count}] PRE-NAN ret_mean min/max/mean={rmn[0]:.3g}/{rmn[1]:.3g}/{rmn[2]:.3g} "
                    f"nan={rmn[3]} inf={rmn[4]} | ret_risk min/max/mean={rrk[0]:.3g}/{rrk[1]:.3g}/{rrk[2]:.3g} "
                    f"nan={rrk[3]} inf={rrk[4]}"
                )
                _LOG.debug(
                    f"[DIAG upd={self._update_count}] PopArt mean: mu={float(pm.mean):.4g} sigma={float(pm.std):.4g} debias={float(pm.debias):.4g} | "
                    f"PopArt cvar: mu={float(pc.mean):.4g} sigma={float(pc.std):.4g} debias={float(pc.debias):.4g}"
                )
        # Sanitise inputs: a single NaN slipping in (e.g. from a previously
        # NaN-poisoned PopArt buffer) would otherwise wreck params here.
        adv_mean_raw = torch.nan_to_num(adv_mean_raw, nan=0.0, posinf=0.0, neginf=0.0)
        adv_risk_raw = torch.nan_to_num(adv_risk_raw, nan=0.0, posinf=0.0, neginf=0.0)
        returns_mean = torch.nan_to_num(returns_mean, nan=0.0, posinf=0.0, neginf=0.0)
        returns_risk = torch.nan_to_num(returns_risk, nan=0.0, posinf=0.0, neginf=0.0)
        old_values_mean = torch.nan_to_num(old_values_mean, nan=0.0)
        old_values_cvar = torch.nan_to_num(old_values_cvar, nan=0.0)
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
        if torch.isfinite(actor_total):
            self.actor_optim.zero_grad(set_to_none=True)
            actor_total.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
            if all(p.grad is None or torch.isfinite(p.grad).all() for p in self.actor.parameters()):
                self.actor_optim.step()
        # Critic (dual-head).
        v_mean_pred, v_cvar_pred = self.critic(global_state, agent_sigmas=agent_sigmas)
        if first_mb:
            with torch.no_grad():
                vm = v_mean_pred
                vc = v_cvar_pred
                _LOG.debug(
                    f"[DIAG upd={self._update_count}] v_mean_pred min/max/mean={float(vm.min()):.4g}/{float(vm.max()):.4g}/{float(vm.mean()):.4g} | "
                    f"v_cvar_pred min/max/mean={float(vc.min()):.4g}/{float(vc.max()):.4g}/{float(vc.mean()):.4g}"
                )
        v_mean_loss = self._value_loss(
            v_mean_pred, old_values_mean, returns_mean, self.critic.popart_mean,
        )
        v_cvar_loss = self._value_loss(
            v_cvar_pred, old_values_cvar, returns_risk, self.critic.popart_cvar,
        )
        if first_mb:
            _LOG.debug(
                f"[DIAG upd={self._update_count}] RAW LOSS v_mean={float(v_mean_loss):.6g} v_cvar={float(v_cvar_loss):.6g}"
            )
        value_loss = v_mean_loss + self.cfg.cvar_loss_coef * v_cvar_loss
        critic_total = self.cfg.value_loss_coef * value_loss
        if torch.isfinite(critic_total):
            self.critic_optim.zero_grad(set_to_none=True)
            critic_total.backward()
            pre_clip_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
            if first_mb:
                _LOG.debug(
                    f"[DIAG upd={self._update_count}] critic grad_norm (pre-clip)={float(pre_clip_norm):.6g} max_grad_norm={self.cfg.max_grad_norm}"
                )
            if all(p.grad is None or torch.isfinite(p.grad).all() for p in self.critic.parameters()):
                self.critic_optim.step()
        if first_mb:
            self._diag_emitted = True
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
