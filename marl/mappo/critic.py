"""MAPPO centralised critic.

Phase 2.5 introduces **RISE-MAPPO** with two novel modules:

* :class:`GPUncertaintyAttention` — attention over per-agent features
  whose queries are modulated by environmental GP uncertainty (sigma)
  at each agent's position. Agents in high-uncertainty regions issue
  stronger queries, biasing the critic to coordinate toward unexplored
  areas.
* Dual-head output — the critic returns ``(V_mean, V_cvar)`` from a
  shared backbone via two separate small MLP heads. Each head is
  PopArt-normalised independently.

When :attr:`CriticConfig.use_rise` is ``False`` the critic falls back
to the Phase-1 single-scalar centralised value head and is bit-for-bit
backwards-compatible with the existing 51 tests / training pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from marl.utils import PopArt


@dataclass
class CriticConfig:
    grid_size: int = 100
    num_robots: int = 3
    cnn_channels: Tuple[int, int, int] = (16, 32, 64)
    mlp_hidden: int = 128
    fusion_hidden: int = 128
    use_popart: bool = True
    # RISE-MAPPO settings (Phase 2.5).
    use_rise: bool = False
    gp_attention_eta: float = 0.5  # FIXED: was 1.0, now matches default.yaml
    world_size: float = 10.0

def _build_global_cnn(in_channels: int, channels: Tuple[int, int, int]) -> nn.Sequential:
    c1, c2, c3 = channels
    return nn.Sequential(
        nn.Conv2d(in_channels, c1, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(output_size=(2, 2)),
        nn.Flatten(),
    )


class GPUncertaintyAttention(nn.Module):
    """Attention with queries modulated by per-agent GP sigma.

    Standard MARL attention scores ``Q·Kᵀ`` from agent features alone.
    We additionally inject the environmental uncertainty ``σ_i`` at
    each agent's position into the query: agents in uncertain regions
    generate stronger queries, focusing the critic on coordination
    toward unexplored / risky areas.
    """

    def __init__(self, feature_dim: int, eta: float = 1.0) -> None:
        super().__init__()
        self.eta = float(eta)
        self.W_query = nn.Linear(feature_dim, feature_dim)
        self.W_key = nn.Linear(feature_dim, feature_dim)
        self.W_value = nn.Linear(feature_dim, feature_dim)
        self.sigma_embed = nn.Sequential(
            nn.Linear(1, feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        agent_features: torch.Tensor,
        agent_sigmas: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Aggregate per-agent features under uncertainty-modulated attention.

        Args
        ----
        agent_features : (B, N, D) per-agent embeddings.
        agent_sigmas   : (B, N) GP sigma at each agent's position.

        Returns
        -------
        attended : (B, D) attention-pooled global feature.
        weights  : (B, N, N) attention weights (for diagnostics).
        """
        sigma_feat = self.sigma_embed(agent_sigmas.unsqueeze(-1))
        # Q is sigma-modulated (agents in uncertain regions issue stronger
        # queries); K is also sigma-modulated so other agents *attend more
        # to* peers in uncertain regions, biasing the critic to focus on
        # coordination toward unexplored / risky areas.
        Q = self.W_query(agent_features + self.eta * sigma_feat)
        K = self.W_key(agent_features + self.eta * sigma_feat)
        V = self.W_value(agent_features)
        d_k = float(Q.shape[-1])
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
        # Direct column bias: agent j with higher σ_j receives more attention
        # from every row regardless of feature content. This makes the
        # uncertainty signal a first-class first-order term in addition to
        # the learned QK content interaction.
        sigma_bias = self.eta * agent_sigmas.unsqueeze(1)  # (B, 1, N)
        scores = scores + sigma_bias
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, V)
        return attended.mean(dim=1), weights


class Critic(nn.Module):
    """Centralised value network. Phase-1 single-head OR Phase-2.5 RISE dual-head."""

    def __init__(self, cfg: CriticConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.cnn = _build_global_cnn(in_channels=2, channels=cfg.cnn_channels)
        cnn_out = cfg.cnn_channels[2] * 2 * 2
        vec_in = cfg.num_robots * 3 + cfg.num_robots + 2
        self.vec_mlp = nn.Sequential(
            nn.Linear(vec_in, cfg.mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.mlp_hidden, cfg.mlp_hidden),
            nn.ReLU(inplace=True),
        )
        if cfg.use_rise:
            # Backbone fusion: produce shared_features of dim fusion_hidden.
            self.fusion = nn.Sequential(
                nn.Linear(cnn_out + cfg.mlp_hidden, cfg.fusion_hidden),
                nn.ReLU(inplace=True),
            )
            # Per-agent embedding from [x, y, theta, energy] → fusion_hidden.
            self.agent_feat = nn.Sequential(
                nn.Linear(4, cfg.fusion_hidden),
                nn.ReLU(inplace=True),
            )
            self.gp_attention = GPUncertaintyAttention(
                feature_dim=cfg.fusion_hidden, eta=cfg.gp_attention_eta,
            )
            self.v_mean_head = nn.Sequential(
                nn.Linear(cfg.fusion_hidden, cfg.fusion_hidden // 2),
                nn.ReLU(inplace=True),
                nn.Linear(cfg.fusion_hidden // 2, 1),
            )
            self.v_cvar_head = nn.Sequential(
                nn.Linear(cfg.fusion_hidden, cfg.fusion_hidden // 2),
                nn.ReLU(inplace=True),
                nn.Linear(cfg.fusion_hidden // 2, 1),
            )
            self.popart_mean = PopArt() if cfg.use_popart else None
            self.popart_cvar = PopArt() if cfg.use_popart else None
            # Backwards-compat alias (training/runner code reads ``.popart``).
            self.popart = self.popart_mean
        else:
            self.head = nn.Sequential(
                nn.Linear(cnn_out + cfg.mlp_hidden, cfg.fusion_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(cfg.fusion_hidden, 1),
            )
            self.popart = PopArt() if cfg.use_popart else None

    @staticmethod
    def _build_image_input(state: Dict[str, torch.Tensor]) -> torch.Tensor:
        occ = state["occupancy"]
        sigma = state["gp_sigma"]
        if occ.dim() == 3:
            occ = occ.unsqueeze(1)
            sigma = sigma.unsqueeze(1)
        return torch.cat([occ, sigma], dim=1)

    @staticmethod
    def _build_vector_input(state: Dict[str, torch.Tensor]) -> torch.Tensor:
        robot_states = state["robot_states"]
        energies = state["energies"]
        stats = state["stats"]
        if robot_states.dim() == 3:
            robot_states = robot_states.flatten(start_dim=1)
        return torch.cat([robot_states, energies, stats], dim=-1)

    def _per_agent_features(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Per-agent embedding of shape (B, N, fusion_hidden)."""
        robot_states = state["robot_states"]            # (B, N, 3)
        energies = state["energies"]                    # (B, N)
        feat_in = torch.cat([robot_states, energies.unsqueeze(-1)], dim=-1)
        return self.agent_feat(feat_in)

    def extract_agent_sigmas(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Bilinear-sample ``state['gp_sigma']`` at each robot's (x, y).

        Returns
        -------
        sigmas : (B, N) tensor of GP sigma at each agent position.
        """
        sigma = state["gp_sigma"]                  # (B, G, G)
        robot_states = state["robot_states"]        # (B, N, 3)
        if sigma.dim() == 2:
            sigma = sigma.unsqueeze(0)
            robot_states = robot_states.unsqueeze(0)
        B, N = robot_states.shape[:2]
        xy = robot_states[..., :2]                  # (B, N, 2)
        norm_xy = 2.0 * xy / float(self.cfg.world_size) - 1.0
        grid = norm_xy.unsqueeze(1)                 # (B, 1, N, 2)
        sigma_4d = sigma.unsqueeze(1)               # (B, 1, G, G)
        sampled = F.grid_sample(
            sigma_4d, grid, mode="bilinear", padding_mode="border", align_corners=False,
        )
        return sampled.squeeze(1).squeeze(1)        # (B, N)

    def forward(
        self,
        state: Dict[str, torch.Tensor],
        agent_sigmas: Union[torch.Tensor, None] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute centralised value(s).

        Phase-1 mode (``use_rise=False``) returns ``V(s)`` of shape ``(B,)``.
        RISE mode returns ``(V_mean, V_cvar)`` each of shape ``(B,)``.
        """
        img = self._build_image_input(state)
        vec = self._build_vector_input(state)
        fused_in = torch.cat([self.cnn(img), self.vec_mlp(vec)], dim=-1)
        if not self.cfg.use_rise:
            return self.head(fused_in).squeeze(-1)
        shared = self.fusion(fused_in)
        if agent_sigmas is not None and self.cfg.gp_attention_eta != 0.0:
            agent_feats = self._per_agent_features(state)
            attended, _ = self.gp_attention(agent_feats, agent_sigmas)
            shared = shared + attended
        elif agent_sigmas is not None:
            # eta == 0 → attention disabled (ablation); skip residual.
            pass
        v_mean = self.v_mean_head(shared).squeeze(-1)
        v_cvar = self.v_cvar_head(shared).squeeze(-1)
        return v_mean, v_cvar
