"""MAPPO actor (decentralised policy).

The actor consumes a per-agent observation dict and outputs logits over
the discrete subgoal action space. Image features (occupancy patch and
GP uncertainty patch) go through a small CNN; vector features are
concatenated and routed through an MLP. CNN and MLP features are fused
in a final MLP head.

The class is deliberately compact (~150 LOC) so it is easy to read and
modify: this is research code, not a production training stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical


@dataclass
class ActorConfig:
    patch_size: int = 48
    num_actions: int = 25
    num_robots: int = 3
    cnn_channels: Tuple[int, int, int] = (16, 32, 32)
    mlp_hidden: int = 128
    fusion_hidden: int = 128


def _vector_feature_dim(num_robots: int) -> int:
    """Sum of vector-component dims fed into the MLP branch."""
    rel = max(num_robots - 1, 1) * 2
    energy = 1
    coverage = 1
    agent_id = num_robots
    return rel + energy + coverage + agent_id


def _build_cnn(in_channels: int, channels: Tuple[int, int, int]) -> nn.Sequential:
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


class Actor(nn.Module):
    """Per-agent actor network."""

    def __init__(self, cfg: ActorConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.cnn = _build_cnn(in_channels=2, channels=cfg.cnn_channels)
        cnn_out = cfg.cnn_channels[2] * 2 * 2
        vec_in = _vector_feature_dim(cfg.num_robots)
        self.vec_mlp = nn.Sequential(
            nn.Linear(vec_in, cfg.mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.mlp_hidden, cfg.mlp_hidden),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(cnn_out + cfg.mlp_hidden, cfg.fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.fusion_hidden, cfg.num_actions),
        )

    @staticmethod
    def _build_image_input(obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        occ = obs["occupancy_patch"]
        sigma = obs["gp_sigma_patch"]
        if occ.dim() == 3:
            occ = occ.unsqueeze(1)
            sigma = sigma.unsqueeze(1)
        elif occ.dim() == 4 and occ.shape[1] != 1:
            # Already (B, C, H, W) with C=1. Leave as-is.
            pass
        return torch.cat([occ, sigma], dim=1)

    @staticmethod
    def _build_vector_input(obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        rel = obs["rel_robots"]
        if rel.dim() == 3:
            rel = rel.flatten(start_dim=1)
        elif rel.dim() == 2 and rel.shape[1] == 2:
            # Single sample (N-1, 2) coming through.
            rel = rel.flatten().unsqueeze(0)
        energy = obs["energy"]
        coverage = obs["coverage"]
        agent_id = obs["agent_id"]
        return torch.cat([rel, energy, coverage, agent_id], dim=-1)

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return action logits for ``obs`` (Categorical over subgoals)."""
        img = self._build_image_input(obs)
        vec = self._build_vector_input(obs)
        h_img = self.cnn(img)
        h_vec = self.vec_mlp(vec)
        h = torch.cat([h_img, h_vec], dim=-1)
        return self.head(h)

    def get_action(
        self,
        obs: Dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action; return ``(action, log_prob, entropy)``."""
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def evaluate_action(
        self,
        obs: Dict[str, torch.Tensor],
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Re-evaluate ``log_prob`` and ``entropy`` for stored actions."""
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(action), dist.entropy()
