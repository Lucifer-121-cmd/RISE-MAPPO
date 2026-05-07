"""MAPPO centralised critic.

Consumes the global state dict and outputs a scalar value V(s). The
state contains the merged occupancy grid, the GP sigma map, all robot
poses, energies, and a small "stats" vector. PopArt normalisation is
applied to value targets externally; the critic's raw output is the
*unnormalised* value (as in the original MAPPO PopArt setup).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn

from marl.utils import PopArt


@dataclass
class CriticConfig:
    grid_size: int = 100
    num_robots: int = 3
    cnn_channels: Tuple[int, int, int] = (16, 32, 64)
    mlp_hidden: int = 128
    fusion_hidden: int = 128
    use_popart: bool = True


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


class Critic(nn.Module):
    """Centralised value network with optional PopArt normalisation."""

    def __init__(self, cfg: CriticConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.cnn = _build_global_cnn(in_channels=2, channels=cfg.cnn_channels)
        cnn_out = cfg.cnn_channels[2] * 2 * 2
        vec_in = cfg.num_robots * 3 + cfg.num_robots + 2  # robot_states, energies, stats
        self.vec_mlp = nn.Sequential(
            nn.Linear(vec_in, cfg.mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.mlp_hidden, cfg.mlp_hidden),
            nn.ReLU(inplace=True),
        )
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

    def forward(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return scalar V(state) for the centralised global state."""
        img = self._build_image_input(state)
        vec = self._build_vector_input(state)
        h = torch.cat([self.cnn(img), self.vec_mlp(vec)], dim=-1)
        return self.head(h).squeeze(-1)
