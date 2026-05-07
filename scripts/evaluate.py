"""Evaluate a saved MAPPO checkpoint.

Loads ``--checkpoint``, runs ``--episodes`` rollouts under the env
config in ``--config``, and prints aggregate metrics: mean return,
coverage, detected fraction, collisions per episode.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml

from envs.multi_robot_search_env import EnvConfig, MultiRobotSearchEnv
from marl.mappo.actor import Actor, ActorConfig


_LOG = logging.getLogger("paper3.eval")


def _load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def _build_env_cfg(raw: dict, seed: int) -> EnvConfig:
    env_raw = dict(raw.get("env", {}))
    rew = dict(raw.get("reward", {}))
    return EnvConfig(seed=seed, **{**env_raw, **{f"w_{k}": v for k, v in rew.items() if k.startswith("w")}})


def evaluate(
    actor: Actor,
    env: MultiRobotSearchEnv,
    episodes: int,
    deterministic: bool,
    device: torch.device,
) -> Dict[str, float]:
    returns: List[float] = []
    coverages: List[float] = []
    detections: List[float] = []
    collisions: List[float] = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        ep_return = 0.0
        info: Dict[str, float] = {}
        while not done:
            local = {k: torch.from_numpy(np.stack([obs[a][k] for a in env.agents], axis=0)).float().to(device)
                     for k in env.observation_shapes()}
            with torch.no_grad():
                action, _, _ = actor.get_action(local, deterministic=deterministic)
            act_dict = {a: int(action[i].item()) for i, a in enumerate(env.agents)}
            obs, rew_dict, term, trunc, info = env.step(act_dict)
            ep_return += float(next(iter(rew_dict.values())))
            done = any(term.values()) or any(trunc.values())
        returns.append(ep_return)
        coverages.append(float(info.get("coverage", 0.0)))
        detections.append(float(info.get("detected", 0.0)))
        collisions.append(float(info.get("collisions_total", 0.0)))
    return {
        "mean_return": float(np.mean(returns)),
        "mean_coverage": float(np.mean(coverages)),
        "mean_detected": float(np.mean(detections)),
        "mean_collisions": float(np.mean(collisions)),
        "n_episodes": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s | %(message)s")
    raw = _load_yaml(args.config)
    env = MultiRobotSearchEnv(_build_env_cfg(raw, args.seed))
    actor = Actor(ActorConfig(
        num_actions=env.num_subgoal_actions,
        num_robots=env.cfg.num_robots,
    ))
    state = torch.load(args.checkpoint, map_location=args.device)
    actor.load_state_dict(state["actor"])
    actor.to(args.device).eval()
    metrics = evaluate(actor, env, args.episodes, args.deterministic, torch.device(args.device))
    _LOG.info("evaluation metrics: %s", metrics)


if __name__ == "__main__":
    main()
