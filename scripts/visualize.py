"""Render an episode and save snapshots.

Lightweight tool used during paper-figure prep. Produces:
 * a final-frame PNG of the world + robot trajectories;
 * a four-pane snapshot at t=0, T/4, T/2, T (when ``--snapshots`` is set).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from envs.multi_robot_search_env import EnvConfig, MultiRobotSearchEnv
from marl.mappo.actor import Actor, ActorConfig


_LOG = logging.getLogger("paper3.viz")


def _load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def rollout(
    env: MultiRobotSearchEnv,
    actor: Optional[Actor],
    device: torch.device,
    deterministic: bool = True,
):
    obs, _ = env.reset()
    traj: List[np.ndarray] = []
    snapshots = []
    done = False
    step = 0
    while not done:
        if actor is not None:
            local = {k: torch.from_numpy(np.stack([obs[a][k] for a in env.agents], axis=0)).float().to(device)
                     for k in env.observation_shapes()}
            with torch.no_grad():
                action, _, _ = actor.get_action(local, deterministic=deterministic)
            act_dict = {a: int(action[i].item()) for i, a in enumerate(env.agents)}
        else:
            act_dict = {a: int(np.random.randint(env.num_subgoal_actions)) for a in env.agents}
        obs, _, term, trunc, _ = env.step(act_dict)
        poses = np.array([env._states[a].pose.copy() for a in env.agents])
        traj.append(poses)
        if step in (0, env.cfg.max_steps // 4, env.cfg.max_steps // 2):
            snapshots.append(poses.copy())
        step += 1
        done = any(term.values()) or any(trunc.values())
    snapshots.append(np.array([env._states[a].pose.copy() for a in env.agents]))
    return env, np.stack(traj), snapshots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("results/episode.png"))
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s | %(message)s")

    raw = _load_yaml(args.config)
    env_raw = dict(raw.get("env", {}))
    rew = dict(raw.get("reward", {}))
    env = MultiRobotSearchEnv(EnvConfig(seed=0, **{**env_raw, **{f"w_{k}": v for k, v in rew.items() if k.startswith("w")}}))
    actor: Optional[Actor] = None
    if args.checkpoint is not None:
        actor = Actor(ActorConfig(num_actions=env.num_subgoal_actions, num_robots=env.cfg.num_robots))
        state = torch.load(args.checkpoint, map_location=args.device)
        actor.load_state_dict(state["actor"])
        actor.to(args.device).eval()
    env, traj, _ = rollout(env, actor, torch.device(args.device))
    fig, ax = plt.subplots(figsize=(7, 7))
    env.render(ax)
    for r in range(traj.shape[1]):
        ax.plot(traj[:, r, 0], traj[:, r, 1], "-", alpha=0.6)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    _LOG.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
