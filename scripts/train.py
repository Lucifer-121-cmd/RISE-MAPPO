"""Main training entry point for Paper 3.

Usage::

    python scripts/train.py --config configs/default.yaml --seed 42
    python scripts/train.py --config configs/default.yaml --smoke  # 2 updates

The script loads YAML, applies CLI overrides, builds env+actor+critic
+algorithm via :func:`marl.mappo.runner.build_default_pipeline`, and
runs the training loop.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import yaml

from envs.multi_robot_search_env import EnvConfig
from marl.mappo.algorithm import MAPPOConfig
from marl.mappo.runner import RunnerConfig, build_default_pipeline


_LOG = logging.getLogger("paper3.train")


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f)


def _make_env_cfg(raw: Dict[str, Any], seed: int) -> EnvConfig:
    env_raw = dict(raw.get("env", {}))
    rew = dict(raw.get("reward", {}))
    return EnvConfig(
        num_robots=env_raw.get("num_robots", 3),
        world_size=env_raw.get("world_size", 10.0),
        max_steps=env_raw.get("max_steps", 500),
        sensor_range=env_raw.get("sensor_range", 1.5),
        dt=env_raw.get("dt", 0.1),
        difficulty=env_raw.get("difficulty", "medium"),
        num_targets=env_raw.get("num_targets", 5),
        num_obstacles=env_raw.get("num_obstacles", 10),
        num_hazards=env_raw.get("num_hazards", 3),
        subgoal_steps=env_raw.get("subgoal_steps", 25),
        detect_range=env_raw.get("detect_range", 0.4),
        robot_radius=env_raw.get("robot_radius", 0.105),
        energy_budget=env_raw.get("energy_budget", 100.0),
        add_noise=env_raw.get("add_noise", False),
        use_dynamic_step=env_raw.get("use_dynamic_step", False),
        seed=seed,
        w_coverage=rew.get("w_coverage", 1.0),
        w_detection=rew.get("w_detection", 5.0),
        w_cvar_risk=rew.get("w_cvar_risk", 0.5),
        w_energy=rew.get("w_energy", 0.3),
        w_coordination=rew.get("w_coordination", 0.2),
        w_collision=rew.get("w_collision", 10.0),
    )


def _make_mappo_cfg(raw: Dict[str, Any]) -> MAPPOConfig:
    m = dict(raw.get("mappo", {}))
    return MAPPOConfig(
        actor_lr=m.get("actor_lr", 5.0e-4),
        critic_lr=m.get("critic_lr", 5.0e-4),
        ppo_epoch=m.get("ppo_epoch", 10),
        clip_param=m.get("clip_param", 0.2),
        value_loss_coef=m.get("value_loss_coef", 0.5),
        entropy_coef=m.get("entropy_coef", 0.01),
        max_grad_norm=m.get("max_grad_norm", 10.0),
        num_mini_batch=m.get("num_mini_batch", 4),
        use_popart=m.get("use_popart", True),
        use_rise=m.get("use_rise", False),
        lambda_risk=m.get("lambda_risk", 0.1),
        cvar_loss_coef=m.get("cvar_loss_coef", 0.5),
        gp_attention_eta=m.get("gp_attention_eta", 1.0),
        gp_attention_heads=m.get("gp_attention_heads", 1),
    )


def _make_runner_cfg(raw: Dict[str, Any], seed: int, smoke: bool) -> RunnerConfig:
    t = dict(raw.get("training", {}))
    return RunnerConfig(
        rollout_length=8 if smoke else t.get("rollout_length", 256),
        num_envs=1 if smoke else t.get("num_envs", 4),
        n_training_updates=2 if smoke else t.get("n_training_updates", 1000),
        log_interval=t.get("log_interval", 1),
        save_interval=1 if smoke else t.get("save_interval", 50),
        eval_interval=t.get("eval_interval", 25),
        device=t.get("device", "cuda"),
        seed=seed,
        save_dir=t.get("save_dir", "results/checkpoints"),
        use_wandb=False if smoke else t.get("use_wandb", False),
        wandb_project=t.get("wandb_project", "paper3-marl-search"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MAPPO on the multi-robot search env.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true", help="Tiny config for end-to-end test.")
    parser.add_argument("--device", type=str, default=None, help="Override training device.")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)s | %(message)s",
        level=getattr(logging, args.log_level.upper(), logging.INFO),
    )
    raw = _load_config(args.config)
    env_cfg = _make_env_cfg(raw, args.seed)
    mappo_cfg = _make_mappo_cfg(raw)
    runner_cfg = _make_runner_cfg(raw, args.seed, args.smoke)
    if args.device is not None:
        runner_cfg.device = args.device
    _LOG.info(
        "config: env.num_robots=%d  rollout=%d  num_envs=%d  updates=%d  device=%s",
        env_cfg.num_robots,
        runner_cfg.rollout_length,
        runner_cfg.num_envs,
        runner_cfg.n_training_updates,
        runner_cfg.device,
    )
    runner = build_default_pipeline(env_cfg, runner_cfg, mappo_cfg)
    runner.train()


if __name__ == "__main__":
    main()
