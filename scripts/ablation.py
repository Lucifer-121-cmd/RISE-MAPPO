"""Run ablation experiments by zeroing reward weights one at a time.

Each ablation is a short MAPPO training run with a single reward
component disabled. Outputs a CSV of summary metrics.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, List

import yaml

from envs.multi_robot_search_env import EnvConfig
from marl.mappo.algorithm import MAPPOConfig
from marl.mappo.runner import RunnerConfig, build_default_pipeline


_LOG = logging.getLogger("paper3.ablation")


_ABLATIONS: List[Dict[str, float]] = [
    {"name": "full"},
    {"name": "no_cvar", "w_cvar_risk": 0.0},
    {"name": "no_energy", "w_energy": 0.0},
    {"name": "no_coordination", "w_coordination": 0.0},
    {"name": "no_collision_penalty", "w_collision": 0.0},
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--out", type=Path, default=Path("results/ablation.csv"))
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s | %(message)s")

    with args.config.open("r") as f:
        raw = yaml.safe_load(f)
    rows: List[Dict[str, float]] = []
    for spec in _ABLATIONS:
        rew = dict(raw.get("reward", {}))
        for k, v in spec.items():
            if k == "name":
                continue
            rew[k] = v
        env_raw = dict(raw.get("env", {}))
        env_cfg = EnvConfig(
            seed=42,
            **env_raw,
            **{f"w_{k}": v for k, v in rew.items() if k.startswith("w_")},
        )
        runner_cfg = RunnerConfig(
            rollout_length=128,
            num_envs=2,
            n_training_updates=args.updates,
            log_interval=10,
            save_interval=args.updates,
            device=args.device,
            seed=42,
            save_dir=str(Path("results/ablation") / spec["name"]),
        )
        mappo_cfg = MAPPOConfig()
        runner = build_default_pipeline(env_cfg, runner_cfg, mappo_cfg)
        runner.train()
        # Just log; full evaluation hooks live in scripts/evaluate.py.
        rows.append({"name": spec["name"], "updates": args.updates})
        _LOG.info("ablation %s done", spec["name"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "updates"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
