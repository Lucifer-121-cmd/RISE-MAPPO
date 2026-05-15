"""Evaluation entry point for RISE-MAPPO + baselines.

Loads a policy (trained checkpoint or baseline from registry), runs N
episodes under a scenario config, collects per-timestep data, and
persists metrics + raw episode blobs for downstream plotting. Runs
strictly on CPU (training holds the GPU).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

# Repo-root on sys.path so `python scripts/evaluate.py ...` works without -m.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analysis.metrics import (  # noqa: E402
    METRIC_KEYS,
    EpisodeData,
    compute_all_metrics,
    coverage_over_time,
)
from baselines import BASELINE_REGISTRY  # noqa: E402
from baselines.base_policy import BasePolicy  # noqa: E402
from baselines.trained_policy import AblationPolicy, TrainedPolicy  # noqa: E402
from envs.multi_robot_search_env import EnvConfig, MultiRobotSearchEnv  # noqa: E402


_LOG = logging.getLogger("paper3.evaluate")
_VALID_ENV_FIELDS = set(EnvConfig.__dataclass_fields__.keys())


def _load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def _merge(base: dict, override: dict) -> dict:
    """Shallow per-section merge (override wins on duplicate keys)."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _build_env_cfg(merged: dict, seed: int) -> EnvConfig:
    raw = dict(merged.get("env", {}))
    rew = dict(merged.get("reward", {}))
    for rk, rv in rew.items():
        key = rk if rk.startswith("w_") else f"w_{rk}"
        raw[key] = rv
    # Pull top-level 'mpc' section into the env config so _make_controller()
    # receives the configured MPC parameters rather than LyapunovMPCConfig
    # dataclass defaults.  The 'mpc' key lives outside 'env' in the YAML.
    if "mpc" in merged:
        raw["mpc"] = merged["mpc"]
    # Drop keys not in EnvConfig (e.g. scenario-only fields).
    safe = {k: v for k, v in raw.items() if k in _VALID_ENV_FIELDS}
    dropped = set(raw.keys()) - set(safe.keys())
    if dropped:
        _LOG.debug("_build_env_cfg dropped keys not in EnvConfig: %s", sorted(dropped))
    return EnvConfig(seed=seed, **safe)


def build_policy(
    kind: str,
    *,
    checkpoint: Path = None,
    config: Path = None,
    name: str = "",
    deterministic: bool = True,
    device: str = "cpu",
    num_actions: int = 25,
    num_robots: int = 3,
) -> BasePolicy:
    if kind == "trained":
        if checkpoint is None or config is None:
            raise ValueError("--checkpoint and --config required for --policy trained")
        if name:
            return AblationPolicy(
                checkpoint_path=checkpoint, config_path=config, name=name,
                deterministic=deterministic, device=device,
            )
        return TrainedPolicy(
            checkpoint_path=checkpoint, config_path=config,
            deterministic=deterministic, device=device,
        )
    if kind not in BASELINE_REGISTRY:
        raise KeyError(f"unknown policy '{kind}'. Options: trained, {list(BASELINE_REGISTRY)}")
    if kind == "random":
        return BASELINE_REGISTRY[kind](num_actions=num_actions, seed=0)
    return BASELINE_REGISTRY[kind](num_actions=num_actions)


def _build_scene(env: MultiRobotSearchEnv) -> Dict[str, np.ndarray]:
    """Augment env.global_state() with the coverage map + scenario scalars."""
    gs = env.global_state()
    gs["coverage_map"] = env._coverage_mask.copy()
    gs["world_size"] = float(env.cfg.world_size)
    return gs


def _per_robot_pose(env: MultiRobotSearchEnv) -> np.ndarray:
    return np.stack([env._states[a].pose.copy() for a in env.agents], axis=0)


def _per_robot_energy(env: MultiRobotSearchEnv) -> np.ndarray:
    return np.array([env._states[a].energy for a in env.agents], dtype=np.float32)


def _per_robot_cvar(env: MultiRobotSearchEnv) -> np.ndarray:
    out = np.zeros(env.cfg.num_robots, dtype=np.float32)
    for i, a in enumerate(env.agents):
        out[i] = float(env.gp.cvar_risk(env._states[a].pose[:2]))
    return out


def _per_robot_crashed(env: MultiRobotSearchEnv) -> np.ndarray:
    return np.array([1.0 if env._states[a].crashed else 0.0 for a in env.agents], dtype=np.float32)


def _per_robot_lyapunov(env: MultiRobotSearchEnv, ep: EpisodeData = None) -> np.ndarray:
    """Compute Lyapunov V(x) = 0.5 * ||pose - reference||² for each robot.

    When ``ep.lyapunov_reference`` is set (baselines that change subgoals every
    MARL step), the reference is a fixed spawn position so the metric measures
    convergence toward a stationary point.  Otherwise the reference is the
    robot's current committed subgoal (RISE-MAPPO / trained policies).
    """
    out = np.zeros(env.cfg.num_robots, dtype=np.float32)
    for i, a in enumerate(env.agents):
        pose = env._states[a].pose[:2]
        if ep is not None and ep.lyapunov_reference is not None:
            ref = ep.lyapunov_reference[i]
        else:
            ref = env._states[a].last_subgoal
        dx = pose[0] - ref[0]
        dy = pose[1] - ref[1]
        out[i] = 0.5 * (dx * dx + dy * dy)
    return out


def run_episode(policy: BasePolicy, env: MultiRobotSearchEnv, seed: int) -> EpisodeData:
    """One eval episode. Returns the populated :class:`EpisodeData`."""
    obs, _info = env.reset(seed=seed)
    policy.reset(num_robots=env.cfg.num_robots)
    ep = EpisodeData(
        num_robots=env.cfg.num_robots,
        num_targets=env.cfg.num_targets,
        max_steps=env.cfg.max_steps,
        world_size=float(env.cfg.world_size),
    )
    # For baselines that change subgoals every MARL step (Nearest Frontier,
    # Random, Voronoi), set a fixed Lyapunov reference at the spawn position
    # so the metric measures convergence toward a stationary point rather than
    # a moving target (which would trivially yield monotonic_fraction = 1.0).
    if not isinstance(policy, (TrainedPolicy, AblationPolicy)):
        ep.lyapunov_reference = _per_robot_pose(env)[:, :2].copy()
    energy_init = _per_robot_energy(env)
    detected_prev = 0
    crashed_prev = _per_robot_crashed(env)
    done = False
    while not done:
        scene = _build_scene(env)
        action_arr = policy.get_actions(obs, scene)
        action_dict = {a: int(action_arr[i]) for i, a in enumerate(env.agents)}
        obs, rew_dict, term, trunc, info = env.step(action_dict)
        done = any(term.values()) or any(trunc.values())
        # Per-step bookkeeping.
        ep.robot_positions.append(_per_robot_pose(env))
        ep.coverage_maps.append(env._coverage_mask.copy())
        ep.gp_uncertainty.append(env.gp.uncertainty_grid().copy())
        ep.cvar_values.append(_per_robot_cvar(env))
        ep.lyapunov_values.append(_per_robot_lyapunov(env, ep))
        cur_energy = _per_robot_energy(env)
        ep.energy_consumed.append((energy_init - cur_energy).astype(np.float32))
        crashed_now = _per_robot_crashed(env)
        ep.collisions.append((crashed_now - crashed_prev).clip(min=0.0))
        crashed_prev = crashed_now
        detected_now = int(info.get("detected", 0))
        ep.detections.append(max(0, detected_now - detected_prev))
        detected_prev = detected_now
        ep.rewards.append(float(next(iter(rew_dict.values()))))
        # Fine-grained per-tick positions for exploration overlap (Fix 4).
        positions_tick = info.get("positions_history", None)
        if positions_tick is not None:
            ep.positions_per_tick.extend(positions_tick)
        # MPC solve times: not surfaced by the env yet → zeros placeholder.
        ep.mpc_solve_times.append(np.zeros(env.cfg.num_robots, dtype=np.float32))
    ep.total_targets_found = detected_prev
    return ep


def evaluate_policy(
    policy: BasePolicy,
    env: MultiRobotSearchEnv,
    eval_cfg: dict,
) -> List[EpisodeData]:
    """Run N episodes and collect data per episode."""
    n_eps = int(eval_cfg.get("num_episodes", 10))
    seeds = list(eval_cfg.get("seeds", [0]))
    episodes: List[EpisodeData] = []
    for ep_idx in range(n_eps):
        seed = int(seeds[ep_idx % len(seeds)]) + ep_idx
        _LOG.info("episode %d / %d (seed=%d)", ep_idx + 1, n_eps, seed)
        ep = run_episode(policy, env, seed)
        episodes.append(ep)
    return episodes


def summarise(episodes: List[EpisodeData]) -> Dict[str, Dict[str, float]]:
    """mean ± std summary across episodes."""
    rows = [compute_all_metrics(ep) for ep in episodes]
    out: Dict[str, Dict[str, float]] = {}
    for key in METRIC_KEYS:
        vals = np.array([r[key] for r in rows], dtype=np.float64)
        out[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "median": float(np.median(vals)),
        }
    return out


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def save_results(
    episodes: List[EpisodeData],
    summary: Dict[str, Dict[str, float]],
    out_dir: Path,
    scenario_cfg: dict,
    eval_cfg: dict,
    save_trajectories: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Summary JSON.
    (out_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
    # Per-episode CSV.
    csv_path = out_dir / "metrics_per_episode.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", *METRIC_KEYS])
        for i, ep in enumerate(episodes):
            metrics = compute_all_metrics(ep)
            w.writerow([i, *[metrics[k] for k in METRIC_KEYS]])
    # Coverage curves (N_eps, T_max).  Episodes that terminate early (crashes,
    # energy depletion, early detection) are padded with NaN rather than the
    # last coverage value to avoid inflating aggregate statistics.  Downstream
    # plotting and analysis must use nan-aware aggregation (np.nanmean, etc.).
    curves = [coverage_over_time(ep) for ep in episodes]
    T_max = max((len(c) for c in curves), default=0)
    arr = np.full((len(curves), T_max), np.nan, dtype=np.float32)
    for i, c in enumerate(curves):
        if len(c) == 0:
            continue
        arr[i, : len(c)] = c
    np.save(out_dir / "coverage_curves.npy", arr)
    # Raw episode dumps for replotting.
    if save_trajectories:
        traj_dir = out_dir / "episode_data"
        traj_dir.mkdir(exist_ok=True)
        for i, ep in enumerate(episodes):
            np.savez_compressed(
                traj_dir / f"ep_{i:03d}.npz",
                robot_positions=np.stack(ep.robot_positions, axis=0) if ep.robot_positions else np.zeros(0),
                coverage_maps=np.stack(ep.coverage_maps, axis=0) if ep.coverage_maps else np.zeros(0),
                gp_uncertainty=np.stack(ep.gp_uncertainty, axis=0) if ep.gp_uncertainty else np.zeros(0),
                cvar_values=np.stack(ep.cvar_values, axis=0) if ep.cvar_values else np.zeros(0),
                lyapunov_values=np.stack(ep.lyapunov_values, axis=0) if ep.lyapunov_values else np.zeros(0),
                energy_consumed=np.stack(ep.energy_consumed, axis=0) if ep.energy_consumed else np.zeros(0),
                collisions=np.stack(ep.collisions, axis=0) if ep.collisions else np.zeros(0),
                detections=np.array(ep.detections, dtype=np.int32),
                rewards=np.array(ep.rewards, dtype=np.float32),
                # Fine-grained per-tick positions for trajectory plotting and
                # exploration overlap verification (Fix 4).
                positions_per_tick=(
                    np.stack(ep.positions_per_tick, axis=0)
                    if ep.positions_per_tick else np.zeros(0)
                ),
                num_robots=ep.num_robots,
                num_targets=ep.num_targets,
                max_steps=ep.max_steps,
                total_targets_found=ep.total_targets_found,
                world_size=ep.world_size,
            )
    # Reproducibility: copy configs.
    (out_dir / "scenario.yaml").write_text(yaml.safe_dump(scenario_cfg, sort_keys=False))
    (out_dir / "eval.yaml").write_text(yaml.safe_dump(eval_cfg, sort_keys=False))


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate RISE-MAPPO / baselines.")
    p.add_argument("--policy", required=True,
                   choices=["trained", *BASELINE_REGISTRY.keys()])
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    p.add_argument("--scenario", type=Path, required=True)
    p.add_argument("--eval-config", type=Path, default=Path("configs/eval_default.yaml"))
    p.add_argument("--name", type=str, default="",
                   help="Override the policy display name (used in output paths).")
    p.add_argument("--num-episodes", type=int, default=None,
                   help="Override eval_config.num_episodes.")
    return p


def main(argv: List[str] = None) -> Tuple[Path, Dict[str, Dict[str, float]]]:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s | %(message)s")
    base_cfg = _load_yaml(args.config)
    scenario_cfg = _load_yaml(args.scenario)
    eval_cfg = _load_yaml(args.eval_config).get("eval", {})
    if args.num_episodes is not None:
        eval_cfg["num_episodes"] = int(args.num_episodes)
    merged = _merge(base_cfg, scenario_cfg)
    # Merge eval-level env overrides (Phase-2 toggles: use_real_gp, use_lyap_mpc,
    # gp_update_interval, gp_obs_noise_std, etc.).  Without this step the eval
    # config's ``env`` section is silently discarded and all evaluations run with
    # the proportional controller + Phase-1 decay GP, producing meaningless CVaR
    # and Lyapunov metrics.
    eval_env_overrides = _load_yaml(args.eval_config).get("env", {})
    if eval_env_overrides:
        merged = _merge(merged, {"env": eval_env_overrides})
    env_cfg = _build_env_cfg(merged, seed=int(eval_cfg.get("seeds", [0])[0]))
    env = MultiRobotSearchEnv(env_cfg)
    policy = build_policy(
        args.policy,
        checkpoint=args.checkpoint,
        config=args.config,
        name=args.name,
        deterministic=bool(eval_cfg.get("deterministic", True)),
        device=str(eval_cfg.get("device", "cpu")),
        num_actions=env.num_subgoal_actions,
        num_robots=env.cfg.num_robots,
    )
    episodes = evaluate_policy(policy, env, eval_cfg)
    summary = summarise(episodes)
    scen_name = str(scenario_cfg.get("scenario", {}).get("name", args.scenario.stem))
    out_dir = Path(eval_cfg.get("results_dir", "results/eval")) / scen_name / _safe_name(policy.name)
    save_results(
        episodes,
        summary,
        out_dir,
        scenario_cfg=scenario_cfg,
        eval_cfg=eval_cfg,
        save_trajectories=bool(eval_cfg.get("save_trajectories", True)),
    )
    _LOG.info("wrote results to %s", out_dir)
    _LOG.info("summary: %s", {k: v["mean"] for k, v in summary.items()})
    return out_dir, summary


if __name__ == "__main__":
    main()
