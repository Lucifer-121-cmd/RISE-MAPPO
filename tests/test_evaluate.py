"""End-to-end tests for the evaluation pipeline."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analysis.metrics import METRIC_KEYS  # noqa: E402
from scripts.evaluate import main as eval_main  # noqa: E402


def _write_eval_cfg(tmp_path: Path, episodes: int = 2, results_dir: Path = None) -> Path:
    cfg = tmp_path / "eval.yaml"
    rd = results_dir if results_dir is not None else (tmp_path / "out")
    cfg.write_text(
        "eval:\n"
        f"  num_episodes: {episodes}\n"
        "  deterministic: true\n"
        "  device: \"cpu\"\n"
        "  save_trajectories: true\n"
        f"  results_dir: \"{rd}\"\n"
        "  seeds: [42, 43]\n"
    )
    return cfg


def _write_simple_scenario(tmp_path: Path, max_steps: int = 5, num_robots: int = 2) -> Path:
    p = tmp_path / "scenario.yaml"
    p.write_text(
        "scenario:\n"
        "  name: \"smoke\"\n"
        "env:\n"
        f"  num_robots: {num_robots}\n"
        "  world_size: 4.0\n"
        f"  max_steps: {max_steps}\n"
        "  num_targets: 2\n"
        "  num_obstacles: 1\n"
        "  num_hazards: 0\n"
        "  subgoal_steps: 4\n"
    )
    return p


def test_evaluate_random_policy_creates_output(tmp_path: Path):
    out_root = tmp_path / "out"
    eval_cfg = _write_eval_cfg(tmp_path, episodes=2, results_dir=out_root)
    scen = _write_simple_scenario(tmp_path, max_steps=5, num_robots=2)
    out_dir, summary = eval_main([
        "--policy", "random",
        "--scenario", str(scen),
        "--eval-config", str(eval_cfg),
        "--config", str(_REPO_ROOT / "configs" / "default.yaml"),
    ])
    assert Path(out_dir).exists()
    assert (out_dir / "metrics_summary.json").exists()
    parsed = json.loads((out_dir / "metrics_summary.json").read_text())
    for key in METRIC_KEYS:
        assert key in parsed
        assert "mean" in parsed[key]
        assert np.isfinite(parsed[key]["mean"])
    # CSV one row per episode + header.
    rows = list(csv.reader((out_dir / "metrics_per_episode.csv").open()))
    assert len(rows) == 3  # header + 2 episodes
    # Coverage curves saved.
    curves = np.load(out_dir / "coverage_curves.npy")
    assert curves.shape[0] == 2
    # Trajectory dumps exist.
    traj = sorted((out_dir / "episode_data").glob("ep_*.npz"))
    assert len(traj) == 2


def test_evaluate_nearest_frontier_runs(tmp_path: Path):
    out_root = tmp_path / "out"
    eval_cfg = _write_eval_cfg(tmp_path, episodes=2, results_dir=out_root)
    scen = _write_simple_scenario(tmp_path, max_steps=5, num_robots=2)
    out_dir, summary = eval_main([
        "--policy", "nearest_frontier",
        "--scenario", str(scen),
        "--eval-config", str(eval_cfg),
        "--config", str(_REPO_ROOT / "configs" / "default.yaml"),
    ])
    assert (out_dir / "metrics_summary.json").exists()
    assert summary["coverage_rate"]["mean"] >= 0.0


def test_evaluate_trained_loads_checkpoint(tmp_path: Path):
    ckpt = _REPO_ROOT / "results" / "checkpoints" / "mappo_upd50.pt"
    if not ckpt.exists():
        pytest.skip("no checkpoint available")
    out_root = tmp_path / "out"
    eval_cfg = _write_eval_cfg(tmp_path, episodes=1, results_dir=out_root)
    # Trained policy expects num_robots matching the training config (3).
    scen = _write_simple_scenario(tmp_path, max_steps=3, num_robots=3)
    out_dir, summary = eval_main([
        "--policy", "trained",
        "--checkpoint", str(ckpt),
        "--scenario", str(scen),
        "--eval-config", str(eval_cfg),
        "--config", str(_REPO_ROOT / "configs" / "default.yaml"),
    ])
    assert (out_dir / "metrics_summary.json").exists()
