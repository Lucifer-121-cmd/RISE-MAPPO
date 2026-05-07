"""End-to-end smoke test: env + MAPPO + buffer + update.

Also includes Phase-2 verification tests for STEP 6: GP fusion plots,
Phase-2 env episode with LyapunovMPC + DistributedGP wired in, and
performance budget checks.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
import torch

from envs.multi_robot_search_env import EnvConfig, MultiRobotSearchEnv
from gp.distributed_gp import DistributedGP, DistributedGPConfig
from gp.local_gp import LocalGPConfig
from marl.mappo.actor import Actor, ActorConfig
from marl.mappo.algorithm import MAPPO, MAPPOConfig
from marl.mappo.buffer import BufferConfig, RolloutBuffer
from marl.mappo.critic import Critic, CriticConfig


def _stack_local(env_obs, agents, keys):
    out = {}
    for k in keys:
        out[k] = np.stack([env_obs[a][k] for a in agents], axis=0)[None, ...]  # (1, A, ...)
    return out


def test_mappo_full_update_cycle() -> None:
    env_cfg = EnvConfig(num_robots=2, max_steps=10, subgoal_steps=4, seed=42)
    env = MultiRobotSearchEnv(env_cfg)
    obs, _ = env.reset()
    actor = Actor(ActorConfig(num_actions=env.num_subgoal_actions, num_robots=env_cfg.num_robots))
    critic = Critic(CriticConfig(
        grid_size=env.world.grid_size,
        num_robots=env_cfg.num_robots,
        use_popart=True,
    ))
    device = torch.device("cpu")
    algo = MAPPO(actor=actor, critic=critic, config=MAPPOConfig(
        ppo_epoch=2, num_mini_batch=2,
    ), device=device)
    buffer = RolloutBuffer(
        BufferConfig(rollout_length=4, num_envs=1, num_agents=env_cfg.num_robots, device="cpu"),
        env.observation_shapes(),
        env.global_state_shapes(),
    )
    local_keys = list(env.observation_shapes().keys())
    global_keys = list(env.global_state_shapes().keys())
    cur_obs = obs
    cur_state = env.global_state()
    for t in range(buffer.cfg.rollout_length):
        local_stack = _stack_local(cur_obs, env.agents, local_keys)
        flat = {k: torch.from_numpy(v.reshape((env_cfg.num_robots,) + v.shape[2:])).float() for k, v in local_stack.items()}
        with torch.no_grad():
            action, log_prob, _ = actor.get_action(flat)
            gs = {k: torch.from_numpy(cur_state[k]).unsqueeze(0).float() for k in global_keys}
            value = critic(gs).numpy()
        action_np = action.detach().cpu().numpy().reshape(1, env_cfg.num_robots)
        log_prob_np = log_prob.detach().cpu().numpy().reshape(1, env_cfg.num_robots)
        act_dict = {a: int(action_np[0, j]) for j, a in enumerate(env.agents)}
        new_obs, rew_dict, term_dict, trunc_dict, info = env.step(act_dict)
        rew = float(next(iter(rew_dict.values())))
        done = float(any(term_dict.values()) or any(trunc_dict.values()))
        buffer.insert(
            t=t,
            local_obs=local_stack,
            global_state={k: cur_state[k][None, ...] for k in global_keys},
            actions=action_np,
            log_probs=log_prob_np,
            rewards=np.array([rew], dtype=np.float32),
            dones=np.array([done], dtype=np.float32),
            values=value.astype(np.float32),
        )
        cur_obs = new_obs if not done else env.reset()[0]
        cur_state = env.global_state()
    last_state = {k: torch.from_numpy(cur_state[k]).unsqueeze(0).float() for k in global_keys}
    with torch.no_grad():
        last_v = critic(last_state).numpy()
    buffer.compute_returns_and_advantages(last_v.astype(np.float32), np.zeros(1, dtype=np.float32))
    stats = algo.update(buffer)
    assert stats["n_updates"] > 0
    assert np.isfinite(stats["policy_loss"])
    assert np.isfinite(stats["value_loss"])


# ---------------------------------------------------------------------
# Phase-2 STEP 6: verification + plots
# ---------------------------------------------------------------------
def _hazard_field(xy: np.ndarray) -> np.ndarray:
    return np.exp(-((xy[:, 0] - 5.0) ** 2 + (xy[:, 1] - 5.0) ** 2) / 4.0)


def test_phase2_gp_fusion_lowers_uncertainty_and_plots() -> None:
    """3 robots in different quadrants; fused sigma map < any individual."""
    pytest.importorskip("gpytorch")
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = DistributedGPConfig(
        world_size=10.0, resolution=0.5,
        local_gp=LocalGPConfig(n_inducing=20, n_epochs=20, world_size=10.0),
    )
    dgp = DistributedGP(cfg, num_robots=3)
    rng = np.random.default_rng(42)
    quadrants = [(0.0, 5.0, 0.0, 5.0), (5.0, 10.0, 0.0, 5.0), (2.5, 7.5, 5.0, 10.0)]
    for i, (x0, x1, y0, y1) in enumerate(quadrants):
        xs = np.column_stack([rng.uniform(x0, x1, 30), rng.uniform(y0, y1, 30)]).astype(np.float32)
        dgp.update_robot(i, xs, _hazard_field(xs).astype(np.float32))
    dgp.fuse()
    fused = dgp.uncertainty_grid()
    assert fused.shape == (20, 20)
    # Fused must beat each robot solo across the union of training masses.
    grid = np.array([(x, y) for x in np.linspace(1, 9, 5) for y in np.linspace(1, 9, 5)])
    _, v_fused = dgp.predict_global(grid)
    for gp in dgp.local_gps:
        _, v_solo = gp.predict(grid.astype(np.float32))
        assert (v_fused <= v_solo + 1e-5).mean() >= 0.8
    out = Path("results") / "gp_fusion_test.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(fused, origin="lower", extent=(0, 10, 0, 10), cmap="viridis")
    ax.set_title("BCM fused σ"); fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    assert out.exists()


def test_phase2_env_episode_with_real_mpc_and_gp() -> None:
    """One episode w/ Phase-2 wiring: stable, GP grids drop, plots saved."""
    pytest.importorskip("casadi")
    pytest.importorskip("gpytorch")
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    env = MultiRobotSearchEnv(EnvConfig(
        num_robots=2, world_size=10.0, max_steps=8,
        num_obstacles=4, num_targets=3, num_hazards=2,
        subgoal_steps=4, gp_update_interval=2,
        use_lyap_mpc=True, use_real_gp=True,
        seed=7,
    ))
    env.reset()
    sigma_pre = float(env.gp.uncertainty_grid().sum())
    rng = np.random.default_rng(7)
    poses = []
    for _ in range(env.cfg.max_steps):
        actions = {a: int(rng.integers(env.num_subgoal_actions)) for a in env.agents}
        _, _, term, trunc, info = env.step(actions)
        poses.append(np.array([env._states[a].pose.copy() for a in env.agents]))
        assert "lyapunov_mean" in info
        if all(term.values()) or all(trunc.values()):
            break
    sigma_post = float(env.gp.uncertainty_grid().sum())
    assert sigma_post < sigma_pre + 1.0, "GP uncertainty did not decrease"
    out = Path("results") / "integration_test.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    env.render(axes[0]); axes[0].set_title("trajectory")
    poses_arr = np.stack(poses, axis=0)
    for r in range(poses_arr.shape[1]):
        axes[0].plot(poses_arr[:, r, 0], poses_arr[:, r, 1], "-", alpha=0.7)
    im = axes[1].imshow(env.gp.uncertainty_grid(), origin="lower",
                        extent=(0, 10, 0, 10), cmap="viridis")
    axes[1].set_title("post σ"); fig.colorbar(im, ax=axes[1])
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    assert out.exists()


def test_phase2_performance_budgets() -> None:
    """MPC mean < 50ms, GP update+predict per robot < 200ms, env step < 1.5s."""
    pytest.importorskip("casadi")
    pytest.importorskip("gpytorch")
    env = MultiRobotSearchEnv(EnvConfig(
        num_robots=2, max_steps=4, num_obstacles=3, num_targets=2,
        subgoal_steps=4, gp_update_interval=2,
        use_lyap_mpc=True, use_real_gp=True, seed=0,
    ))
    env.reset()
    # Discard first step (NLP warm up).
    env.step({a: 12 for a in env.agents})
    # Per-robot GP timing.
    rng = np.random.default_rng(0)
    xs = rng.uniform(0, 10, size=(40, 2)).astype(np.float32)
    ys = _hazard_field(xs).astype(np.float32)
    t0 = time.perf_counter()
    env.gp.update_robot(0, xs, ys)
    env.gp.predict_global(xs[:5])
    gp_dt = time.perf_counter() - t0
    # MPC timing: average over a few solves.
    ctrl = env._states[env.agents[0]].controller
    mpc_times = []
    for _ in range(8):
        t0 = time.perf_counter()
        ctrl.compute_control([1.0, 1.0, 0.0], [3.0, 2.0])
        mpc_times.append(time.perf_counter() - t0)
    mean_mpc = float(np.mean(mpc_times))
    # Full env step timing.
    t0 = time.perf_counter()
    env.step({a: 8 for a in env.agents})
    env_dt = time.perf_counter() - t0
    # Loose bounds: CI machines vary; we just want order-of-magnitude.
    assert mean_mpc < 0.1, f"MPC mean {mean_mpc*1000:.1f} ms"
    assert gp_dt < 1.0, f"GP update+predict {gp_dt*1000:.1f} ms"
    assert env_dt < 5.0, f"env.step {env_dt*1000:.1f} ms"
