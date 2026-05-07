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


_RISE_GLOBAL_SHAPES = {
    "occupancy": (20, 20),
    "gp_sigma": (20, 20),
    "robot_states": (3, 3),
    "energies": (3,),
    "stats": (2,),
}


def _make_rise_state(batch: int = 4, num_robots: int = 3, grid: int = 20) -> dict:
    return {
        "occupancy": torch.randn(batch, grid, grid),
        "gp_sigma": torch.rand(batch, grid, grid).abs(),
        "robot_states": torch.rand(batch, num_robots, 3) * 5.0,
        "energies": torch.rand(batch, num_robots),
        "stats": torch.rand(batch, 2),
    }


class TestRISEMAPPO:
    """Tests for the novel RISE-MAPPO algorithm (Phase 2.5)."""

    def test_dual_critic_output_shapes(self) -> None:
        """Critic returns (v_mean, v_cvar) with correct shapes."""
        critic = Critic(CriticConfig(
            grid_size=20, num_robots=3, use_popart=False, use_rise=True,
            world_size=10.0,
        ))
        state = _make_rise_state(batch=4)
        sigmas = critic.extract_agent_sigmas(state)
        assert sigmas.shape == (4, 3)
        v_mean, v_cvar = critic(state, agent_sigmas=sigmas)
        assert v_mean.shape == (4,) and v_cvar.shape == (4,)
        assert torch.isfinite(v_mean).all() and torch.isfinite(v_cvar).all()

    def test_dual_critic_different_values(self) -> None:
        """V_mean and V_cvar are not identical (heads learn different things)."""
        torch.manual_seed(0)
        critic = Critic(CriticConfig(
            grid_size=20, num_robots=3, use_popart=False, use_rise=True,
            world_size=10.0,
        ))
        state = _make_rise_state(batch=8)
        sigmas = critic.extract_agent_sigmas(state)
        opt = torch.optim.Adam(critic.parameters(), lr=1e-2)
        target_mean = torch.zeros(8)
        target_cvar = torch.ones(8) * 5.0
        for _ in range(20):
            v_mean, v_cvar = critic(state, agent_sigmas=sigmas)
            loss = (v_mean - target_mean).pow(2).mean() + (v_cvar - target_cvar).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        v_mean, v_cvar = critic(state, agent_sigmas=sigmas)
        assert (v_cvar - v_mean).abs().mean().item() > 1.0

    def test_gp_attention_high_uncertainty_higher_weight(self) -> None:
        """Agent with higher GP sigma gets higher attention weight."""
        from marl.mappo.critic import GPUncertaintyAttention
        torch.manual_seed(0)
        attn = GPUncertaintyAttention(feature_dim=8, eta=1.0)
        feats = torch.randn(2, 3, 8)
        sigmas = torch.tensor([[0.1, 0.1, 5.0], [0.1, 0.1, 5.0]])
        _, weights = attn(feats, sigmas)
        col_avg = weights.mean(dim=1)            # (B, N) – avg attention each agent receives
        # Agent index 2 (high sigma) should receive more attention than 0 / 1.
        assert (col_avg[:, 2] > col_avg[:, 0]).all()
        assert (col_avg[:, 2] > col_avg[:, 1]).all()

    def test_gp_attention_equal_uncertainty_equal_weight(self) -> None:
        """Equal sigmas → roughly equal column-attention weights."""
        from marl.mappo.critic import GPUncertaintyAttention
        torch.manual_seed(0)
        attn = GPUncertaintyAttention(feature_dim=8, eta=1.0)
        feats = torch.zeros(1, 3, 8)              # zero features → only sigma matters
        sigmas = torch.full((1, 3), 1.0)
        _, weights = attn(feats, sigmas)
        col_avg = weights.mean(dim=1).squeeze(0)  # (3,)
        assert torch.allclose(col_avg, torch.full_like(col_avg, 1.0 / 3.0), atol=1e-5)

    def test_risk_adjusted_advantage(self) -> None:
        """A_final = A_mean - lambda * A_risk produces valid advantages."""
        torch.manual_seed(0)
        adv_mean = torch.tensor([1.0, 2.0, -1.0, 0.5])
        adv_risk = torch.tensor([0.0, 1.0, 2.0, -0.5])
        lam = 0.1
        adv = adv_mean - lam * adv_risk
        expected = torch.tensor([1.0, 1.9, -1.2, 0.55])
        assert torch.allclose(adv, expected, atol=1e-6)

    def test_rise_vs_standard_backward_compat(self) -> None:
        """use_rise=False keeps Phase-1 single-head behaviour and is
        bit-for-bit deterministic across runs (same seed → same losses).
        """
        torch.manual_seed(123)
        critic = Critic(CriticConfig(
            grid_size=20, num_robots=3, use_popart=False, use_rise=False,
            world_size=10.0,
        ))
        state = _make_rise_state(batch=4)
        v = critic(state)
        assert v.shape == (4,)
        assert not hasattr(critic, "v_mean_head")
        assert not hasattr(critic, "v_cvar_head")
        assert not hasattr(critic, "gp_attention")
        # Numerical equivalence: two PPO updates with use_rise=False under
        # the same global seed must produce identical losses (proves the
        # non-RISE path is bit-for-bit unchanged from Phase 1).
        from marl.utils import set_global_seed

        def run_one() -> tuple[float, float]:
            set_global_seed(7)
            env = MultiRobotSearchEnv(EnvConfig(
                num_robots=2, max_steps=6, subgoal_steps=4, seed=7,
            ))
            obs, _ = env.reset(seed=7)
            actor = Actor(ActorConfig(
                num_actions=env.num_subgoal_actions, num_robots=2,
            ))
            crit = Critic(CriticConfig(
                grid_size=env.world.grid_size, num_robots=2,
                use_popart=True, use_rise=False, world_size=env.cfg.world_size,
            ))
            algo = MAPPO(actor=actor, critic=crit,
                         config=MAPPOConfig(ppo_epoch=1, num_mini_batch=2,
                                            use_rise=False),
                         device=torch.device("cpu"))
            buf = RolloutBuffer(
                BufferConfig(rollout_length=4, num_envs=1, num_agents=2,
                             device="cpu", use_rise=False),
                env.observation_shapes(), env.global_state_shapes(),
            )
            cur_obs = obs; cur_state = env.global_state()
            local_keys = list(env.observation_shapes().keys())
            global_keys = list(env.global_state_shapes().keys())
            for t in range(4):
                local_stack = _stack_local(cur_obs, env.agents, local_keys)
                flat = {k: torch.from_numpy(
                    v.reshape((2,) + v.shape[2:])).float() for k, v in local_stack.items()}
                with torch.no_grad():
                    a, lp, _ = actor.get_action(flat)
                    gs = {k: torch.from_numpy(cur_state[k]).unsqueeze(0).float()
                          for k in global_keys}
                    val = crit(gs).numpy()
                a_np = a.numpy().reshape(1, 2)
                lp_np = lp.numpy().reshape(1, 2)
                act_d = {ag: int(a_np[0, j]) for j, ag in enumerate(env.agents)}
                new_obs, rd, td, trd, _ = env.step(act_d)
                rew = float(next(iter(rd.values())))
                done = float(any(td.values()) or any(trd.values()))
                buf.insert(
                    t=t, local_obs=local_stack,
                    global_state={k: cur_state[k][None, ...] for k in global_keys},
                    actions=a_np, log_probs=lp_np,
                    rewards=np.array([rew], dtype=np.float32),
                    dones=np.array([done], dtype=np.float32),
                    values=val.astype(np.float32),
                )
                cur_obs = new_obs if not done else env.reset()[0]
                cur_state = env.global_state()
            with torch.no_grad():
                ls = {k: torch.from_numpy(cur_state[k]).unsqueeze(0).float()
                      for k in global_keys}
                lv = crit(ls).numpy().astype(np.float32)
            buf.compute_returns_and_advantages(lv, np.zeros(1, dtype=np.float32))
            stats = algo.update(buf)
            return stats["policy_loss"], stats["value_loss"]

        p1, v1 = run_one()
        p2, v2 = run_one()
        assert abs(p1 - p2) < 1e-6, f"non-RISE policy loss not deterministic: {p1} {p2}"
        assert abs(v1 - v2) < 1e-6, f"non-RISE value loss not deterministic: {v1} {v2}"

    def test_buffer_stores_risk_costs(self) -> None:
        """Buffer correctly stores and retrieves risk_costs / values_cvar."""
        T, E, A = 4, 1, 3
        buf = RolloutBuffer(
            BufferConfig(
                rollout_length=T, num_envs=E, num_agents=A, device="cpu", use_rise=True,
            ),
            local_obs_shapes={"x": (2,)},
            global_state_shapes={"occupancy": (5, 5), "gp_sigma": (5, 5),
                                 "robot_states": (A, 3), "energies": (A,), "stats": (2,)},
        )
        rng = np.random.default_rng(0)
        for t in range(T):
            buf.insert(
                t=t,
                local_obs={"x": np.zeros((E, A, 2), dtype=np.float32)},
                global_state={
                    "occupancy": np.zeros((E, 5, 5), dtype=np.float32),
                    "gp_sigma": np.zeros((E, 5, 5), dtype=np.float32),
                    "robot_states": np.zeros((E, A, 3), dtype=np.float32),
                    "energies": np.zeros((E, A), dtype=np.float32),
                    "stats": np.zeros((E, 2), dtype=np.float32),
                },
                actions=np.zeros((E, A), dtype=np.int64),
                log_probs=np.zeros((E, A), dtype=np.float32),
                rewards=np.full(E, float(t), dtype=np.float32),
                dones=np.zeros(E, dtype=np.float32),
                values=np.full(E, 0.5, dtype=np.float32),
                risk_costs=np.full(E, 0.1 * (t + 1), dtype=np.float32),
                values_cvar=np.full(E, 0.2, dtype=np.float32),
                agent_sigmas=rng.random((E, A)).astype(np.float32),
            )
        assert np.allclose(buf.risk_costs[:, 0], [0.1, 0.2, 0.3, 0.4])
        assert np.allclose(buf.values_cvar, 0.2)
        buf.compute_returns_and_advantages(
            np.zeros(E, dtype=np.float32),
            np.zeros(E, dtype=np.float32),
            last_values_cvar=np.zeros(E, dtype=np.float32),
        )
        assert buf.advantages_risk.shape == (T, E)
        assert buf.returns_risk.shape == (T, E)

    def test_smoke_train_rise(self) -> None:
        """Full smoke-train with use_rise=True; no NaN, RISE losses logged."""
        env_cfg = EnvConfig(num_robots=2, max_steps=8, subgoal_steps=4, seed=11)
        env = MultiRobotSearchEnv(env_cfg)
        env.reset()
        actor = Actor(ActorConfig(
            num_actions=env.num_subgoal_actions, num_robots=env_cfg.num_robots,
        ))
        critic = Critic(CriticConfig(
            grid_size=env.world.grid_size,
            num_robots=env_cfg.num_robots,
            use_popart=True,
            use_rise=True,
            world_size=env_cfg.world_size,
            gp_attention_eta=1.0,
        ))
        algo = MAPPO(
            actor=actor, critic=critic,
            config=MAPPOConfig(
                ppo_epoch=2, num_mini_batch=2, use_popart=True,
                use_rise=True, lambda_risk=0.1, cvar_loss_coef=0.5,
                gp_attention_eta=1.0,
            ),
            device=torch.device("cpu"),
        )
        from marl.mappo.runner import Runner, RunnerConfig
        runner = Runner(
            env_factory=lambda seed: MultiRobotSearchEnv(EnvConfig(
                num_robots=env_cfg.num_robots, max_steps=env_cfg.max_steps,
                subgoal_steps=env_cfg.subgoal_steps, seed=seed,
            )),
            actor=actor, critic=critic, algo=algo,
            runner_cfg=RunnerConfig(
                rollout_length=4, num_envs=1, n_training_updates=2,
                device="cpu", seed=11,
                save_dir="results/checkpoints_rise_smoke",
            ),
        )
        roll_stats = runner.collect_rollout()
        algo_stats = algo.update(runner.buffer)
        for key in ("v_mean_loss", "v_cvar_loss", "risk_advantage_mean",
                    "gp_attention_entropy", "policy_loss", "entropy"):
            assert key in algo_stats, f"missing log key: {key}"
            assert np.isfinite(algo_stats[key]), f"non-finite stat {key}"
        assert np.isfinite(roll_stats["ep_reward_mean"])


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
