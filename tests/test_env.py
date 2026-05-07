"""End-to-end tests for the multi-robot search environment."""
from __future__ import annotations

import numpy as np
import pytest

from envs.multi_robot_search_env import EnvConfig, MultiRobotSearchEnv


@pytest.fixture
def small_env() -> MultiRobotSearchEnv:
    return MultiRobotSearchEnv(EnvConfig(
        num_robots=3,
        world_size=10.0,
        max_steps=30,
        num_obstacles=6,
        num_targets=4,
        num_hazards=2,
        subgoal_steps=8,
        seed=0,
    ))


def test_reset_observation_shapes(small_env: MultiRobotSearchEnv) -> None:
    obs, info = small_env.reset()
    expected = small_env.observation_shapes()
    assert set(obs.keys()) == set(small_env.agents)
    for a in small_env.agents:
        for k, shape in expected.items():
            assert obs[a][k].shape == shape, (a, k, obs[a][k].shape, shape)
    assert "coverage" in info


def test_global_state_shapes(small_env: MultiRobotSearchEnv) -> None:
    small_env.reset()
    gs = small_env.global_state()
    expected = small_env.global_state_shapes()
    for k, shape in expected.items():
        assert gs[k].shape == shape


def test_random_policy_runs(small_env: MultiRobotSearchEnv) -> None:
    obs, _ = small_env.reset()
    rng = np.random.default_rng(1)
    coverages: list[float] = []
    for _ in range(20):
        actions = {a: int(rng.integers(small_env.num_subgoal_actions)) for a in small_env.agents}
        obs, rewards, terms, truncs, infos = small_env.step(actions)
        coverages.append(infos["coverage"])
        assert set(rewards.keys()) == set(small_env.agents)
        if all(terms.values()) or all(truncs.values()):
            break
    # Coverage must be non-decreasing in our env.
    cov_arr = np.asarray(coverages)
    assert np.all(np.diff(cov_arr) >= -1e-9)
    assert coverages[-1] >= coverages[0]


def test_action_validation(small_env: MultiRobotSearchEnv) -> None:
    small_env.reset()
    with pytest.raises(ValueError):
        small_env.step({"robot_0": 0})  # missing other agents


def test_invalid_action_index_raises(small_env: MultiRobotSearchEnv) -> None:
    small_env.reset()
    bad = {a: 0 for a in small_env.agents}
    bad["robot_0"] = 999
    with pytest.raises(ValueError):
        small_env.step(bad)


def test_render_returns_axis(small_env: MultiRobotSearchEnv) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    small_env.reset()
    fig, ax = plt.subplots()
    out = small_env.render(ax=ax)
    assert out is ax
    plt.close(fig)


def test_reward_keys_match_agents(small_env: MultiRobotSearchEnv) -> None:
    small_env.reset()
    actions = {a: 12 for a in small_env.agents}  # centre cell
    _, rewards, _, _, _ = small_env.step(actions)
    assert set(rewards.keys()) == set(small_env.agents)
    # Team-shared reward.
    vals = list(rewards.values())
    assert all(np.isclose(v, vals[0]) for v in vals)


def test_episode_termination_on_max_steps() -> None:
    env = MultiRobotSearchEnv(EnvConfig(num_robots=2, max_steps=3, subgoal_steps=2, seed=1))
    env.reset()
    actions = {a: 12 for a in env.agents}
    for _ in range(env.cfg.max_steps):
        _, _, _, truncs, _ = env.step(actions)
    assert all(truncs.values())
