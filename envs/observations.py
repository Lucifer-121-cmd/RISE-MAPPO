"""Observation builders for :class:`envs.multi_robot_search_env.MultiRobotSearchEnv`.

These helpers are factored out so the env file stays under the project's
400-line cap. They depend only on the env's exposed attributes
(``world``, ``gp``, ``_states``, ``_merged_occupancy``, ``cfg``,
``_coverage_mask``, ``_detected_count``).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from envs.multi_robot_search_env import MultiRobotSearchEnv


def ego_occupancy_patch(env: "MultiRobotSearchEnv", pose: np.ndarray, patch_size: int) -> np.ndarray:
    """Return an ego-centric ``(patch_size, patch_size)`` occupancy patch."""
    cell = env.cfg.patch_cell_size or env.world.cfg.resolution
    half = patch_size // 2
    g = env.world.grid_size
    res = env.world.cfg.resolution
    offsets = (np.arange(patch_size) - half) * cell
    gx = pose[0] + offsets[None, :]
    gy = pose[1] + offsets[:, None]
    valid = (gx >= 0) & (gx <= env.cfg.world_size) & (gy >= 0) & (gy <= env.cfg.world_size)
    ix = np.clip((gx / res).astype(int), 0, g - 1)
    iy = np.clip((gy / res).astype(int), 0, g - 1)
    return np.where(valid, env._merged_occupancy[iy, ix], 1.0).astype(np.float32)


def build_observations(env: "MultiRobotSearchEnv", patch_size: int) -> Dict[str, Dict[str, np.ndarray]]:
    """Return ``{agent: obs_dict}`` for the env's current state."""
    n = env.cfg.num_robots
    coverage = float(env._coverage_mask.mean())
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for idx, agent in enumerate(env.agents):
        st = env._states[agent]
        occ_patch = ego_occupancy_patch(env, st.pose, patch_size)
        gp_patch = env.gp.uncertainty_patch(
            st.pose[:2], patch_size=patch_size, cell_size=env.cfg.patch_cell_size,
        )
        others = [env._states[a].pose[:2] - st.pose[:2] for a in env.agents if a != agent]
        rel = np.zeros((1, 2), dtype=np.float32) if not others else np.asarray(others, dtype=np.float32)
        agent_id = np.zeros(n, dtype=np.float32)
        agent_id[idx] = 1.0
        out[agent] = {
            "occupancy_patch": occ_patch.astype(np.float32),
            "gp_sigma_patch": gp_patch.astype(np.float32),
            "rel_robots": rel,
            "energy": np.array([st.energy / max(1e-6, env.cfg.energy_budget)], dtype=np.float32),
            "coverage": np.array([coverage], dtype=np.float32),
            "agent_id": agent_id,
        }
    return out


def build_global_state(env: "MultiRobotSearchEnv") -> Dict[str, np.ndarray]:
    """Return the centralised-critic state dict."""
    n = env.cfg.num_robots
    states_arr = np.zeros((n, 3), dtype=np.float32)
    energies = np.zeros(n, dtype=np.float32)
    for i, a in enumerate(env.agents):
        st = env._states[a]
        states_arr[i] = st.pose
        energies[i] = st.energy / max(1e-6, env.cfg.energy_budget)
    det_frac = env._detected_count() / max(1, env.cfg.num_targets)
    coverage = float(env._coverage_mask.mean())
    return {
        "occupancy": env._merged_occupancy.astype(np.float32),
        "gp_sigma": env.gp.uncertainty_grid().astype(np.float32),
        "robot_states": states_arr,
        "energies": energies,
        "stats": np.array([coverage, det_frac], dtype=np.float32),
    }
