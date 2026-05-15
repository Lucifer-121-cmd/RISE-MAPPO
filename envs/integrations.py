"""Phase-2 controller / GP wiring helpers for the search env.

Factored out of :mod:`envs.multi_robot_search_env` to keep the env
file under the project's 400-line cap. Each helper takes the env
instance explicitly so the env class stays free of Phase-2 noise.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

from envs.grid_world import CircleObstacle
from envs.robot_dynamics import TurtleBot3Params
from mpc.lyapunov_mpc import LyapunovMPC, LyapunovMPCConfig
from mpc.utils import proportional_subgoal_controller

if TYPE_CHECKING:  # pragma: no cover
    from envs.multi_robot_search_env import MultiRobotSearchEnv


def make_controller(env: "MultiRobotSearchEnv", mpc_cfg: dict = None):
    """Return per-robot controller. NLP if ``cfg.use_lyap_mpc`` else proportional.

    All MPC parameters are read from ``mpc_cfg`` (typically sourced from the
    ``mpc`` section of the YAML config).  When ``mpc_cfg`` is ``None`` or
    missing a key, the :class:`LyapunovMPCConfig` dataclass default is used.
    No parameter is hardcoded in this function.
    """
    if env.cfg.use_lyap_mpc:
        cfg = mpc_cfg or {}
        return LyapunovMPC(
            config=LyapunovMPCConfig(
                dt=env.cfg.dt,
                horizon=int(cfg.get("horizon", 12)),
                max_obstacles=int(cfg.get("max_obstacles", 8)),
                soft_lyap_penalty=float(cfg.get("soft_lyap_penalty", 100.0)),
                alpha_lyap=float(cfg.get("alpha_lyap", 0.05)),
                R_diag=cfg.get("R_diag", [0.1, 0.1]),
                Q_diag=cfg.get("Q_diag", [12.0, 12.0, 0.0]),
                S_du_diag=cfg.get("S_du_diag", [1.0, 0.5]),
                P_terminal_scale=float(cfg.get("P_terminal_scale", 15.0)),
                max_iter=int(cfg.get("max_iter", 60)),
                max_cpu_time=float(cfg.get("max_cpu_time", 0.05)),
                d_safe=float(cfg.get("d_safe", 0.5)),
                goal_tolerance=float(cfg.get("goal_tolerance", 0.1)),
                w_energy=float(cfg.get("w_energy", 0.1)),
            ),
            robot_params=TurtleBot3Params(),
        )

    max_v = TurtleBot3Params().max_linear_velocity
    max_omega = TurtleBot3Params().max_angular_velocity

    class _PWrap:
        """Adapter exposing the LyapunovMPC API around a P controller."""

        def step(self_inner, state, goal):
            return proportional_subgoal_controller(
                state=state, goal=goal, max_v=max_v, max_omega=max_omega,
            )

        def compute_control(self_inner, state, goal, obstacles=None, obstacle_margins=None):
            _v, _w, fb = self_inner.step(state, goal)
            return fb

        def reset(self_inner):
            return None

    return _PWrap()


def nearby_obstacle_centres(env: "MultiRobotSearchEnv") -> Optional[np.ndarray]:
    """Return all obstacle (cx, cy) for the MPC obstacle param block."""
    out: List[Tuple[float, float]] = []
    for obs in env.world.obstacles:
        if isinstance(obs, CircleObstacle):
            out.append((obs.cx, obs.cy))
        else:
            out.append((0.5 * (obs.x0 + obs.x1), 0.5 * (obs.y0 + obs.y1)))
    if not out:
        return None
    return np.asarray(out, dtype=float)


def obstacle_margins(env: "MultiRobotSearchEnv", obstacles: Optional[np.ndarray]):
    """CVaR-augmented obstacle margin per slot. Zeros when GP is unfit."""
    if obstacles is None:
        return None
    if not env.cfg.use_real_gp:
        return np.zeros(obstacles.shape[0], dtype=float)
    cvar = env.gp.cvar_risk_at(obstacles, alpha=env.cfg.cvar_alpha)
    return env.cfg.obstacle_margin_scale * np.asarray(cvar, dtype=float)


def flush_gp_buffers(env: "MultiRobotSearchEnv") -> None:
    """Drain each robot's hazard-observation buffer into its local GP."""
    any_data = False
    for i, a in enumerate(env.agents):
        buf = env._states[a].obs_buffer
        if not buf:
            continue
        arr = np.asarray(buf, dtype=np.float32)
        env.gp.update_robot(i, arr[:, :2], arr[:, 2])
        buf.clear()
        any_data = True
    if any_data:
        env.gp.fuse()


def run_low_level_loop(env: "MultiRobotSearchEnv", subgoals: dict) -> dict:
    """Run ``subgoal_steps`` low-level controller ticks; return per-step stats.

    Also records per-tick robot positions so the evaluation loop can compute
    exploration overlap at the fine-grained (low-level) timescale rather than
    the coarse MARL-step timescale.
    """
    collisions = 0
    cvar_total = 0.0
    lyap_total = 0.0
    infeasible = 0
    positions_history: List[np.ndarray] = []  # each entry: (N, 2) per-tick positions
    for tick in range(env.cfg.subgoal_steps):
        obs_xys = nearby_obstacle_centres(env)
        margins = obstacle_margins(env, obs_xys)
        for a in env.agents:
            st = env._states[a]
            if st.crashed or st.energy <= 0.0:
                continue
            if env.cfg.use_lyap_mpc and isinstance(st.controller, LyapunovMPC):
                fb = st.controller.compute_control(
                    state=st.pose, goal=subgoals[a],
                    obstacles=obs_xys, obstacle_margins=margins,
                )
                v, omega = fb.v, fb.omega
                energy_dec = fb.energy_consumed
                lyap_total += fb.lyapunov_value
                if not fb.feasible:
                    infeasible += 1
            else:
                v, omega, fb = st.controller.step(state=st.pose, goal=subgoals[a])
                energy_dec = fb.energy_used * env.cfg.dt
            if env.cfg.use_dynamic_step:
                new_pose = st.dyn.dynamic_step(
                    st.pose, v, omega, env.cfg.dt, add_noise=env.cfg.add_noise,
                )
            else:
                new_pose = st.dyn.kinematic_step(
                    st.pose, v, omega, env.cfg.dt, add_noise=env.cfg.add_noise,
                )
            if env.world.check_collision(new_pose[:2], radius=env.cfg.robot_radius) \
                    or env._robot_robot_collision(a, new_pose[:2]):
                st.crashed = True
                collisions += 1
                env._cumulative_collisions += 1
                continue
            st.pose = new_pose
            st.energy = max(0.0, st.energy - energy_dec)
            env._integrate_visibility(st.pose)
            env.world.check_target_detection(st.pose[:2], detect_range=env.cfg.detect_range)
            if env.cfg.use_real_gp:
                h = float(env.world.get_ground_truth_hazard(st.pose[0], st.pose[1], noisy=False))
                h += float(env._rng.normal(0.0, env.cfg.gp_obs_noise_std))
                st.obs_buffer.append((float(st.pose[0]), float(st.pose[1]), h))
            cvar_total += env.gp.cvar_risk(st.pose[:2])
        # Record per-tick positions for fine-grained exploration overlap.
        positions_history.append(
            np.array([env._states[a].pose[:2].copy() for a in env.agents], dtype=np.float32)
        )
        env.gp.update([st.pose[:2] for st in env._states.values() if not st.crashed])
        if env.cfg.use_real_gp and (tick + 1) % max(1, env.cfg.gp_update_interval) == 0:
            flush_gp_buffers(env)
        if all(env._reached_subgoal(env._states[a], subgoals[a]) for a in env.agents):
            break
    if env.cfg.use_real_gp:
        flush_gp_buffers(env)
    return {
        "collisions": collisions,
        "cvar_total": cvar_total,
        "lyap_total": lyap_total,
        "infeasible": infeasible,
        "positions_history": positions_history,
    }
