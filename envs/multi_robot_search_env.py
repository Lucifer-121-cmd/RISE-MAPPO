"""Hierarchical multi-robot search environment.

PettingZoo-parallel-style API. The MARL action is a discrete subgoal
(one of ``K=25`` cells of a 5×5 ego-centric grid at 1 m spacing).
After commitment, the env runs ``subgoal_steps`` low-level control
ticks per robot via :class:`mpc.lyapunov_mpc.LyapunovMPC`. Per-agent
observation and centralised global-state shapes are exposed via
:meth:`MultiRobotSearchEnv.observation_shapes` /
:meth:`MultiRobotSearchEnv.global_state_shapes`. Rewards are team-shared.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from envs.grid_world import GridWorld, GridWorldConfig
from envs.robot_dynamics import TurtleBot3Dynamics, TurtleBot3Params
from gp.distributed_gp import DistributedGP, DistributedGPConfig
from gp.local_gp import LocalGPConfig
from mpc.lyapunov_mpc import LyapunovMPC, LyapunovMPCConfig
from mpc.utils import ControllerFeedback, proportional_subgoal_controller


_PATCH_SIZE = 48
_SUBGOAL_GRID = 5            # 5×5 → K=25 discrete subgoal actions
_SUBGOAL_SPACING_M = 1.0


@dataclass
class EnvConfig:
    """Top-level environment configuration."""
    num_robots: int = 3
    world_size: float = 10.0
    max_steps: int = 500          # MARL high-level steps
    sensor_range: float = 1.5
    dt: float = 0.1
    difficulty: str = "medium"
    num_targets: int = 5
    num_obstacles: int = 10
    num_hazards: int = 3
    subgoal_steps: int = 25       # low-level ticks per MARL action
    subgoal_grid: int = _SUBGOAL_GRID
    subgoal_spacing: float = _SUBGOAL_SPACING_M
    detect_range: float = 0.4
    robot_radius: float = 0.105   # TB3 Burger footprint
    energy_budget: float = 100.0
    add_noise: bool = False
    seed: Optional[int] = None
    use_dynamic_step: bool = False
    # Reward weights (mirrors configs/default.yaml).
    w_coverage: float = 1.0
    w_detection: float = 5.0
    w_cvar_risk: float = 0.5
    w_energy: float = 0.3
    w_coordination: float = 0.2
    w_collision: float = 10.0
    coord_min_dist: float = 1.5
    # Optional: cell side for the ego-centric patches; defaults to GP res.
    patch_cell_size: Optional[float] = None
    # Phase-2 toggles (default off so Phase-1 tests stay fast).
    use_lyap_mpc: bool = False
    use_real_gp: bool = False
    gp_update_interval: int = 10
    gp_obs_noise_std: float = 0.05
    cvar_alpha: float = 0.95
    obstacle_margin_scale: float = 0.2
    # MPC parameters — passed through to LyapunovMPCConfig via make_controller().
    # When empty, the controller uses its own dataclass defaults.
    mpc: dict = field(default_factory=dict)


@dataclass
class _RobotState:
    """Per-robot bookkeeping inside the env."""
    pose: np.ndarray
    energy: float
    last_subgoal: np.ndarray
    crashed: bool = False
    dyn: TurtleBot3Dynamics = field(default_factory=TurtleBot3Dynamics)
    controller: Any = None  # LyapunovMPC or proportional callable
    obs_buffer: List[Tuple[float, float, float]] = field(default_factory=list)


class MultiRobotSearchEnv:
    """Hierarchical MARL environment. PettingZoo-parallel-flavoured API."""

    metadata = {"render_modes": ["matplotlib"], "name": "paper3_multi_robot_search"}

    def __init__(self, cfg: Optional[EnvConfig] = None) -> None:
        self.cfg = cfg or EnvConfig()
        if self.cfg.num_robots < 1:
            raise ValueError("num_robots must be >= 1")
        if self.cfg.subgoal_grid * self.cfg.subgoal_grid != _SUBGOAL_GRID ** 2:
            # Allow other K values but warn — assumed 25 elsewhere.
            pass
        self._rng = np.random.default_rng(self.cfg.seed)
        self.world = GridWorld(GridWorldConfig(
            world_size=self.cfg.world_size,
            num_obstacles=self.cfg.num_obstacles,
            num_targets=self.cfg.num_targets,
            num_hazards=self.cfg.num_hazards,
            difficulty=self.cfg.difficulty,
            detect_range=self.cfg.detect_range,
            seed=self.cfg.seed,
        ))
        self.gp = DistributedGP(
            DistributedGPConfig(
                world_size=self.cfg.world_size,
                resolution=self.world.cfg.resolution,
                cvar_alpha=self.cfg.cvar_alpha,
                local_gp=LocalGPConfig(world_size=self.cfg.world_size),
            ),
            num_robots=self.cfg.num_robots if self.cfg.use_real_gp else 0,
        )
        self.agents: List[str] = [f"robot_{i}" for i in range(self.cfg.num_robots)]
        self._states: Dict[str, _RobotState] = {}
        self._coverage_mask = np.zeros(
            (self.world.grid_size, self.world.grid_size), dtype=bool
        )
        self._merged_occupancy = np.zeros_like(self._coverage_mask, dtype=np.float32)
        self._high_level_step = 0
        self._cumulative_collisions = 0

    @property
    def num_subgoal_actions(self) -> int:
        return self.cfg.subgoal_grid ** 2

    @property
    def patch_size(self) -> int:
        return _PATCH_SIZE

    def observation_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Return per-agent observation key → shape mapping."""
        n = self.cfg.num_robots
        return {
            "occupancy_patch": (_PATCH_SIZE, _PATCH_SIZE),
            "gp_sigma_patch": (_PATCH_SIZE, _PATCH_SIZE),
            "rel_robots": (max(n - 1, 1), 2),
            "energy": (1,),
            "coverage": (1,),
            "agent_id": (n,),
        }

    def global_state_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Return centralised-critic state key → shape mapping."""
        g = self.world.grid_size
        n = self.cfg.num_robots
        return {
            "occupancy": (g, g),
            "gp_sigma": (g, g),
            "robot_states": (n, 3),
            "energies": (n,),
            "stats": (2,),
        }

    def reset(
        self,
        *,
        seed: Optional[int] = None,
    ) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Any]]:
        """Reset env to a freshly sampled world; return ``(obs, info)``."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self.cfg.seed = seed
        self.world.reset(seed=seed)
        self.gp.reset()
        self._coverage_mask.fill(False)
        self._merged_occupancy = self.world.get_occupancy_grid().copy()
        self._high_level_step = 0
        self._cumulative_collisions = 0
        # Spawn each robot in a free cell, well-separated.
        spawn_avoid: List[Tuple[float, float, float]] = []
        for agent in self.agents:
            x, y = self._sample_free_position(spawn_avoid)
            theta = float(self._rng.uniform(-np.pi, np.pi))
            self._states[agent] = _RobotState(
                pose=np.array([x, y, theta]),
                energy=self.cfg.energy_budget,
                last_subgoal=np.array([x, y]),
                dyn=TurtleBot3Dynamics(
                    params=TurtleBot3Params(),
                    rng=np.random.default_rng(self._rng.integers(2 ** 31)),
                ),
                controller=self._make_controller(),
            )
            spawn_avoid.append((x, y, 0.5))
        # Initial visibility update.
        for st in self._states.values():
            self._integrate_visibility(st.pose)
        self.gp.update([st.pose[:2] for st in self._states.values()])
        info = {"step": 0, "coverage": self._coverage_fraction()}
        return self._build_observations(), info

    def step(
        self,
        actions: Mapping[str, int],
    ) -> Tuple[
        Dict[str, Dict[str, np.ndarray]],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Any],
    ]:
        """Run one MARL step: K low-level ticks per robot toward each subgoal.

        Returns ``(obs, rewards, terminations, truncations, infos)``.
        Rewards are team-shared.
        """
        if set(actions.keys()) != set(self.agents):
            missing = set(self.agents) - set(actions.keys())
            extra = set(actions.keys()) - set(self.agents)
            raise ValueError(f"action keys mismatch: missing={missing} extra={extra}")
        # 1. Convert each MARL action to a subgoal in world coords.
        subgoals = {a: self._action_to_subgoal(self._states[a], int(actions[a])) for a in self.agents}
        for a in self.agents:
            self._states[a].last_subgoal = subgoals[a]
        # 2. Run K low-level ticks per robot in interleaved fashion.
        from envs.integrations import run_low_level_loop
        coverage_before = self._coverage_fraction()
        targets_before = self._detected_count()
        energy_before = sum(st.energy for st in self._states.values())
        loop_stats = run_low_level_loop(self, subgoals)
        collisions_this_step = loop_stats["collisions"]
        cvar_total = loop_stats["cvar_total"]
        lyap_total = loop_stats["lyap_total"]
        infeasible_count = loop_stats["infeasible"]
        # 3. Compute team reward.
        delta_cov = self._coverage_fraction() - coverage_before
        new_targets = self._detected_count() - targets_before
        delta_energy = energy_before - sum(st.energy for st in self._states.values())
        coord_bonus = self._coordination_bonus()
        info_gain = self.gp.information_gain()
        normalised_cvar = cvar_total / max(1, self.cfg.subgoal_steps * self.cfg.num_robots)
        team_reward = (
            self.cfg.w_coverage * (delta_cov + 0.1 * info_gain)
            + self.cfg.w_detection * float(new_targets)
            - self.cfg.w_cvar_risk * normalised_cvar
            - self.cfg.w_energy * (delta_energy / max(1.0, self.cfg.energy_budget))
            + self.cfg.w_coordination * coord_bonus
            - self.cfg.w_collision * float(collisions_this_step)
        )
        # 4. Increment counter and assemble outputs.
        self._high_level_step += 1
        all_targets_found = self._detected_count() == self.cfg.num_targets
        out_of_energy = all(st.energy <= 0.0 for st in self._states.values())
        all_crashed = all(st.crashed for st in self._states.values())
        truncated = self._high_level_step >= self.cfg.max_steps
        terminated = all_targets_found or out_of_energy or all_crashed
        rewards = {a: float(team_reward) for a in self.agents}
        terminations = {a: bool(terminated) for a in self.agents}
        truncations = {a: bool(truncated) for a in self.agents}
        infos = {
            "step": self._high_level_step,
            "coverage": self._coverage_fraction(),
            "detected": self._detected_count(),
            "collisions_step": collisions_this_step,
            "collisions_total": self._cumulative_collisions,
            "info_gain": info_gain,
            "team_reward": float(team_reward),
            "lyapunov_mean": (
                lyap_total / max(1, self.cfg.subgoal_steps * self.cfg.num_robots)
            ),
            "mpc_infeasible": int(infeasible_count),
            # RISE-MAPPO: normalised CVaR risk (same scale as reward's
            # cvar term) and per-agent GP sigma. Storing raw cvar_total
            # blew up the CVaR critic head because risk targets ended up
            # ~subgoal_steps×num_robots larger than reward magnitude.
            "risk_cost": float(normalised_cvar),
            "agent_sigmas": self._per_agent_sigmas(),
            # Fine-grained per-tick positions for exploration overlap metric.
            "positions_history": loop_stats.get("positions_history", []),
        }
        return self._build_observations(), rewards, terminations, truncations, infos

    def render(self, ax=None):
        """Render world + robots with matplotlib."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))
        self.world.render(ax)
        for a, st in self._states.items():
            x, y, th = st.pose
            colour = "red" if st.crashed else "black"
            ax.plot(x, y, "o", color=colour)
            ax.arrow(x, y, 0.3 * np.cos(th), 0.3 * np.sin(th),
                     head_width=0.08, color=colour)
            ax.plot(st.last_subgoal[0], st.last_subgoal[1], "x", color="purple")
        return ax

    def global_state(self) -> Dict[str, np.ndarray]:
        """Return the centralised-critic state dict for the current step."""
        return self._build_global_state()

    def close(self) -> None:
        """Release env resources. No-op; matches Gym/PettingZoo convention."""
        return None

    def _make_controller(self):
        """Return per-robot controller (Phase-2 NLP if configured, else P).

        Passes the ``mpc`` config section (if present) so every MPC parameter
        is YAML-configurable with no hardcoded values in the construction path.
        """
        from envs.integrations import make_controller
        mpc_cfg = getattr(self.cfg, "mpc", None) or {}
        return make_controller(self, mpc_cfg)

    def _sample_free_position(
        self,
        avoid: List[Tuple[float, float, float]],
        max_tries: int = 200,
    ) -> Tuple[float, float]:
        for _ in range(max_tries):
            x = float(self._rng.uniform(0.5, self.cfg.world_size - 0.5))
            y = float(self._rng.uniform(0.5, self.cfg.world_size - 0.5))
            if self.world.check_collision((x, y), radius=self.cfg.robot_radius):
                continue
            if any((x - cx) ** 2 + (y - cy) ** 2 < r ** 2 for (cx, cy, r) in avoid):
                continue
            return x, y
        return (1.0, 1.0)

    def _action_to_subgoal(self, st: _RobotState, action: int) -> np.ndarray:
        K = self.cfg.subgoal_grid
        if not 0 <= action < K * K:
            raise ValueError(f"action {action} outside [0, {K * K})")
        row = action // K
        col = action % K
        half = K // 2
        dx = (col - half) * self.cfg.subgoal_spacing
        dy = (row - half) * self.cfg.subgoal_spacing
        gx = float(np.clip(st.pose[0] + dx, 0.1, self.cfg.world_size - 0.1))
        gy = float(np.clip(st.pose[1] + dy, 0.1, self.cfg.world_size - 0.1))
        return np.array([gx, gy])

    def _reached_subgoal(self, st: _RobotState, goal: np.ndarray, tol: float = 0.15) -> bool:
        return float(np.hypot(goal[0] - st.pose[0], goal[1] - st.pose[1])) <= tol

    def _integrate_visibility(self, robot_pose: np.ndarray) -> None:
        vis = self.world.get_visibility_mask(robot_pose[:2], self.cfg.sensor_range)
        self._coverage_mask |= vis

    def _coverage_fraction(self) -> float:
        return float(self._coverage_mask.mean())

    def _detected_count(self) -> int:
        return int(sum(1 for t in self.world.targets if t.detected))

    def _nearby_obstacle_centres(self) -> Optional[np.ndarray]:
        from envs.integrations import nearby_obstacle_centres
        return nearby_obstacle_centres(self)

    def _obstacle_margins(self, obstacles: Optional[np.ndarray]) -> Optional[np.ndarray]:
        from envs.integrations import obstacle_margins
        return obstacle_margins(self, obstacles)

    def _flush_gp_buffers(self) -> None:
        from envs.integrations import flush_gp_buffers
        flush_gp_buffers(self)

    def _robot_robot_collision(self, agent: str, new_xy: np.ndarray) -> bool:
        """Return True if ``new_xy`` overlaps another live robot's footprint."""
        r = self.cfg.robot_radius
        for other_id, other_st in self._states.items():
            if other_id == agent or other_st.crashed:
                continue
            dx = new_xy[0] - other_st.pose[0]
            dy = new_xy[1] - other_st.pose[1]
            if dx * dx + dy * dy <= (2.0 * r) ** 2:
                return True
        return False

    def _coordination_bonus(self) -> float:
        if self.cfg.num_robots < 2:
            return 0.0
        positions = np.array([st.pose[:2] for st in self._states.values()])
        diffs = positions[:, None, :] - positions[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        np.fill_diagonal(dists, np.inf)
        min_dist = float(dists.min())
        return float(np.clip(min_dist / max(1e-6, self.cfg.coord_min_dist), 0.0, 1.0))

    def _build_observations(self) -> Dict[str, Dict[str, np.ndarray]]:
        from envs.observations import build_observations
        return build_observations(self, _PATCH_SIZE)

    def _build_global_state(self) -> Dict[str, np.ndarray]:
        from envs.observations import build_global_state
        return build_global_state(self)

    def _per_agent_sigmas(self) -> np.ndarray:
        """Per-agent GP sigma at each robot's position (RISE-MAPPO attention)."""
        sigma_grid = self.gp.uncertainty_grid()
        g = self.world.grid_size
        res = self.world.cfg.resolution
        out = np.zeros(self.cfg.num_robots, dtype=np.float32)
        for i, a in enumerate(self.agents):
            x, y = self._states[a].pose[:2]
            ix = int(np.clip(x / res, 0, g - 1))
            iy = int(np.clip(y / res, 0, g - 1))
            out[i] = float(sigma_grid[iy, ix])
        return out
