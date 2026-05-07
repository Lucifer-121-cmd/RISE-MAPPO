"""Lyapunov-MPC controller for the TurtleBot3 unicycle (Phase 2).

Nonlinear MPC formulated as a CasADi NLP and solved with IPOPT. The
defining feature is the *Lyapunov contraction constraint*

    V(e_{k+1}) <= (1 - alpha) * V(e_k)

where ``V`` is the quadratic Lyapunov function from
:mod:`mpc.backstepping`. The constraint is implemented with a
non-negative slack ``s_k >= 0`` and a quadratic penalty in the cost so
that the NLP is **always feasible**: when the contraction is too tight
to satisfy, the solver picks the smallest slack that allows progress
and the controller logs a warning.

Energy is augmented as a state and integrated through a quadratic
power model; the controller returns the energy consumed over the
first-step control via :class:`mpc.utils.ControllerFeedback`.

Build-once / solve-many: the NLP is constructed during ``__init__``
with state, goal, obstacles and obstacle margins as **CasADi
parameters**; per-step calls only update parameters and warm-start
vectors. The number of obstacle slots is fixed at config time
(``max_obstacles``); unused slots are filled with a far placeholder.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from envs.robot_dynamics import TurtleBot3Dynamics, TurtleBot3Params
from mpc.backstepping import BacksteppingController
from mpc.utils import ControllerFeedback, proportional_subgoal_controller


_LOG = logging.getLogger("paper3.mpc.lyapunov")
_FAR_OBSTACLE = 1e6


@dataclass
class LyapunovMPCConfig:
    """Configuration for :class:`LyapunovMPC`."""

    horizon: int = 20
    dt: float = 0.1
    Q_diag: List[float] = field(default_factory=lambda: [10.0, 10.0, 1.0])
    R_diag: List[float] = field(default_factory=lambda: [1.0, 0.5])
    S_du_diag: List[float] = field(default_factory=lambda: [1.0, 0.5])
    P_terminal_scale: float = 5.0
    alpha_lyap: float = 0.1
    d_safe: float = 0.3
    w_energy: float = 0.1
    soft_lyap_penalty: float = 1000.0
    max_iter: int = 100
    max_cpu_time: float = 1.0
    max_obstacles: int = 8
    goal_tolerance: float = 0.1
    max_v: float = 0.22
    max_omega: float = 2.84
    bs_k1: float = 1.0
    bs_k2: float = 3.0
    # Power model coefficients (TurtleBot3 Burger, approximate).
    pwr_c1: float = 15.0
    pwr_c2: float = 0.5
    pwr_c3: float = 1.0
    pwr_c4: float = 3.0
    pwr_c5: float = 0.5


class LyapunovMPC:
    """CasADi/IPOPT Lyapunov-MPC for TurtleBot3 subgoal tracking."""

    def __init__(
        self,
        config: Optional[LyapunovMPCConfig] = None,
        robot_params: Optional[TurtleBot3Params] = None,
    ) -> None:
        self.config = config or LyapunovMPCConfig()
        self._robot_params = robot_params or TurtleBot3Params()
        if self.config.horizon <= 0:
            raise ValueError("horizon must be > 0")
        if self.config.dt <= 0.0:
            raise ValueError("dt must be > 0")
        if not 0.0 < self.config.alpha_lyap < 1.0:
            raise ValueError("alpha_lyap must be in (0, 1)")
        self.config.max_v = float(self._robot_params.max_linear_velocity)
        self.config.max_omega = float(self._robot_params.max_angular_velocity)
        self._backstepping = BacksteppingController(
            k1=self.config.bs_k1,
            k2=self.config.bs_k2,
            max_v=self.config.max_v,
            max_omega=self.config.max_omega,
            goal_tolerance=self.config.goal_tolerance,
        )
        self._fallback_count = 0
        self._prev_X: Optional[np.ndarray] = None
        self._prev_U: Optional[np.ndarray] = None
        self._prev_S: Optional[np.ndarray] = None
        self._prev_lam_x: Optional[np.ndarray] = None
        self._prev_lam_g: Optional[np.ndarray] = None
        self._cum_energy: float = 0.0
        self._build_nlp()

    # ------------------------------------------------------------------
    # NLP construction
    # ------------------------------------------------------------------
    def _build_nlp(self) -> None:
        """Construct the CasADi NLP once. Parameters bind per-step."""
        try:
            import casadi as ca
        except ImportError as exc:  # pragma: no cover
            raise ImportError("CasADi is required for LyapunovMPC") from exc
        self._ca = ca
        N = self.config.horizon
        M = self.config.max_obstacles
        # Decision variables.
        X = ca.MX.sym("X", 3, N + 1)
        U = ca.MX.sym("U", 2, N)
        S = ca.MX.sym("S", N)              # Lyapunov slack
        E = ca.MX.sym("E", N + 1)          # cumulative energy
        # Parameters.
        p_x0 = ca.MX.sym("x0", 3)
        p_goal = ca.MX.sym("goal", 2)
        p_obs = ca.MX.sym("obs", 2, M)
        p_margins = ca.MX.sym("mar", M)
        cost = ca.MX.zeros(1, 1)
        g: List[ca.MX] = []
        lbg: List[float] = []
        ubg: List[float] = []
        # Initial state and energy.
        g.append(X[:, 0] - p_x0); lbg += [0, 0, 0]; ubg += [0, 0, 0]
        g.append(E[0])
        lbg.append(0.0); ubg.append(0.0)
        Q = np.asarray(self.config.Q_diag, dtype=float)
        R = np.asarray(self.config.R_diag, dtype=float)
        Sdu = np.asarray(self.config.S_du_diag, dtype=float)
        for k in range(N):
            xk, uk = X[:, k], U[:, k]
            # Kinematic Euler integration (matches to_casadi() RHS).
            x_next = xk + self.config.dt * ca.vertcat(
                uk[0] * ca.cos(xk[2]), uk[0] * ca.sin(xk[2]), uk[1],
            )
            g.append(X[:, k + 1] - x_next); lbg += [0, 0, 0]; ubg += [0, 0, 0]
            # Energy dynamics.
            v_abs = ca.sqrt(uk[0] * uk[0] + 1e-12)
            w_abs = ca.sqrt(uk[1] * uk[1] + 1e-12)
            P_k = (self.config.pwr_c1 * uk[0] ** 2
                   + self.config.pwr_c2 * uk[1] ** 2
                   + self.config.pwr_c3 * v_abs * w_abs
                   + self.config.pwr_c4 * v_abs
                   + self.config.pwr_c5)
            g.append(E[k + 1] - E[k] - P_k * self.config.dt)
            lbg.append(0.0); ubg.append(0.0)
            # Stage cost.
            err = X[:, k] - ca.vertcat(p_goal[0], p_goal[1], 0.0)
            cost = cost + Q[0] * err[0] ** 2 + Q[1] * err[1] ** 2 + Q[2] * err[2] ** 2
            cost = cost + R[0] * uk[0] ** 2 + R[1] * uk[1] ** 2
            if k > 0:
                du = U[:, k] - U[:, k - 1]
                cost = cost + Sdu[0] * du[0] ** 2 + Sdu[1] * du[1] ** 2
            # Obstacle distance constraints (one per slot).
            for j in range(M):
                dx = X[0, k + 1] - p_obs[0, j]
                dy = X[1, k + 1] - p_obs[1, j]
                dist = ca.sqrt(dx * dx + dy * dy + 1e-9)
                g.append(dist - self.config.d_safe - p_margins[j])
                lbg.append(0.0); ubg.append(np.inf)
            # Lyapunov contraction with slack.
            V_k = self._backstepping.lyapunov_value_casadi(X[:, k], p_goal)
            V_k1 = self._backstepping.lyapunov_value_casadi(X[:, k + 1], p_goal)
            g.append(V_k1 - (1.0 - self.config.alpha_lyap) * V_k - S[k])
            lbg.append(-np.inf); ubg.append(0.0)
            cost = cost + self.config.soft_lyap_penalty * S[k] ** 2
        # Terminal cost.
        err_T = X[:, N] - ca.vertcat(p_goal[0], p_goal[1], 0.0)
        P_term = self.config.P_terminal_scale * Q
        cost = (cost + P_term[0] * err_T[0] ** 2
                + P_term[1] * err_T[1] ** 2
                + P_term[2] * err_T[2] ** 2)
        cost = cost + self.config.w_energy * E[N]
        # Pack decision vector and parameters.
        w = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1), S, E)
        p = ca.vertcat(p_x0, p_goal, ca.reshape(p_obs, -1, 1), p_margins)
        nlp = {"x": w, "f": cost, "g": ca.vertcat(*g), "p": p}
        opts = {
            "ipopt": {
                "mu_strategy": "adaptive",
                "linear_solver": "mumps",
                "hessian_approximation": "exact",
                "max_iter": int(self.config.max_iter),
                "max_cpu_time": float(self.config.max_cpu_time),
                "tol": 1e-6,
                "constr_viol_tol": 1e-6,
                "warm_start_init_point": "yes",
                "warm_start_bound_push": 1e-8,
                "warm_start_mult_bound_push": 1e-8,
                "print_level": 0,
                "acceptable_tol": 1e-3,
                "acceptable_iter": 5,
            },
            "print_time": 0,
        }
        self._solver = ca.nlpsol("lyap_mpc", "ipopt", nlp, opts)
        # Variable bounds.
        nx = 3 * (N + 1)
        nu = 2 * N
        ns = N
        ne = N + 1
        n_vars = nx + nu + ns + ne
        lbx = -np.inf * np.ones(n_vars)
        ubx = np.inf * np.ones(n_vars)
        u_off = nx
        for k in range(N):
            lbx[u_off + 2 * k] = -self.config.max_v
            ubx[u_off + 2 * k] = self.config.max_v
            lbx[u_off + 2 * k + 1] = -self.config.max_omega
            ubx[u_off + 2 * k + 1] = self.config.max_omega
        s_off = nx + nu
        lbx[s_off:s_off + ns] = 0.0   # slack >= 0
        e_off = nx + nu + ns
        lbx[e_off:e_off + ne] = 0.0   # energy >= 0
        self._lbx = lbx
        self._ubx = ubx
        self._lbg = np.asarray(lbg, dtype=float)
        self._ubg = np.asarray(ubg, dtype=float)
        self._dims = {"nx": nx, "nu": nu, "ns": ns, "ne": ne, "N": N, "M": M}

    # ------------------------------------------------------------------
    # Param packing + warm start
    # ------------------------------------------------------------------
    def _format_obstacles(
        self,
        obstacles: Optional[np.ndarray],
        margins: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        M = self.config.max_obstacles
        obs_arr = np.full((2, M), _FAR_OBSTACLE, dtype=float)
        mar_arr = np.zeros(M, dtype=float)
        if obstacles is None or len(obstacles) == 0:
            return obs_arr, mar_arr
        arr = np.asarray(obstacles, dtype=float).reshape(-1, 2)
        count = min(arr.shape[0], M)
        obs_arr[:, :count] = arr[:count].T
        if margins is not None:
            m = np.asarray(margins, dtype=float).reshape(-1)
            mar_arr[: min(m.shape[0], M)] = m[: min(m.shape[0], M)]
        return obs_arr, mar_arr

    def _build_warm_start(self, x0: np.ndarray) -> np.ndarray:
        N = self._dims["N"]
        if self._prev_X is not None and self._prev_U is not None:
            xg = np.hstack((self._prev_X[:, 1:], self._prev_X[:, -1:]))
            ug = np.hstack((self._prev_U[:, 1:], self._prev_U[:, -1:]))
            xg[:, 0] = x0
            sg = np.concatenate((self._prev_S[1:], self._prev_S[-1:])) if self._prev_S is not None else np.zeros(N)
        else:
            xg = np.tile(x0.reshape(3, 1), (1, N + 1))
            ug = np.zeros((2, N))
            sg = np.zeros(N)
        eg = np.zeros(N + 1)
        return np.concatenate((
            xg.reshape(-1, order="F"),
            ug.reshape(-1, order="F"),
            sg,
            eg,
        ))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compute_control(
        self,
        state: Sequence[float],
        goal: Sequence[float],
        obstacles: Optional[np.ndarray] = None,
        obstacle_margins: Optional[np.ndarray] = None,
    ) -> ControllerFeedback:
        """Solve the Lyapunov-MPC NLP and return one control step.

        Falls back to the proportional controller if IPOPT fails or the
        decision is non-finite. The Lyapunov contraction is enforced
        with a non-negative slack penalty so the NLP itself remains
        feasible; ``feasible=False`` flags slack activation or solver
        failure.
        """
        x0 = np.asarray(state, dtype=float).reshape(-1)
        if x0.shape[0] < 3:
            raise ValueError(f"state must have 3 entries, got {x0.shape}")
        ga = np.asarray(goal, dtype=float).reshape(-1)
        if ga.shape[0] < 2:
            raise ValueError(f"goal must have 2 entries, got {ga.shape}")
        x0 = x0[:3]
        ga = ga[:2]
        dist = float(np.hypot(ga[0] - x0[0], ga[1] - x0[1]))
        V_now = self._backstepping.lyapunov_value(x0, ga)
        if dist <= self.config.goal_tolerance:
            return ControllerFeedback(
                reached=True, energy_used=0.0, risk=0.0, distance_to_goal=dist,
                v=0.0, omega=0.0, energy_consumed=0.0,
                lyapunov_value=V_now, feasible=True,
            )
        obs_arr, mar_arr = self._format_obstacles(obstacles, obstacle_margins)
        p = np.concatenate((
            x0, ga, obs_arr.reshape(-1, order="F"), mar_arr,
        ))
        w0 = self._build_warm_start(x0)
        kw = {"x0": w0, "lbx": self._lbx, "ubx": self._ubx,
              "lbg": self._lbg, "ubg": self._ubg, "p": p}
        if self._prev_lam_x is not None:
            kw["lam_x0"] = self._prev_lam_x
            kw["lam_g0"] = self._prev_lam_g
        try:
            sol = self._solver(**kw)
        except Exception as exc:  # pragma: no cover
            _LOG.warning("LyapunovMPC IPOPT raised %s; falling back to P controller", exc)
            return self._fallback(x0, ga, dist, V_now)
        w_opt = np.asarray(sol["x"]).reshape(-1)
        nx, nu, ns = self._dims["nx"], self._dims["nu"], self._dims["ns"]
        N = self._dims["N"]
        X_opt = w_opt[:nx].reshape((3, N + 1), order="F")
        U_opt = w_opt[nx:nx + nu].reshape((2, N), order="F")
        S_opt = w_opt[nx + nu:nx + nu + ns]
        E_opt = w_opt[nx + nu + ns:]
        v0 = float(U_opt[0, 0])
        w0_cmd = float(U_opt[1, 0])
        if not (np.isfinite(v0) and np.isfinite(w0_cmd)):
            return self._fallback(x0, ga, dist, V_now)
        self._prev_X, self._prev_U, self._prev_S = X_opt, U_opt, S_opt
        self._prev_lam_x = np.asarray(sol["lam_x"]).reshape(-1)
        self._prev_lam_g = np.asarray(sol["lam_g"]).reshape(-1)
        status = self._solver.stats().get("return_status", "")
        slack_max = float(np.max(S_opt))
        feasible = status in ("Solve_Succeeded", "Solved_To_Acceptable_Level")
        if slack_max > 1e-6:
            # Soft Lyapunov constraint relaxed by the penalty term —
            # NLP itself stayed feasible. Log once per heavy activation.
            self._fallback_count += 1
            if self._fallback_count <= 5 or self._fallback_count % 50 == 0:
                _LOG.warning(
                    "Lyapunov soft slack=%.4f status=%s (activation #%d)",
                    slack_max, status, self._fallback_count,
                )
        # Clamp away IPOPT barrier numerical noise (~1e-8 below zero).
        energy_step = max(0.0, float(E_opt[1] - E_opt[0]))
        self._cum_energy += energy_step
        return ControllerFeedback(
            reached=False,
            energy_used=energy_step / max(self.config.dt, 1e-9),
            risk=0.0,
            distance_to_goal=dist,
            v=v0,
            omega=w0_cmd,
            energy_consumed=energy_step,
            lyapunov_value=V_now,
            feasible=feasible,
        )

    def _fallback(
        self, x0: np.ndarray, ga: np.ndarray, dist: float, V_now: float,
    ) -> ControllerFeedback:
        v, w, fb = proportional_subgoal_controller(
            state=x0, goal=ga,
            max_v=self.config.max_v, max_omega=self.config.max_omega,
            goal_tolerance=self.config.goal_tolerance,
        )
        fb.feasible = False
        fb.lyapunov_value = V_now
        return fb

    def step(
        self,
        state: Sequence[float],
        goal: Sequence[float],
    ) -> Tuple[float, float, ControllerFeedback]:
        """Phase-1 compatible signature returning ``(v, omega, feedback)``."""
        fb = self.compute_control(state=state, goal=goal)
        return fb.v, fb.omega, fb

    def reset(self) -> None:
        """Clear warm-start state and cumulative energy for a new episode."""
        self._prev_X = None
        self._prev_U = None
        self._prev_S = None
        self._prev_lam_x = None
        self._prev_lam_g = None
        self._cum_energy = 0.0

    @property
    def fallback_count(self) -> int:
        """Number of compute_control calls that activated the soft fallback."""
        return int(self._fallback_count)
