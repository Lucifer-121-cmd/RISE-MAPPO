"""Backstepping controller for the TurtleBot3 unicycle model.

Phase-2 STEP 1 of Paper 3. The controller drives the robot to a 2-D
subgoal (no reference heading) using a quadratic Lyapunov function in
the body-frame tracking error. The same Lyapunov function is exported
symbolically (CasADi) so :mod:`mpc.lyapunov_mpc` can attach the
contraction constraint ``V(e_{k+1}) <= (1 - alpha) V(e_k)`` to the NLP.

Tracking error in the robot body frame::

    e_x =  cos(theta) * (x_g - x) + sin(theta) * (y_g - y)
    e_y = -sin(theta) * (x_g - x) + cos(theta) * (y_g - y)

Lyapunov function::

    V(e) = 0.5 * (e_x**2 + e_y**2)

Backstepping point-tracking control law (no heading reference)::

    v     = k1 * e_x
    omega = k2 * atan2(e_y, e_x)

Both commands are saturated to the TurtleBot3 Burger limits.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


@dataclass
class BacksteppingConfig:
    """Configuration for :class:`BacksteppingController`."""

    k1: float = 1.0
    k2: float = 3.0
    max_v: float = 0.22
    max_omega: float = 2.84
    goal_tolerance: float = 0.05


class BacksteppingController:
    """Nonlinear backstepping point-tracking controller.

    Parameters
    ----------
    k1 : float
        Forward velocity gain on the body-frame longitudinal error.
    k2 : float
        Steering gain on the line-of-sight angle to the subgoal.
    max_v, max_omega : float
        Actuator saturation limits (TurtleBot3 Burger defaults).
    goal_tolerance : float
        Distance below which :meth:`compute_control` returns zero
        commands and reports the goal as reached.
    """

    def __init__(
        self,
        k1: float = 1.0,
        k2: float = 3.0,
        max_v: float = 0.22,
        max_omega: float = 2.84,
        goal_tolerance: float = 0.05,
    ) -> None:
        if k1 <= 0.0 or k2 <= 0.0:
            raise ValueError("backstepping gains k1, k2 must be > 0")
        if max_v <= 0.0 or max_omega <= 0.0:
            raise ValueError("actuator limits must be > 0")
        self.cfg = BacksteppingConfig(
            k1=float(k1),
            k2=float(k2),
            max_v=float(max_v),
            max_omega=float(max_omega),
            goal_tolerance=float(goal_tolerance),
        )

    # ------------------------------------------------------------------
    # Tracking-error helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _body_frame_error(
        state: Sequence[float],
        goal: Sequence[float],
    ) -> Tuple[float, float]:
        """Return ``(e_x, e_y)`` body-frame tracking error to ``goal``."""
        s = np.asarray(state, dtype=float).reshape(-1)
        g = np.asarray(goal, dtype=float).reshape(-1)
        if s.shape[0] < 3:
            raise ValueError(f"state must have at least 3 entries, got {s.shape}")
        if g.shape[0] < 2:
            raise ValueError(f"goal must have at least 2 entries, got {g.shape}")
        dx = float(g[0] - s[0])
        dy = float(g[1] - s[1])
        c, sn = float(np.cos(s[2])), float(np.sin(s[2]))
        e_x = c * dx + sn * dy
        e_y = -sn * dx + c * dy
        return e_x, e_y

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compute_control(
        self,
        state: Sequence[float],
        goal: Sequence[float],
    ) -> Tuple[float, float]:
        """Compute saturated ``(v, omega)`` backstepping control.

        Returns zero commands when the robot is within
        :attr:`BacksteppingConfig.goal_tolerance` of ``goal``.
        """
        e_x, e_y = self._body_frame_error(state, goal)
        dist = float(np.hypot(e_x, e_y))
        if dist <= self.cfg.goal_tolerance:
            return 0.0, 0.0
        v = self.cfg.k1 * e_x
        # atan2(e_y, e_x) is the line-of-sight angle to the subgoal in
        # the body frame; it is bounded in (-pi, pi] which keeps omega
        # well-conditioned even close to the goal.
        omega = self.cfg.k2 * float(np.arctan2(e_y, e_x))
        v = float(np.clip(v, -self.cfg.max_v, self.cfg.max_v))
        omega = float(np.clip(omega, -self.cfg.max_omega, self.cfg.max_omega))
        return v, omega

    def lyapunov_value(
        self,
        state: Sequence[float],
        goal: Sequence[float],
    ) -> float:
        """Compute ``V(e) = 0.5 * (e_x**2 + e_y**2)`` at ``(state, goal)``."""
        e_x, e_y = self._body_frame_error(state, goal)
        return 0.5 * (e_x * e_x + e_y * e_y)

    def lyapunov_value_casadi(self, x_sym, goal_param):
        """Return the symbolic Lyapunov expression ``V(e)`` for CasADi.

        Parameters
        ----------
        x_sym : casadi.SX or casadi.MX
            Symbolic state vector ``[x, y, theta]``.
        goal_param : casadi.SX or casadi.MX
            Symbolic / parameter vector ``[x_g, y_g]`` (size 2 minimum;
            extra entries are ignored).

        Returns
        -------
        casadi expression
            ``0.5 * (e_x**2 + e_y**2)`` written in body-frame tracking
            error of ``x_sym`` relative to ``goal_param``.
        """
        try:
            import casadi as ca  # local import keeps casadi optional
        except ImportError as exc:  # pragma: no cover
            raise ImportError("CasADi is required for lyapunov_value_casadi()") from exc

        dx = goal_param[0] - x_sym[0]
        dy = goal_param[1] - x_sym[1]
        c = ca.cos(x_sym[2])
        s = ca.sin(x_sym[2])
        e_x = c * dx + s * dy
        e_y = -s * dx + c * dy
        return 0.5 * (e_x * e_x + e_y * e_y)
