"""Low-level controller utilities.

This file holds the **Phase-1 placeholder** controller used by the
multi-robot search environment until Lyapunov-MPC is implemented in
Phase 2. The interface intentionally matches the signature that
``mpc.lyapunov_mpc.LyapunovMPC.step`` will expose, so the env never
needs to change when we swap controllers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from envs.robot_dynamics import normalize_angle


@dataclass
class ControllerFeedback:
    """Tuple returned by every low-level controller call.

    Phase-1 fields (proportional fallback)
    --------------------------------------
    reached : bool
        Whether the robot is within ``goal_tolerance`` of the subgoal.
    energy_used : float
        Energy consumed *per second* of control (legacy; multiplied by
        dt at the env caller). Phase-2 controllers populate
        :attr:`energy_consumed` instead.
    risk : float
        CVaR risk encountered during this step (Phase 1: 0.0).
    distance_to_goal : float
        Euclidean distance from the current state to the subgoal.

    Phase-2 fields (Lyapunov-MPC)
    -----------------------------
    v, omega : float
        First-step control commands returned by the NLP.
    energy_consumed : float
        Energy consumed over this step (J), from the power model.
    lyapunov_value : float
        ``V(e)`` at the current state.
    feasible : bool
        True if IPOPT returned an acceptable solution; False if the
        soft Lyapunov fallback was activated or the solver failed.
    """

    reached: bool = False
    energy_used: float = 0.0
    risk: float = 0.0
    distance_to_goal: float = 0.0
    v: float = 0.0
    omega: float = 0.0
    energy_consumed: float = 0.0
    lyapunov_value: float = 0.0
    feasible: bool = True


def proportional_subgoal_controller(
    state: Sequence[float],
    goal: Sequence[float],
    max_v: float,
    max_omega: float,
    kv: float = 0.6,
    kw: float = 2.0,
    goal_tolerance: float = 0.1,
) -> tuple[float, float, ControllerFeedback]:
    """Phase-1 proportional pose regulator.

    Returns ``(v, omega, feedback)``. Replaced wholesale by Lyapunov-MPC
    in Phase 2 without changing the environment interface.

    Parameters
    ----------
    state : (3,) array-like
        ``[x, y, theta]``.
    goal : (2,) array-like
        Subgoal position.
    max_v, max_omega : float
        Actuator saturation limits.
    kv, kw : float
        Proportional gains.
    goal_tolerance : float
        Distance below which ``reached`` is True.
    """
    s = np.asarray(state, dtype=float).reshape(-1)
    g = np.asarray(goal, dtype=float).reshape(-1)
    dx = g[0] - s[0]
    dy = g[1] - s[1]
    dist = float(np.hypot(dx, dy))
    angle_err = float(normalize_angle(np.arctan2(dy, dx) - s[2]))
    # Slow down when far off-axis to reduce drift on tight headings.
    heading_factor = max(0.0, np.cos(angle_err))
    v = float(np.clip(kv * dist * heading_factor, -max_v, max_v))
    omega = float(np.clip(kw * angle_err, -max_omega, max_omega))
    reached = dist <= goal_tolerance
    if reached:
        v = 0.0
        omega = 0.0
    fb = ControllerFeedback(
        reached=reached,
        energy_used=abs(v) + 0.1 * abs(omega),
        risk=0.0,
        distance_to_goal=dist,
        v=v,
        omega=omega,
        energy_consumed=abs(v) + 0.1 * abs(omega),
        lyapunov_value=0.5 * dist * dist,
        feasible=True,
    )
    return v, omega, fb
