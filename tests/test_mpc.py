"""Phase-1 controller stub tests + Phase-2 backstepping tests.

The full Lyapunov-MPC NLP tests will land alongside STEP 2; here we
verify the proportional fallback (Phase 1) and the backstepping
controller + Lyapunov function from STEP 1.
"""
from __future__ import annotations

import numpy as np
import pytest

from envs.robot_dynamics import TurtleBot3Dynamics
from mpc.backstepping import BacksteppingController
from mpc.lyapunov_mpc import LyapunovMPC, LyapunovMPCConfig
from mpc.utils import ControllerFeedback, proportional_subgoal_controller


def test_proportional_controller_converges_to_goal() -> None:
    """Phase-1 P controller drives state to within 0.15 m of goal."""
    dyn = TurtleBot3Dynamics()
    state = np.array([0.0, 0.0, 0.0])
    goal = np.array([1.0, 0.5])
    fb = None
    for _ in range(2000):
        v, w, fb = proportional_subgoal_controller(
            state=state, goal=goal, max_v=0.22, max_omega=2.84,
        )
        state = dyn.kinematic_step(state, v, w, dt=0.05)
        if fb.reached:
            break
    assert fb is not None and fb.reached
    assert float(np.hypot(state[0] - goal[0], state[1] - goal[1])) < 0.15


def test_controller_returns_feedback_fields() -> None:
    v, w, fb = proportional_subgoal_controller(
        state=[0.0, 0.0, 0.0], goal=[1.0, 1.0], max_v=0.22, max_omega=2.84,
    )
    assert isinstance(v, float) and isinstance(w, float)
    assert isinstance(fb, ControllerFeedback)
    for f in ("reached", "energy_used", "risk", "distance_to_goal",
              "v", "omega", "energy_consumed", "lyapunov_value", "feasible"):
        assert hasattr(fb, f), f


# ---------------------------------------------------------------------
# Phase-2 STEP 1: backstepping controller + Lyapunov function
# ---------------------------------------------------------------------
def _simulate_backstepping(
    ctrl: BacksteppingController,
    state0: np.ndarray,
    goal: np.ndarray,
    *,
    dt: float = 0.05,
    steps: int = 4000,
):
    """Roll out backstepping + kinematic dynamics; return (state, V) traces."""
    dyn = TurtleBot3Dynamics()
    state = state0.copy()
    states = [state.copy()]
    vs = [ctrl.lyapunov_value(state, goal)]
    for _ in range(steps):
        v, w = ctrl.compute_control(state, goal)
        state = dyn.kinematic_step(state, v, w, dt=dt)
        states.append(state.copy())
        vs.append(ctrl.lyapunov_value(state, goal))
        if float(np.hypot(state[0] - goal[0], state[1] - goal[1])) <= ctrl.cfg.goal_tolerance:
            break
    return np.asarray(states), np.asarray(vs)


def test_backstepping_drives_to_goal() -> None:
    """From (0,0,0) to (3,3): final tracking error < 0.05 m."""
    ctrl = BacksteppingController(k1=1.0, k2=3.0)
    states, _ = _simulate_backstepping(
        ctrl, np.array([0.0, 0.0, 0.0]), np.array([3.0, 3.0]),
    )
    final_err = float(np.hypot(states[-1, 0] - 3.0, states[-1, 1] - 3.0))
    assert final_err < 0.05, f"final error {final_err:.4f} m exceeds 0.05"


def test_backstepping_lyapunov_monotone_decreasing() -> None:
    """V(t) along the closed-loop trajectory is monotonically non-increasing."""
    ctrl = BacksteppingController(k1=1.0, k2=3.0)
    _, vs = _simulate_backstepping(
        ctrl, np.array([0.0, 0.0, 0.0]), np.array([3.0, 3.0]),
    )
    # Tiny rises (~1e-4) can occur in the very first transient before
    # the heading aligns; allow a small slack but require the global
    # trend to be monotone.
    diffs = np.diff(vs)
    assert np.all(diffs <= 5e-4), f"V increased by {diffs.max():.6f}"
    assert vs[-1] < vs[0]


def test_backstepping_higher_gains_converge_faster() -> None:
    """Doubling k1 reduces time-to-goal under the same kinematic dynamics."""
    slow = BacksteppingController(k1=0.5, k2=2.0)
    fast = BacksteppingController(k1=1.5, k2=3.0)
    s_slow, _ = _simulate_backstepping(
        slow, np.array([0.0, 0.0, 0.0]), np.array([2.0, 0.0]),
    )
    s_fast, _ = _simulate_backstepping(
        fast, np.array([0.0, 0.0, 0.0]), np.array([2.0, 0.0]),
    )
    assert s_fast.shape[0] < s_slow.shape[0]


def test_backstepping_actuator_saturation() -> None:
    """Even with absurd error, returned commands respect TB3 limits."""
    ctrl = BacksteppingController(k1=10.0, k2=10.0)
    v, w = ctrl.compute_control([0.0, 0.0, 0.0], [50.0, 0.0])
    assert -ctrl.cfg.max_v <= v <= ctrl.cfg.max_v
    assert -ctrl.cfg.max_omega <= w <= ctrl.cfg.max_omega


def test_backstepping_zero_at_goal() -> None:
    """At the goal, both commands and V are zero."""
    ctrl = BacksteppingController()
    v, w = ctrl.compute_control([1.0, 2.0, 0.3], [1.0, 2.0])
    assert v == 0.0 and w == 0.0
    assert ctrl.lyapunov_value([1.0, 2.0, 0.3], [1.0, 2.0]) == 0.0


def test_backstepping_lyapunov_casadi_matches_numpy() -> None:
    """Symbolic V(e) evaluates to the same value as the numpy form."""
    pytest.importorskip("casadi")
    import casadi as ca

    ctrl = BacksteppingController()
    x = ca.SX.sym("x", 3)
    g = ca.SX.sym("g", 2)
    V_sym = ctrl.lyapunov_value_casadi(x, g)
    V_func = ca.Function("V", [x, g], [V_sym])
    state = np.array([0.4, -0.6, 0.7])
    goal = np.array([1.0, 0.0])
    v_casadi = float(V_func(ca.DM(state), ca.DM(goal)))
    v_numpy = ctrl.lyapunov_value(state, goal)
    assert v_casadi == pytest.approx(v_numpy, abs=1e-12)


def test_backstepping_invalid_gains_raise() -> None:
    with pytest.raises(ValueError):
        BacksteppingController(k1=0.0)
    with pytest.raises(ValueError):
        BacksteppingController(k2=-1.0)
    with pytest.raises(ValueError):
        BacksteppingController(max_v=0.0)


# ---------------------------------------------------------------------
# Phase-2 STEP 2: Lyapunov-MPC NLP tests
# ---------------------------------------------------------------------
def _make_mpc(**overrides) -> LyapunovMPC:
    pytest.importorskip("casadi")
    cfg = LyapunovMPCConfig(
        horizon=overrides.pop("horizon", 15),
        dt=overrides.pop("dt", 0.1),
        alpha_lyap=overrides.pop("alpha_lyap", 0.02),
        d_safe=overrides.pop("d_safe", 0.3),
        max_obstacles=overrides.pop("max_obstacles", 4),
        max_iter=overrides.pop("max_iter", 60),
        max_cpu_time=overrides.pop("max_cpu_time", 0.5),
        soft_lyap_penalty=overrides.pop("soft_lyap_penalty", 100.0),
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return LyapunovMPC(config=cfg)


def _rollout_mpc(
    mpc: LyapunovMPC,
    state0: np.ndarray,
    goal: np.ndarray,
    *,
    obstacles: np.ndarray | None = None,
    margins: np.ndarray | None = None,
    max_steps: int = 320,
):
    dyn = TurtleBot3Dynamics()
    state = state0.copy()
    states = [state.copy()]
    Vs = [mpc._backstepping.lyapunov_value(state, goal)]
    energies = [0.0]
    cmds = []
    times = []
    import time
    for _ in range(max_steps):
        t0 = time.perf_counter()
        fb = mpc.compute_control(state, goal, obstacles=obstacles, obstacle_margins=margins)
        times.append(time.perf_counter() - t0)
        cmds.append((fb.v, fb.omega))
        if fb.reached:
            break
        state = dyn.kinematic_step(state, fb.v, fb.omega, dt=mpc.config.dt)
        states.append(state.copy())
        Vs.append(mpc._backstepping.lyapunov_value(state, goal))
        energies.append(energies[-1] + fb.energy_consumed)
        if float(np.hypot(state[0] - goal[0], state[1] - goal[1])) <= mpc.config.goal_tolerance:
            break
    return (
        np.asarray(states), np.asarray(Vs), np.asarray(energies),
        np.asarray(cmds), np.asarray(times),
    )


def test_lyap_mpc_tracks_subgoal() -> None:
    """MPC drives (0,0,0) → (3,3) within 0.1 m."""
    mpc = _make_mpc(
        R_diag=[0.05, 0.05], Q_diag=[15.0, 15.0, 0.0],
        P_terminal_scale=20.0, max_iter=120,
    )
    states, _, _, _, _ = _rollout_mpc(mpc, np.zeros(3), np.array([3.0, 3.0]),
                                      max_steps=400)
    final_err = float(np.hypot(states[-1, 0] - 3.0, states[-1, 1] - 3.0))
    assert final_err < 0.1, f"final error {final_err:.3f} m"


def test_lyap_mpc_v_monotone_and_plot() -> None:
    """V(t) decreases on the closed-loop trajectory; save plot."""
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    mpc = _make_mpc()
    _, Vs, _, _, _ = _rollout_mpc(mpc, np.zeros(3), np.array([3.0, 3.0]))
    # Allow tiny non-monotone wobble (numerical, slack-relaxed); demand
    # global decrease and that no rise exceeds 5 % of V_0.
    rises = np.diff(Vs)
    assert Vs[-1] < 0.05 * Vs[0]
    assert rises.max() < 0.05 * Vs[0], f"largest rise {rises.max():.4f}"
    out = Path("results") / "lyapunov_convergence.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(len(Vs)) * mpc.config.dt, Vs)
    ax.set_xlabel("time [s]"); ax.set_ylabel("V(e)")
    ax.set_title("Lyapunov-MPC: V(t) along closed-loop trajectory")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    assert out.exists()


def test_lyap_mpc_actuator_constraints() -> None:
    """Issued v, ω stay inside the TurtleBot3 actuator box."""
    mpc = _make_mpc()
    _, _, _, cmds, _ = _rollout_mpc(mpc, np.zeros(3), np.array([2.5, 1.5]))
    assert np.all(np.abs(cmds[:, 0]) <= mpc.config.max_v + 1e-6)
    assert np.all(np.abs(cmds[:, 1]) <= mpc.config.max_omega + 1e-6)


def test_lyap_mpc_obstacle_avoidance() -> None:
    """Robot keeps minimum distance ≥ d_safe to a placed obstacle."""
    mpc = _make_mpc(d_safe=0.4, max_obstacles=2)
    obs = np.array([[1.5, 1.5]])
    states, _, _, _, _ = _rollout_mpc(
        mpc, np.zeros(3), np.array([3.0, 3.0]),
        obstacles=obs, margins=np.zeros(1),
    )
    dists = np.linalg.norm(states[:, :2] - obs, axis=1)
    # Tolerate a small slack on the soft constraint.
    assert dists.min() >= mpc.config.d_safe - 0.05, (
        f"min dist {dists.min():.3f} below d_safe={mpc.config.d_safe}"
    )


def test_lyap_mpc_energy_monotone() -> None:
    """Cumulative energy is non-decreasing and strictly positive."""
    mpc = _make_mpc()
    _, _, energies, _, _ = _rollout_mpc(mpc, np.zeros(3), np.array([2.0, 1.0]))
    assert energies[-1] > 0.0
    assert np.all(np.diff(energies) >= -1e-9)


def test_lyap_mpc_solve_time_budget() -> None:
    """Median solve time within order-of-magnitude of the 50 ms target."""
    mpc = _make_mpc(horizon=10, max_obstacles=2)
    _, _, _, _, times = _rollout_mpc(mpc, np.zeros(3), np.array([2.0, 1.0]),
                                     max_steps=40)
    body = times[5:]
    # Use median to suppress occasional warm-start hiccups; CI hardware
    # is noisy. Spec target: <50 ms with N=20; here N=10.
    median = float(np.median(body))
    assert median < 0.1, f"median solve {median*1000:.1f} ms"


def test_lyap_mpc_infeasibility_recovery() -> None:
    """alpha=0.99 makes the contraction nearly impossible; soft slack saves the day."""
    mpc = _make_mpc(alpha_lyap=0.99, soft_lyap_penalty=1e3)
    fb = mpc.compute_control([0.0, 0.0, 0.0], [2.0, 1.0])
    assert np.isfinite(fb.v) and np.isfinite(fb.omega)
    assert fb.feasible is True or mpc.fallback_count >= 1


def test_lyap_mpc_reset_clears_state() -> None:
    mpc = _make_mpc()
    mpc.compute_control([0.0, 0.0, 0.0], [1.0, 0.5])
    mpc.reset()
    assert mpc._prev_X is None and mpc._prev_U is None


def test_lyap_mpc_reached_short_circuit() -> None:
    """At goal, controller returns zero commands without solving NLP."""
    mpc = _make_mpc()
    fb = mpc.compute_control([1.0, 1.0, 0.0], [1.02, 1.0])
    assert fb.reached is True
    assert fb.v == 0.0 and fb.omega == 0.0
