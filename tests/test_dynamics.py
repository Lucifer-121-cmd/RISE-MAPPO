"""Sanity tests for :mod:`envs.robot_dynamics`."""
from __future__ import annotations

import numpy as np
import pytest

from envs.robot_dynamics import (
    TurtleBot3Dynamics,
    TurtleBot3Params,
    normalize_angle,
)


def test_normalize_angle_basic() -> None:
    assert np.isclose(normalize_angle(0.0), 0.0)
    # Boundary at ±pi may map to either ±pi; both represent the same point.
    assert np.isclose(abs(normalize_angle(np.pi)), np.pi)
    assert np.isclose(normalize_angle(-np.pi + 1e-9), -np.pi + 1e-9, atol=1e-6)
    assert np.isclose(abs(normalize_angle(3 * np.pi)), np.pi)
    assert np.isclose(abs(normalize_angle(-3 * np.pi)), np.pi)
    # Range check on a sweep.
    sweep = np.linspace(-10.0, 10.0, 101)
    out = normalize_angle(sweep)
    assert np.all(out > -np.pi - 1e-9) and np.all(out <= np.pi + 1e-9)


def test_forward_drive_kinematic() -> None:
    """v = 0.1 m/s straight for 10 s ≈ 1 m of travel."""
    dyn = TurtleBot3Dynamics()
    state = np.zeros(3)
    dt = 0.05
    steps = int(10.0 / dt)
    for _ in range(steps):
        state = dyn.kinematic_step(state, v=0.1, omega=0.0, dt=dt)
    assert state[0] == pytest.approx(1.0, abs=1e-2)
    assert abs(state[1]) < 1e-9
    assert abs(state[2]) < 1e-9


def test_pure_rotation_kinematic() -> None:
    """omega = 1.0 for pi seconds → ~pi rad rotation."""
    dyn = TurtleBot3Dynamics()
    state = np.zeros(3)
    dt = 0.01
    steps = int(np.pi / dt)
    for _ in range(steps):
        state = dyn.kinematic_step(state, v=0.0, omega=1.0, dt=dt)
    assert abs(state[0]) < 1e-9
    assert abs(state[1]) < 1e-9
    # Wraps to ~+pi or ~-pi depending on rounding; both are acceptable.
    assert abs(abs(state[2]) - np.pi) < 5e-2


def test_actuator_saturation() -> None:
    """Commands above the limit must be clipped, not amplified."""
    dyn = TurtleBot3Dynamics()
    state = np.zeros(3)
    # Forward at 0.5 m/s (over-saturated) for 1 s should produce
    # at most 0.22 m of travel (clipped to max_linear_velocity).
    dt = 0.05
    for _ in range(int(1.0 / dt)):
        state = dyn.kinematic_step(state, v=0.5, omega=0.0, dt=dt)
    assert state[0] == pytest.approx(0.22, abs=1e-3)


def test_dynamic_step_lag_then_settle() -> None:
    """First-order lag: realised v approaches command after several tau."""
    params = TurtleBot3Params(linear_time_constant=0.2)
    dyn = TurtleBot3Dynamics(params=params)
    state = np.zeros(3)
    dt = 0.02
    # Drive long enough (5*tau) that filter is essentially settled.
    for _ in range(int(2.0 / dt)):
        state = dyn.dynamic_step(state, v_cmd=0.1, omega_cmd=0.0, dt=dt)
    # Expected travel ≈ 0.1 * (2.0 - tau) with one-tau ramp-up loss.
    assert state[0] > 0.15  # at least most of nominal travel realised
    assert state[0] < 0.25


def test_circle_trajectory_kinematic() -> None:
    """Constant v and omega should produce a circle of radius v/omega."""
    dyn = TurtleBot3Dynamics()
    state = np.zeros(3)
    dt = 0.01
    v = 0.1
    omega = 0.5
    radius = v / omega
    pts = []
    for _ in range(int(2 * np.pi / omega / dt)):
        state = dyn.kinematic_step(state, v=v, omega=omega, dt=dt)
        pts.append(state[:2].copy())
    pts = np.asarray(pts)
    centre = np.array([0.0, radius])
    radii = np.linalg.norm(pts - centre, axis=1)
    assert radii.std() < 5e-3


def test_casadi_export() -> None:
    """to_casadi() returns a callable function consistent with kinematic_step."""
    pytest.importorskip("casadi")
    import casadi as ca

    dyn = TurtleBot3Dynamics()
    _x, _u, _rhs, f = dyn.to_casadi()
    # Numerical evaluation of f at (state, u) matches numpy formulas.
    x_val = np.array([0.0, 0.0, 0.5])
    u_val = np.array([0.1, 0.2])
    out = np.asarray(f(ca.DM(x_val), ca.DM(u_val))).reshape(-1)
    expected = np.array([
        u_val[0] * np.cos(x_val[2]),
        u_val[0] * np.sin(x_val[2]),
        u_val[1],
    ])
    np.testing.assert_allclose(out, expected, atol=1e-10)


def test_state_shape_validation() -> None:
    dyn = TurtleBot3Dynamics()
    with pytest.raises(ValueError):
        dyn.kinematic_step(np.zeros(2), v=0.0, omega=0.0, dt=0.1)
    with pytest.raises(ValueError):
        dyn.kinematic_step(np.zeros(3), v=0.0, omega=0.0, dt=0.0)
