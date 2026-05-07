"""TurtleBot3 Burger dynamics.

Provides a single class :class:`TurtleBot3Dynamics` that exposes both a
purely kinematic step and a first-order dynamic step (with linear and
angular inertial lag). The same parameters are reused inside the
Lyapunov-MPC formulation; a CasADi symbolic export is exposed via
:meth:`TurtleBot3Dynamics.to_casadi` so the MPC layer in Phase 2 picks up
the identical model.

State convention
----------------
``state = [x, y, theta]`` with ``theta`` in radians, wrapped to
``(-pi, pi]`` after every integration step.

Control input
-------------
``u = [v, omega]`` (linear m/s, angular rad/s); both are saturated to the
TurtleBot3 Burger physical limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


def normalize_angle(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap ``angle`` (rad) into the half-open interval ``(-pi, pi]``."""
    wrapped = (angle + np.pi) % (2.0 * np.pi) - np.pi
    # Map the open boundary -pi back to +pi so the range is (-pi, pi].
    if np.isscalar(angle):
        return float(np.pi) if wrapped <= -np.pi + 1e-12 else float(wrapped)
    return np.where(wrapped <= -np.pi + 1e-12, np.pi, wrapped)


@dataclass
class TurtleBot3Params:
    """Physical parameters of the TurtleBot3 Burger.

    Defaults match the values in the Paper 3 master prompt and the
    Robotis specification sheet for the Burger platform.
    """

    max_linear_velocity: float = 0.22       # m/s
    max_angular_velocity: float = 2.84      # rad/s
    wheel_radius: float = 0.033             # m
    wheel_separation: float = 0.160         # m
    mass: float = 1.0                       # kg (with battery, approximate)
    moment_of_inertia: float = 8.7e-3       # kg·m^2 (Burger, approximate)
    linear_time_constant: float = 0.20      # s, first-order velocity lag
    angular_time_constant: float = 0.10     # s, first-order rate lag
    process_noise_std: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0])
    )

    def saturate(self, v: float, omega: float) -> Tuple[float, float]:
        """Clip ``(v, omega)`` to actuator limits."""
        v_clipped = float(np.clip(v, -self.max_linear_velocity, self.max_linear_velocity))
        w_clipped = float(np.clip(omega, -self.max_angular_velocity, self.max_angular_velocity))
        return v_clipped, w_clipped


class TurtleBot3Dynamics:
    """Differential-drive dynamics for the TurtleBot3 Burger.

    Two integration modes are provided:

    * ``kinematic_step`` — ideal unicycle model; ``u`` is applied
      instantaneously.
    * ``dynamic_step`` — first-order lag on linear and angular
      velocities, modelling motor inertia.

    Both modes optionally inject Gaussian process noise on the state to
    emulate the sim-to-real gap.
    """

    def __init__(
        self,
        params: Optional[TurtleBot3Params] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.params = params or TurtleBot3Params()
        self._rng = rng if rng is not None else np.random.default_rng()
        # Internal velocity state used by ``dynamic_step`` only.
        self._v_filt: float = 0.0
        self._omega_filt: float = 0.0

    def reset_filter(self) -> None:
        """Reset internal velocity filter state to zero."""
        self._v_filt = 0.0
        self._omega_filt = 0.0

    @staticmethod
    def _ensure_state(state: np.ndarray) -> np.ndarray:
        s = np.asarray(state, dtype=float).reshape(-1)
        if s.shape != (3,):
            raise ValueError(f"state must have shape (3,), got {s.shape}")
        return s

    def kinematic_step(
        self,
        state: np.ndarray,
        v: float,
        omega: float,
        dt: float,
        add_noise: bool = False,
    ) -> np.ndarray:
        """Apply one ideal-unicycle Euler step.

        Parameters
        ----------
        state : array-like, shape (3,)
            Current ``[x, y, theta]`` state.
        v, omega : float
            Commanded linear/angular velocity. Saturated internally.
        dt : float
            Integration step (s). Must be positive.
        add_noise : bool, default False
            If True, inject zero-mean Gaussian process noise scaled by
            :attr:`TurtleBot3Params.process_noise_std`.
        """
        if dt <= 0.0:
            raise ValueError(f"dt must be > 0, got {dt}")
        s = self._ensure_state(state)
        v_c, w_c = self.params.saturate(v, omega)
        x, y, th = s
        x_new = x + v_c * np.cos(th) * dt
        y_new = y + v_c * np.sin(th) * dt
        th_new = normalize_angle(th + w_c * dt)
        new_state = np.array([x_new, y_new, th_new])
        if add_noise:
            new_state = new_state + self._rng.normal(0.0, 1.0, size=3) * self.params.process_noise_std
            new_state[2] = normalize_angle(new_state[2])
        return new_state

    def dynamic_step(
        self,
        state: np.ndarray,
        v_cmd: float,
        omega_cmd: float,
        dt: float,
        add_noise: bool = False,
    ) -> np.ndarray:
        """Apply one Euler step with first-order velocity lag.

        Models actuator inertia: the realised velocities chase the
        commanded ones with time constants
        :attr:`TurtleBot3Params.linear_time_constant` and
        :attr:`TurtleBot3Params.angular_time_constant`.
        """
        if dt <= 0.0:
            raise ValueError(f"dt must be > 0, got {dt}")
        v_cmd_c, w_cmd_c = self.params.saturate(v_cmd, omega_cmd)
        tau_v = max(self.params.linear_time_constant, 1e-6)
        tau_w = max(self.params.angular_time_constant, 1e-6)
        # Continuous-time first-order lag → exact discretisation.
        a_v = float(np.exp(-dt / tau_v))
        a_w = float(np.exp(-dt / tau_w))
        self._v_filt = a_v * self._v_filt + (1.0 - a_v) * v_cmd_c
        self._omega_filt = a_w * self._omega_filt + (1.0 - a_w) * w_cmd_c
        return self.kinematic_step(state, self._v_filt, self._omega_filt, dt, add_noise=add_noise)

    def to_casadi(self):
        """Return symbolic CasADi dynamics for use inside Lyapunov-MPC.

        Returns a tuple ``(state_sym, input_sym, rhs_sym, f_func)`` where
        ``rhs_sym`` is the continuous-time RHS ``\\dot{x} = f(x, u)`` and
        ``f_func`` is a CasADi function with signature
        ``f(x, u) -> dx``. Discretisation is left to the MPC layer (it
        chooses RK4 or Euler depending on horizon length).
        """
        try:
            import casadi as ca  # local import: optional dependency at runtime
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "CasADi is required for to_casadi(); pip install casadi"
            ) from exc

        x = ca.SX.sym("x", 3)            # [x, y, theta]
        u = ca.SX.sym("u", 2)            # [v, omega]
        dx = ca.vertcat(
            u[0] * ca.cos(x[2]),
            u[0] * ca.sin(x[2]),
            u[1],
        )
        f_func = ca.Function("tb3_kin", [x, u], [dx])
        return x, u, dx, f_func

    @property
    def state_dim(self) -> int:
        return 3

    @property
    def control_dim(self) -> int:
        return 2

    def control_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(u_min, u_max)`` arrays for actuator saturation."""
        u_max = np.array([self.params.max_linear_velocity, self.params.max_angular_velocity])
        return -u_max, u_max
