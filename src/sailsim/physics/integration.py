"""Numerical integration methods for ODE solving."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

# Type alias for the derivative function: f(t, state) -> d_state/dt
DerivativeFunc = Callable[[float, NDArray[np.float64]], NDArray[np.float64]]


def rk4_step(
    f: DerivativeFunc,
    t: float,
    state: NDArray[np.float64],
    dt: float,
) -> NDArray[np.float64]:
    """Single step of the 4th-order Runge-Kutta integrator.

    Args:
        f: derivative function f(t, state) -> d_state/dt
        t: current time [s]
        state: current state vector
        dt: time step [s]

    Returns:
        New state vector after one time step.
    """
    k1 = f(t, state)
    k2 = f(t + 0.5 * dt, state + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, state + 0.5 * dt * k2)
    k4 = f(t + dt, state + dt * k3)

    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
