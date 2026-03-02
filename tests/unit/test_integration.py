"""Test RK4 integrator against known analytical solutions."""

import numpy as np

from sailsim.physics.integration import rk4_step


def test_rk4_constant_derivative():
    """Constant derivative f(t,y) = 1 => y(t) = y0 + t."""
    state = np.array([0.0])
    for _ in range(100):
        state = rk4_step(lambda t, y: np.array([1.0]), 0.0, state, 0.01)
    assert abs(state[0] - 1.0) < 1e-10


def test_rk4_exponential_decay():
    """dy/dt = -y => y(t) = y0 * exp(-t). Start at y=1."""
    state = np.array([1.0])
    dt = 0.01
    t = 0.0
    for _ in range(1000):
        state = rk4_step(lambda t, y: -y, t, state, dt)
        t += dt
    expected = np.exp(-10.0)
    assert abs(state[0] - expected) < 1e-8


def test_rk4_harmonic_oscillator():
    """d²x/dt² = -x => x(t) = cos(t), v(t) = -sin(t).
    State = [x, v], derivatives = [v, -x].
    """
    state = np.array([1.0, 0.0])  # x=1, v=0
    dt = 0.001
    t = 0.0

    def derivatives(t, s):
        return np.array([s[1], -s[0]])

    for _ in range(int(2 * np.pi / dt)):
        state = rk4_step(derivatives, t, state, dt)
        t += dt

    # After one full period, should be back to [1, 0]
    # Tolerance accounts for accumulated numerical error over ~6283 steps
    assert abs(state[0] - 1.0) < 1e-3
    assert abs(state[1] - 0.0) < 1e-3


def test_rk4_vector_system():
    """Test with a 6-element state vector (like 6-DOF)."""
    state = np.zeros(6)
    state[0] = 1.0

    def derivatives(t, s):
        d = np.zeros(6)
        d[0] = -s[0]  # exponential decay
        d[5] = 0.1  # constant yaw rate
        return d

    dt = 0.01
    t = 0.0
    for _ in range(100):
        state = rk4_step(derivatives, t, state, dt)
        t += dt

    assert abs(state[0] - np.exp(-1.0)) < 1e-6
    assert abs(state[5] - 0.1) < 1e-6  # yaw = 0.1 * 1s
