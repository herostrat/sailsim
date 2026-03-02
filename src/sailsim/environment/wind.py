"""Wind environment models.

Provides constant, gust (Ornstein-Uhlenbeck), and shifting wind models.
All models implement the same interface: get(t) -> WindState.
"""

from __future__ import annotations

import numpy as np

from sailsim.core.types import WindState


class ConstantWind:
    """Constant true wind (no gusts or variation)."""

    def __init__(self, speed: float, direction: float) -> None:
        """
        Args:
            speed: true wind speed [m/s]
            direction: direction wind comes FROM [rad] (NED, 0=North)
        """
        self._state = WindState(speed=speed, direction=direction)

    def get(self, t: float) -> WindState:
        """Return wind state at time t (constant, ignores t)."""
        return self._state


class GustWind:
    """Wind with gusts modeled as an Ornstein-Uhlenbeck process.

    The wind speed fluctuates around a base value with mean-reverting
    noise. The direction remains constant.

    The OU process is: dX = -theta*(X - mu)*dt + sigma*dW
    where theta = 1/tau (mean reversion rate).
    """

    def __init__(
        self,
        base_speed: float,
        direction: float,
        gust_intensity: float = 2.0,
        gust_tau: float = 10.0,
        seed: int | None = None,
    ) -> None:
        """
        Args:
            base_speed: mean wind speed [m/s]
            direction: wind direction [rad] (constant)
            gust_intensity: std deviation of speed fluctuations [m/s]
            gust_tau: time constant for mean reversion [s]
            seed: random seed for reproducibility
        """
        self._base_speed = base_speed
        self._direction = direction
        self._intensity = gust_intensity
        self._tau = gust_tau
        self._rng = np.random.default_rng(seed)

        # Pre-generate the process at fine resolution for deterministic lookup
        self._dt_internal = 0.01  # 100 Hz internal resolution
        self._duration = 0.0
        self._speeds: list[float] = []
        self._current_speed = base_speed

    def _ensure_generated(self, t: float) -> None:
        """Extend pre-generated time series to cover time t."""
        if t <= self._duration and self._speeds:
            return

        target = t + 10.0  # generate ahead
        dt = self._dt_internal
        theta = 1.0 / self._tau
        sigma = self._intensity * np.sqrt(2.0 * theta)

        speed = self._current_speed
        if not self._speeds:
            self._speeds.append(speed)
            self._duration = 0.0

        while self._duration < target:
            dw = self._rng.normal() * np.sqrt(dt)
            speed += -theta * (speed - self._base_speed) * dt + sigma * dw
            speed = max(0.0, speed)  # wind speed cannot be negative
            self._speeds.append(speed)
            self._duration += dt

        self._current_speed = speed

    def get(self, t: float) -> WindState:
        """Return wind state at time t."""
        t = max(0.0, t)
        self._ensure_generated(t)

        # Linear interpolation between pre-generated samples
        idx = t / self._dt_internal
        i = int(idx)
        i = min(i, len(self._speeds) - 2)
        frac = idx - i
        speed = self._speeds[i] * (1.0 - frac) + self._speeds[i + 1] * frac

        return WindState(speed=float(speed), direction=self._direction)


class ShiftingWind:
    """Wind with gradually changing direction.

    Supports two modes:
    - "linear": direction changes at a constant rate [rad/s]
    - "sinusoidal": direction oscillates around base with given amplitude and period
    """

    def __init__(
        self,
        speed: float,
        base_direction: float,
        mode: str = "sinusoidal",
        rate: float = 0.0,
        amplitude: float = 0.0,
        period: float = 300.0,
    ) -> None:
        """
        Args:
            speed: constant wind speed [m/s]
            base_direction: base direction [rad]
            mode: "linear" or "sinusoidal"
            rate: direction change rate [rad/s] (linear mode)
            amplitude: oscillation amplitude [rad] (sinusoidal mode)
            period: oscillation period [s] (sinusoidal mode)
        """
        self._speed = speed
        self._base_direction = base_direction
        self._mode = mode
        self._rate = rate
        self._amplitude = amplitude
        self._period = period

    def get(self, t: float) -> WindState:
        """Return wind state at time t."""
        if self._mode == "linear":
            direction = self._base_direction + self._rate * t
        else:  # sinusoidal
            direction = self._base_direction + self._amplitude * np.sin(
                2.0 * np.pi * t / self._period
            )

        # Normalize to [-pi, pi]
        direction = float(np.arctan2(np.sin(direction), np.cos(direction)))
        return WindState(speed=self._speed, direction=direction)
