"""Ocean current environment models.

All models implement: get(t) -> CurrentState.
"""

from __future__ import annotations

import numpy as np

from sailsim.core.types import CurrentState


class NoCurrent:
    """Null-object: no ocean current."""

    def get(self, t: float) -> CurrentState:
        return CurrentState(speed=0.0, direction=0.0)


class ConstantCurrent:
    """Constant ocean current."""

    def __init__(self, speed: float, direction: float) -> None:
        """
        Args:
            speed: current speed [m/s]
            direction: direction current flows TOWARDS [rad] in NED
        """
        self._state = CurrentState(speed=speed, direction=direction)

    def get(self, t: float) -> CurrentState:
        return self._state


class TidalCurrent:
    """Sinusoidally varying tidal current.

    Speed oscillates around base_speed with given amplitude and period.
    Direction remains constant.
    """

    def __init__(
        self,
        base_speed: float,
        amplitude: float,
        period: float,
        direction: float,
        phase: float = 0.0,
    ) -> None:
        """
        Args:
            base_speed: mean current speed [m/s]
            amplitude: speed oscillation amplitude [m/s]
            period: oscillation period [s]
            direction: constant flow direction [rad]
            phase: initial phase offset [rad]
        """
        self._base_speed = base_speed
        self._amplitude = amplitude
        self._period = period
        self._direction = direction
        self._phase = phase

    def get(self, t: float) -> CurrentState:
        speed = self._base_speed + self._amplitude * np.sin(
            2.0 * np.pi * t / self._period + self._phase
        )
        speed = max(0.0, speed)
        return CurrentState(speed=float(speed), direction=self._direction)
