"""PID heading autopilot with anti-windup.

A simple but effective heading controller that commands rudder angle
based on heading error, its integral, and its derivative (yaw rate).
"""

from __future__ import annotations

import numpy as np

from sailsim.core.types import ControlCommand, SensorData


class PIDAutopilot:
    """PID heading controller with anti-windup clamping."""

    def __init__(
        self,
        kp: float = 1.0,
        ki: float = 0.05,
        kd: float = 0.8,
        rudder_max: float = 0.52,  # ~30 degrees
        integral_max: float = 1.0,  # anti-windup clamp [rad]
        auto_sail_trim: bool = False,
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.rudder_max = rudder_max
        self.integral_max = integral_max
        self.auto_sail_trim = auto_sail_trim

        self.target_heading: float = 0.0
        self._integral: float = 0.0

    def set_target_heading(self, heading: float) -> None:
        """Set desired heading [rad]."""
        self.target_heading = heading
        self._integral = 0.0  # reset integral on target change

    def compute(self, sensors: SensorData, dt: float) -> ControlCommand:
        """Compute rudder command from heading error."""
        # Heading error with proper wrapping to [-pi, pi]
        error = self.target_heading - sensors.heading
        error = np.arctan2(np.sin(error), np.cos(error))

        # Integral with anti-windup
        self._integral += error * dt
        self._integral = np.clip(
            self._integral,
            -self.integral_max,
            self.integral_max,
        )

        # Derivative (use measured yaw rate, not numerical differentiation)
        derivative = -sensors.yaw_rate  # negative because yaw rate opposes error reduction

        # PID output
        rudder = self.kp * error + self.ki * self._integral + self.kd * derivative

        # Clamp to physical limits
        rudder = np.clip(rudder, -self.rudder_max, self.rudder_max)

        trim = 0.5
        if self.auto_sail_trim:
            from sailsim.physics.aerodynamics import optimal_sail_trim

            trim = optimal_sail_trim(sensors.apparent_wind_angle)

        return ControlCommand(rudder_angle=float(rudder), sail_trim=trim)
