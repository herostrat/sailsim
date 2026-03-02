"""Fixed rudder autopilot for simulator validation.

Outputs a constant rudder angle regardless of sensor input.
Useful for testing physics behavior without controller interference.
"""

from __future__ import annotations

from sailsim.core.types import ControlCommand, SensorData


class FixedRudderAutopilot:
    """Always outputs the same rudder angle. No control logic."""

    def __init__(self, rudder_angle: float = 0.0) -> None:
        """
        Args:
            rudder_angle: fixed rudder deflection [rad]
        """
        self._rudder_angle = rudder_angle

    def set_target_heading(self, heading: float) -> None:
        """No-op: fixed rudder ignores heading targets."""
        pass

    def compute(self, sensors: SensorData, dt: float) -> ControlCommand:
        return ControlCommand(rudder_angle=self._rudder_angle)
