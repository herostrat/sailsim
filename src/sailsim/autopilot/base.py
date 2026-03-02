"""Autopilot protocol interface.

Uses Python Protocol (structural subtyping) so any class with a matching
`compute` method works as an autopilot - no inheritance required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from sailsim.core.types import ControlCommand, SensorData


class AutopilotProtocol(Protocol):
    """Interface that all autopilots must satisfy."""

    def compute(self, sensors: SensorData, dt: float) -> ControlCommand:
        """Compute control command from sensor data.

        Args:
            sensors: current sensor readings
            dt: time since last call [s]

        Returns:
            Control command for the vessel.
        """
        ...

    def set_target_heading(self, heading: float) -> None:
        """Set the desired heading [rad]."""
        ...
