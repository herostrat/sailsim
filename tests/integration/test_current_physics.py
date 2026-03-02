"""Integration tests for ocean current physics.

Verifies that current affects position (GPS/SOG) but not water-relative dynamics.
"""

from __future__ import annotations

import numpy as np

from sailsim.core.types import ControlCommand, CurrentState, VesselState, WindState
from sailsim.vessel.yacht_3dof import Yacht3DOF


class TestCurrentPhysics:
    """Verify Fossen-correct current handling: kinematics only."""

    def test_boat_drifts_with_current(self):
        """A stationary boat in current should drift in current direction."""
        yacht = Yacht3DOF()
        yacht.reset(VesselState())

        wind = WindState(speed=0.0, direction=0.0)
        control = ControlCommand()
        current = CurrentState(speed=1.0, direction=0.0)  # 1 m/s north

        # Step for 10 seconds
        for _ in range(200):
            yacht.step(wind, control, 0.05, current=current)

        # Should have drifted north (~10m) with no sway
        assert yacht.state.x > 8.0, f"Expected northward drift, got x={yacht.state.x:.2f}"
        assert abs(yacht.state.y) < 1.0

    def test_stw_not_affected_by_current(self):
        """Speed through water should be independent of current."""
        yacht_no_current = Yacht3DOF()
        yacht_with_current = Yacht3DOF()

        wind = WindState(speed=5.0, direction=1.047)
        control = ControlCommand()
        current = CurrentState(speed=1.0, direction=np.pi / 2)

        # Run both for 5 seconds
        for _ in range(100):
            yacht_no_current.step(wind, control, 0.05)
            yacht_with_current.step(wind, control, 0.05, current=current)

        # STW should be similar (not identical due to position-dependent effects,
        # but current shouldn't directly change water-relative velocity)
        stw_no = yacht_no_current.state.speed
        stw_with = yacht_with_current.state.speed
        assert abs(stw_no - stw_with) < 0.5, f"STW differs too much: {stw_no:.3f} vs {stw_with:.3f}"

    def test_sog_differs_from_stw(self):
        """SOG should differ from STW when current is present."""
        from sailsim.core.types import SensorData

        yacht = Yacht3DOF()
        wind = WindState(speed=5.0, direction=1.047)
        current = CurrentState(speed=1.0, direction=0.0)
        control = ControlCommand()

        # Let the boat build some speed
        for _ in range(400):
            yacht.step(wind, control, 0.05, current=current)

        sensors = SensorData.from_state(yacht.state, wind, current)
        sensors_no_current = SensorData.from_state(yacht.state, wind)

        # SOG with current should differ from SOG without
        assert abs(sensors.speed_over_ground - sensors_no_current.speed_over_ground) > 0.5

    def test_no_current_backward_compatible(self):
        """step() without current argument should work identically to before."""
        yacht1 = Yacht3DOF()
        yacht2 = Yacht3DOF()

        wind = WindState(speed=5.0, direction=1.047)
        control = ControlCommand()

        for _ in range(100):
            yacht1.step(wind, control, 0.05)
            yacht2.step(wind, control, 0.05, current=None)

        np.testing.assert_array_almost_equal(yacht1.state.eta, yacht2.state.eta)
        np.testing.assert_array_almost_equal(yacht1.state.nu, yacht2.state.nu)
