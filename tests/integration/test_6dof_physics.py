"""Integration tests for 6-DOF yacht physics."""

from __future__ import annotations

import numpy as np

from sailsim.core.types import ControlCommand, VesselState, WindState
from sailsim.vessel.yacht_6dof import Yacht6DOF


class TestYacht6DOFBasics:
    def test_no_nan(self):
        """Simulation should not produce NaN values."""
        yacht = Yacht6DOF()
        wind = WindState(speed=5.0, direction=1.047)
        control = ControlCommand()

        for _ in range(200):
            yacht.step(wind, control, 0.05)

        assert not np.any(np.isnan(yacht.state.eta))
        assert not np.any(np.isnan(yacht.state.nu))

    def test_heeling_with_side_wind(self):
        """Side wind should cause heel (roll)."""
        yacht = Yacht6DOF()
        yacht.reset(
            VesselState(
                eta=np.zeros(6),
                nu=np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            )
        )
        # Beam reach from starboard
        wind = WindState(speed=8.0, direction=np.pi / 2)
        control = ControlCommand()

        for _ in range(400):
            yacht.step(wind, control, 0.05)

        roll = yacht.state.phi
        # Wind from starboard → heel to port (negative roll convention depends on model)
        assert abs(roll) > 0.01, f"Expected heel, got roll={np.degrees(roll):.2f}°"

    def test_roll_stability(self):
        """Released from a heel angle, boat should oscillate and settle."""
        yacht = Yacht6DOF()
        # Start heeled 15°
        initial_eta = np.zeros(6)
        initial_eta[3] = np.radians(15)
        yacht.reset(VesselState(eta=initial_eta, nu=np.zeros(6)))

        wind = WindState(speed=0.0, direction=0.0)
        control = ControlCommand()

        rolls = []
        for _ in range(2000):
            yacht.step(wind, control, 0.05)
            rolls.append(np.degrees(yacht.state.phi))

        # Roll should decrease over time (damped oscillation)
        max_early = max(abs(r) for r in rolls[:200])
        max_late = max(abs(r) for r in rolls[1500:])
        assert max_late < max_early, (
            f"Roll should dampen: early max={max_early:.1f}°, late max={max_late:.1f}°"
        )
        # Final roll should be small
        assert abs(rolls[-1]) < 5.0, f"Roll should settle, got {rolls[-1]:.1f}°"

    def test_6dof_produces_motion(self):
        """Wind should accelerate the 6-DOF boat."""
        yacht = Yacht6DOF()
        wind = WindState(speed=5.0, direction=1.047)
        control = ControlCommand()

        for _ in range(400):
            yacht.step(wind, control, 0.05)

        assert yacht.state.speed > 0.3, f"Expected forward motion, got {yacht.state.speed:.3f} m/s"

    def test_compute_forces_returns_6dof(self):
        """compute_forces should return force arrays."""
        yacht = Yacht6DOF()
        yacht.reset(
            VesselState(
                eta=np.zeros(6),
                nu=np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            )
        )
        wind = WindState(speed=5.0, direction=1.0)
        control = ControlCommand(rudder_angle=0.1)

        forces = yacht.compute_forces(wind, control)
        assert forces.sail.shape == (6,)
        assert forces.rudder.shape == (6,)
        assert forces.keel.shape == (6,)


class TestYacht6DOFvs3DOF:
    def test_surge_sway_yaw_similar(self):
        """In light conditions, 6-DOF surge/sway/yaw should be similar to 3-DOF."""
        from sailsim.vessel.yacht_3dof import Yacht3DOF

        yacht3 = Yacht3DOF()
        yacht6 = Yacht6DOF()

        wind = WindState(speed=3.0, direction=1.047)
        control = ControlCommand()

        for _ in range(200):
            yacht3.step(wind, control, 0.05)
            yacht6.step(wind, control, 0.05)

        # Positions and velocities should be in the same ballpark
        # (not identical due to different force functions)
        assert abs(yacht6.state.x - yacht3.state.x) < 10.0
        assert abs(yacht6.state.y - yacht3.state.y) < 10.0
