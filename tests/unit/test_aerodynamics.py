"""Test sail aerodynamics model."""

import numpy as np
import pytest

from sailsim.physics.aerodynamics import apparent_wind, sail_coefficients, sail_forces_3dof


class TestApparentWind:
    def test_head_to_wind_stationary(self):
        """Boat stationary, wind from north: AWA = 0 (head to wind)."""
        speed, awa = apparent_wind(5.0, 0.0, 0.0, 0.0, 0.0)
        assert speed == pytest.approx(5.0, abs=0.01)
        assert awa == pytest.approx(0.0, abs=0.01)

    def test_beam_wind_starboard(self):
        """Wind from 90° (east), boat heading north, stationary."""
        speed, awa = apparent_wind(5.0, np.pi / 2, 0.0, 0.0, 0.0)
        assert speed == pytest.approx(5.0, abs=0.01)
        assert awa == pytest.approx(np.pi / 2, abs=0.01)  # starboard

    def test_running_downwind(self):
        """Wind from 180° (south), boat heading north, stationary."""
        speed, awa = apparent_wind(5.0, np.pi, 0.0, 0.0, 0.0)
        assert speed == pytest.approx(5.0, abs=0.01)
        assert abs(awa) == pytest.approx(np.pi, abs=0.01)

    def test_boat_speed_reduces_awa(self):
        """Moving boat should shift apparent wind forward."""
        # Wind from beam (90°), boat moving forward
        _, awa_stationary = apparent_wind(5.0, np.pi / 2, 0.0, 0.0, 0.0)
        _, awa_moving = apparent_wind(5.0, np.pi / 2, 3.0, 0.0, 0.0)
        # AWA should be smaller (more forward) when boat moves
        assert abs(awa_moving) < abs(awa_stationary)

    def test_apparent_wind_speed_increases_upwind(self):
        """Moving into the wind increases apparent wind speed."""
        speed_stat, _ = apparent_wind(5.0, 0.0, 0.0, 0.0, 0.0)
        speed_moving, _ = apparent_wind(5.0, 0.0, 3.0, 0.0, 0.0)
        assert speed_moving > speed_stat


class TestSailCoefficients:
    def test_in_irons(self):
        """No useful CL when head-to-wind."""
        cl, cd = sail_coefficients(np.radians(10))
        assert cl == 0.0
        assert cd == pytest.approx(0.05)

    def test_close_hauled_peak(self):
        """Peak CL around 40-50° AWA."""
        cl_30, _ = sail_coefficients(np.radians(30))
        cl_45, _ = sail_coefficients(np.radians(45))
        assert cl_45 > cl_30

    def test_running_no_lift(self):
        """At 170° (running), CL should be ~0, CD high."""
        cl, cd = sail_coefficients(np.radians(170))
        assert cl == pytest.approx(0.0, abs=0.01)
        assert cd > 0.7

    def test_symmetric(self):
        """Port and starboard wind should give same coefficients."""
        cl_pos, cd_pos = sail_coefficients(np.radians(45))
        cl_neg, cd_neg = sail_coefficients(np.radians(-45))
        assert cl_pos == cl_neg
        assert cd_pos == cd_neg


class TestSailForces:
    def test_no_force_in_irons(self):
        """Head-to-wind: minimal forward drive."""
        forces = sail_forces_3dof(5.0, np.radians(5), 50.0, 0.5)
        # X should be small/negative (drag only)
        assert forces[0] < 0.1

    def test_driving_force_on_reach(self):
        """Beam reach should produce positive driving force."""
        forces = sail_forces_3dof(5.0, np.radians(45), 50.0, 0.5)
        assert forces[0] > 0  # forward drive

    def test_heeling_to_leeward(self):
        """Wind from starboard (positive AWA) -> heeling to port (negative Y)."""
        forces = sail_forces_3dof(5.0, np.radians(45), 50.0, 0.5)
        assert forces[1] < 0  # leeward = port = negative Y

    def test_heeling_port_wind(self):
        """Wind from port (negative AWA) -> heeling to starboard (positive Y)."""
        forces = sail_forces_3dof(5.0, np.radians(-45), 50.0, 0.5)
        assert forces[1] > 0  # leeward = starboard = positive Y

    def test_zero_wind_zero_force(self):
        """No wind = no force."""
        forces = sail_forces_3dof(0.0, np.radians(45), 50.0, 0.5)
        np.testing.assert_array_almost_equal(forces, [0, 0, 0])

    def test_yaw_moment_sign(self):
        """CE forward + wind from starboard -> negative Y -> negative N."""
        forces = sail_forces_3dof(5.0, np.radians(45), 50.0, 0.5)
        # N = Y * ce_x, Y<0, ce_x=0.5 => N<0
        assert forces[2] < 0
