"""Tests for sail trim functionality.

Covers optimal_sail_trim(), sail_coefficients with trim, backward compatibility,
and PID auto_sail_trim option.
"""

from __future__ import annotations

import numpy as np
import pytest

from sailsim.physics.aerodynamics import optimal_sail_trim, sail_coefficients, sail_forces_3dof


class TestOptimalSailTrim:
    """Tests for optimal_sail_trim()."""

    def test_close_hauled_gives_high_trim(self):
        """Close-hauled (AWA ~30°) should give trim near 1.0."""
        trim = optimal_sail_trim(np.radians(30))
        assert trim == pytest.approx(1.0, abs=0.05)

    def test_beam_reach_gives_mid_trim(self):
        """Beam reach (AWA ~90°) should give trim around 0.6."""
        trim = optimal_sail_trim(np.radians(90))
        assert 0.5 < trim < 0.7

    def test_broad_reach_gives_low_trim(self):
        """Broad reach (AWA ~140°) should give low trim."""
        trim = optimal_sail_trim(np.radians(140))
        assert trim < 0.35

    def test_running_gives_zero_trim(self):
        """Running (AWA ~180°) should give trim 0.0."""
        trim = optimal_sail_trim(np.radians(180))
        assert trim == pytest.approx(0.0, abs=0.05)

    def test_symmetric_port_starboard(self):
        """Port and starboard AWA should give same trim."""
        for awa_deg in [30, 60, 90, 120, 150]:
            trim_stbd = optimal_sail_trim(np.radians(awa_deg))
            trim_port = optimal_sail_trim(np.radians(-awa_deg))
            assert trim_stbd == pytest.approx(trim_port)

    def test_monotonically_decreasing(self):
        """Trim should decrease as AWA increases from close-hauled to running."""
        awas = [30, 60, 90, 120, 150, 180]
        trims = [optimal_sail_trim(np.radians(a)) for a in awas]
        for i in range(len(trims) - 1):
            assert trims[i] >= trims[i + 1]

    def test_output_clamped_01(self):
        """Output should always be in [0, 1]."""
        for awa_deg in range(0, 181, 5):
            trim = optimal_sail_trim(np.radians(awa_deg))
            assert 0.0 <= trim <= 1.0


class TestSailCoefficientsWithTrim:
    """Tests for sail_coefficients with sail_trim parameter."""

    def test_optimal_trim_beats_wrong_trim(self):
        """At any AWA, optimal trim should give >= CL than a deliberately wrong trim."""
        for awa_deg in [40, 60, 90, 120]:
            awa = np.radians(awa_deg)
            trim_opt = optimal_sail_trim(awa)
            cl_opt, _ = sail_coefficients(awa, trim_opt)

            # Deliberately wrong: if optimal is high, use 0; if low, use 1
            wrong_trim = 0.0 if trim_opt > 0.5 else 1.0
            cl_wrong, _ = sail_coefficients(awa, wrong_trim)

            assert cl_opt >= cl_wrong, (
                f"AWA={awa_deg}°: CL(trim_opt={trim_opt:.2f})={cl_opt:.3f} "
                f"< CL(wrong={wrong_trim})={cl_wrong:.3f}"
            )

    def test_default_trim_no_penalty(self):
        """Default sail_trim=0.5 should never degrade CL (deadzone covers it)."""
        for awa_deg in [30, 60, 90, 120, 150]:
            awa = np.radians(awa_deg)
            cl_default, _ = sail_coefficients(awa, 0.5)
            cl_no_trim, _ = sail_coefficients(awa)  # also 0.5
            assert cl_default == cl_no_trim

    def test_overtrim_increases_drag(self):
        """Sheeting too tight for the AWA should increase CD."""
        # Running: AWA=150°, optimal trim near 0, sheeted tight = overtrimmed
        awa = np.radians(150)
        _, cd_eased = sail_coefficients(awa, 0.0)
        _, cd_sheeted = sail_coefficients(awa, 1.0)
        assert cd_sheeted > cd_eased


class TestSailForces3DOFWithTrim:
    """Tests for sail_forces_3dof with sail_trim parameter."""

    def test_default_trim_backward_compatible(self):
        """Calling without sail_trim should give same result as sail_trim=0.5."""
        aw_speed = 8.0
        awa = np.radians(60)
        sail_area = 50.0
        sail_ce_x = 0.3

        forces_default = sail_forces_3dof(aw_speed, awa, sail_area, sail_ce_x)
        forces_explicit = sail_forces_3dof(aw_speed, awa, sail_area, sail_ce_x, 0.5)
        np.testing.assert_array_equal(forces_default, forces_explicit)

    def test_optimal_trim_more_drive(self):
        """Optimal trim should produce at least as much driving force."""
        aw_speed = 8.0
        sail_area = 50.0
        sail_ce_x = 0.3

        for awa_deg in [45, 90]:
            awa = np.radians(awa_deg)
            trim_opt = optimal_sail_trim(awa)

            forces_opt = sail_forces_3dof(aw_speed, awa, sail_area, sail_ce_x, trim_opt)
            forces_default = sail_forces_3dof(aw_speed, awa, sail_area, sail_ce_x, 0.5)

            # X component is driving force (forward)
            assert forces_opt[0] >= forces_default[0] - 1.0  # allow small tolerance


class TestPIDAutoSailTrim:
    """Tests for PID autopilot auto_sail_trim option."""

    def test_auto_trim_disabled_gives_default(self):
        """With auto_sail_trim=False, sail_trim should be 0.5."""
        from sailsim.autopilot.pid import PIDAutopilot
        from sailsim.core.types import SensorData

        pilot = PIDAutopilot(auto_sail_trim=False)
        sensors = SensorData(heading=0.0, yaw_rate=0.0, apparent_wind_angle=np.radians(60))
        cmd = pilot.compute(sensors, dt=0.05)
        assert cmd.sail_trim == 0.5

    def test_auto_trim_enabled_adjusts_trim(self):
        """With auto_sail_trim=True, sail_trim should match optimal for AWA."""
        from sailsim.autopilot.pid import PIDAutopilot
        from sailsim.core.types import SensorData

        pilot = PIDAutopilot(auto_sail_trim=True)

        for awa_deg in [40, 90, 150]:
            awa = np.radians(awa_deg)
            sensors = SensorData(heading=0.0, yaw_rate=0.0, apparent_wind_angle=awa)
            cmd = pilot.compute(sensors, dt=0.05)
            expected_trim = optimal_sail_trim(awa)
            assert cmd.sail_trim == pytest.approx(expected_trim)
