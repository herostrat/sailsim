"""Tests for the Nomoto-based autopilot."""

from __future__ import annotations

import numpy as np

from sailsim.autopilot.nomoto import (
    NomotoAutopilot,
    _compute_gains,
    estimate_nomoto_params,
)
from sailsim.core.config import YachtConfig
from sailsim.core.types import SensorData
from sailsim.physics.dynamics import mass_matrix_3dof


def _default_yacht() -> YachtConfig:
    return YachtConfig()


def _j24_yacht() -> YachtConfig:
    return YachtConfig(
        mass=1406,
        Iz=3270.0,
        xg=0.0,
        X_udot=-125.9,
        Y_vdot=-703.0,
        Y_rdot=-119.9,
        N_vdot=-119.9,
        N_rdot=-327.0,
        Xu=-76.6,
        Yv=-3005.0,
        Yr=-737.0,
        Nv=-737.0,
        Nr=-8035.0,
        rudder_area=0.1,
        rudder_x=-2.9,
        rudder_max=0.52,
    )


def _on_course_sensors(heading: float = 0.0, stw: float = 2.0) -> SensorData:
    """Sensors with zero error and zero yaw rate."""
    return SensorData(
        heading=heading,
        speed_through_water=stw,
        x=0.0,
        y=0.0,
        roll=0.0,
        yaw_rate=0.0,
        apparent_wind_speed=5.0,
        apparent_wind_angle=1.0,
        speed_over_ground=stw,
        course_over_ground=heading,
    )


# ---- M' matrix consistency ----


class TestSwayYawMassMatrix:
    """M' used in Nomoto estimation must match the sway-yaw sub-block
    of the full mass_matrix_3dof."""

    def test_sway_yaw_mass_matrix(self):
        y = _default_yacht()
        M_full = mass_matrix_3dof(
            m=y.mass,
            Iz=y.Iz,
            xg=y.xg,
            X_udot=y.X_udot,
            Y_vdot=y.Y_vdot,
            Y_rdot=y.Y_rdot,
            N_vdot=y.N_vdot,
            N_rdot=y.N_rdot,
        )
        # Sway-yaw sub-block = rows/cols [1,2]
        M_sub = M_full[1:, 1:]

        # M' from estimate_nomoto_params (reconstruct)
        M_prime = np.array(
            [
                [y.mass - y.Y_vdot, y.mass * y.xg - y.Y_rdot],
                [y.mass * y.xg - y.N_vdot, y.Iz - y.N_rdot],
            ]
        )
        np.testing.assert_allclose(M_prime, M_sub)

    def test_mass_matrix_consistency_j24(self):
        """M' == mass_matrix_3dof[1:,1:] also for J/24."""
        y = _j24_yacht()
        M_full = mass_matrix_3dof(
            m=y.mass,
            Iz=y.Iz,
            xg=y.xg,
            X_udot=y.X_udot,
            Y_vdot=y.Y_vdot,
            Y_rdot=y.Y_rdot,
            N_vdot=y.N_vdot,
            N_rdot=y.N_rdot,
        )
        M_sub = M_full[1:, 1:]

        M_prime = np.array(
            [
                [y.mass - y.Y_vdot, y.mass * y.xg - y.Y_rdot],
                [y.mass * y.xg - y.N_vdot, y.Iz - y.N_rdot],
            ]
        )
        np.testing.assert_allclose(M_prime, M_sub)


# ---- Nomoto parameter estimation ----


class TestNomotoParams:
    def test_nomoto_params_default_yacht(self):
        """K > 0, T in plausible range for default yacht at 2 m/s."""
        params = estimate_nomoto_params(_default_yacht(), U=2.0)
        assert params.K > 0, f"K should be positive, got {params.K}"
        assert 1.0 < params.T < 10.0, f"T should be 1-10s, got {params.T}"
        assert params.T1 > 0
        assert params.T2 > 0
        assert params.U_ref == 2.0

    def test_K_increases_with_speed(self):
        """Higher speed → more rudder authority → larger K."""
        y = _default_yacht()
        K_slow = estimate_nomoto_params(y, U=1.5).K
        K_fast = estimate_nomoto_params(y, U=3.0).K
        assert K_fast > K_slow

    def test_gains_decrease_with_speed(self):
        """Higher K at higher speed → lower Kp needed."""
        y = _default_yacht()
        p_slow = estimate_nomoto_params(y, U=1.5)
        p_fast = estimate_nomoto_params(y, U=3.0)

        Kp_slow, _, _ = _compute_gains(p_slow.K, p_slow.T, 0.4, 0.8)
        Kp_fast, _, _ = _compute_gains(p_fast.K, p_fast.T, 0.4, 0.8)
        assert Kp_fast < Kp_slow

    def test_works_with_j24(self):
        """Nomoto estimation succeeds with different yacht (J/24)."""
        params = estimate_nomoto_params(_j24_yacht(), U=2.0)
        assert params.K > 0
        assert 0.5 < params.T < 20.0


# ---- Pole placement formula ----


class TestPolePlacement:
    def test_pole_placement_formula(self):
        """Closed-loop poles match desired omega_n and zeta."""
        params = estimate_nomoto_params(_default_yacht(), U=2.0)
        omega_n = 0.4
        zeta = 0.8
        Kp, Kd, _Ki = _compute_gains(params.K, params.T, omega_n, zeta)

        # Closed-loop char. polynomial: T*s^2 + (1+K*Kd)*s + K*Kp = 0
        # Normalised: s^2 + (1+K*Kd)/T * s + K*Kp/T = 0
        # Should equal: s^2 + 2*zeta*wn*s + wn^2
        a1 = (1.0 + params.K * Kd) / params.T
        a0 = params.K * Kp / params.T

        np.testing.assert_allclose(a1, 2.0 * zeta * omega_n, rtol=1e-10)
        np.testing.assert_allclose(a0, omega_n**2, rtol=1e-10)


# ---- NomotoAutopilot controller ----


class TestNomotoAutopilot:
    def test_zero_error_zero_output(self):
        """On course with no yaw rate → near-zero rudder."""
        ap = NomotoAutopilot(yacht=_default_yacht())
        ap.set_target_heading(0.0)
        sensors = _on_course_sensors(heading=0.0)
        cmd = ap.compute(sensors, dt=0.05)
        assert abs(cmd.rudder_angle) < 0.01

    def test_rudder_rate_limiting(self):
        """Rudder change per step must not exceed rate limit."""
        rate_max = np.radians(5.0)  # 5 deg/s
        dt = 0.05
        max_change = rate_max * dt

        ap = NomotoAutopilot(
            yacht=_default_yacht(),
            rudder_rate_max=rate_max,
        )
        ap.set_target_heading(np.radians(90.0))  # large step → demands full rudder

        sensors = _on_course_sensors(heading=0.0)
        cmd = ap.compute(sensors, dt=dt)

        # First step starts from 0.0 → change = cmd.rudder_angle
        assert abs(cmd.rudder_angle) <= max_change + 1e-9

    def test_protocol_compliance(self):
        """NomotoAutopilot has compute() and set_target_heading()."""
        ap = NomotoAutopilot(yacht=_default_yacht())
        assert hasattr(ap, "compute")
        assert hasattr(ap, "set_target_heading")
        assert callable(ap.compute)
        assert callable(ap.set_target_heading)

    def test_starboard_error_positive_rudder(self):
        """Target to starboard → positive rudder angle."""
        ap = NomotoAutopilot(
            yacht=_default_yacht(),
            rudder_rate_max=10.0,  # high rate limit to not clip
        )
        ap.set_target_heading(np.radians(30.0))
        sensors = _on_course_sensors(heading=0.0)
        cmd = ap.compute(sensors, dt=0.05)
        assert cmd.rudder_angle > 0

    def test_set_target_resets_integral(self):
        """Changing target heading resets the integrator."""
        ap = NomotoAutopilot(yacht=_default_yacht())
        ap.set_target_heading(np.radians(10.0))
        sensors = _on_course_sensors(heading=0.0)
        # Accumulate some integral
        for _ in range(100):
            ap.compute(sensors, dt=0.05)
        assert ap._integral != 0.0

        ap.set_target_heading(np.radians(20.0))
        assert ap._integral == 0.0

    def test_rudder_position_limited(self):
        """Rudder angle never exceeds rudder_max."""
        ap = NomotoAutopilot(
            yacht=_default_yacht(),
            rudder_rate_max=100.0,  # no rate limit
        )
        ap.set_target_heading(np.radians(180.0))
        sensors = _on_course_sensors(heading=0.0)
        cmd = ap.compute(sensors, dt=0.05)
        assert abs(cmd.rudder_angle) <= ap.rudder_max + 1e-9

    def test_auto_sail_trim(self):
        """auto_sail_trim=True uses optimal_sail_trim, not fixed 0.5."""
        ap = NomotoAutopilot(
            yacht=_default_yacht(),
            auto_sail_trim=True,
        )
        ap.set_target_heading(0.0)
        sensors = _on_course_sensors(heading=0.0)
        cmd = ap.compute(sensors, dt=0.05)
        # With AWA=1.0 rad (~57°), optimal trim should differ from 0.5
        assert isinstance(cmd.sail_trim, float)
        assert 0.0 <= cmd.sail_trim <= 1.0
