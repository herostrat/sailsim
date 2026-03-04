"""Tests for model-based linear stability analysis."""

from __future__ import annotations

import numpy as np
import pytest

from sailsim.analysis.linear import (
    ClosedLoopPoles,
    LinearAnalysisResult,
    StabilityMargins,
    analyze_at_speed,
    build_controller_tf,
    build_plant_tf,
    build_plant_tf_2nd,
    describing_function_rate_limiter,
    sweep_speed,
)
from sailsim.autopilot.nomoto import _compute_gains, estimate_nomoto_params
from sailsim.core.config import YachtConfig


@pytest.fixture()
def yacht() -> YachtConfig:
    return YachtConfig()


# -------------------------------------------------------------------
# Transfer function construction
# -------------------------------------------------------------------

class TestBuildPlantTF:
    def test_coefficients_1st_order(self) -> None:
        """G(s) = K / (T*s^2 + s) — check num and den ratios."""
        K, T = 0.5, 3.0
        G = build_plant_tf(K, T)
        # scipy normalizes so den[0]=1; check ratios instead
        np.testing.assert_array_almost_equal(G.num / G.num[0], [1.0])
        np.testing.assert_array_almost_equal(
            G.den / G.den[0], [1.0, 1.0 / T, 0.0],
        )
        # Verify gain ratio: num[0] / den[0] should give K / T
        assert abs(G.num[0] / G.den[0] - K / T) < 1e-10

    def test_dc_gain_is_infinite(self) -> None:
        """Plant has integrator → DC gain = inf."""
        G = build_plant_tf(0.5, 3.0)
        # Evaluate at s = 0: G(0) = K/0 = inf
        assert G.den[-1] == 0.0

    def test_2nd_order_coefficients(self) -> None:
        """G(s) = K*(T3*s+1) / (s*(T1*s+1)*(T2*s+1))."""
        K, T1, T2, T3 = 0.5, 2.0, 1.0, 0.5
        G = build_plant_tf_2nd(K, T1, T2, T3)
        # scipy normalizes so den[0]=1; check ratios
        expected_num = np.array([K * T3, K])
        expected_den = np.array([T1 * T2, T1 + T2, 1.0, 0.0])
        np.testing.assert_array_almost_equal(
            G.num / G.den[0], expected_num / expected_den[0],
        )
        np.testing.assert_array_almost_equal(
            G.den / G.den[0], expected_den / expected_den[0],
        )


class TestBuildControllerTF:
    def test_coefficients(self) -> None:
        """C(s) = (Kd*s^2 + Kp*s + Ki) / s."""
        Kp, Kd, Ki = 3.0, 10.0, 0.1
        C = build_controller_tf(Kp, Kd, Ki)
        np.testing.assert_array_almost_equal(C.num, [Kd, Kp, Ki])
        np.testing.assert_array_almost_equal(C.den, [1.0, 0.0])


# -------------------------------------------------------------------
# Stability margins
# -------------------------------------------------------------------

class TestComputeMargins:
    def test_stable_system_positive_margins(self, yacht: YachtConfig) -> None:
        """A well-tuned system should have positive GM and PM."""
        result = analyze_at_speed(yacht, U=3.0, omega_n=0.5, zeta=0.8)
        assert result.margins.gain_margin_db > 0
        assert result.margins.phase_margin_deg > 0

    def test_bandwidth_positive(self, yacht: YachtConfig) -> None:
        result = analyze_at_speed(yacht, U=3.0, omega_n=0.5, zeta=0.8)
        assert result.margins.bandwidth_rad_s > 0

    def test_delay_margin_positive(self, yacht: YachtConfig) -> None:
        result = analyze_at_speed(yacht, U=3.0, omega_n=0.5, zeta=0.8)
        assert result.margins.delay_margin_s > 0


# -------------------------------------------------------------------
# Pole analysis
# -------------------------------------------------------------------

class TestPoleAnalysis:
    def test_stable_poles(self, yacht: YachtConfig) -> None:
        """Default yacht at U=2.0 with omega_n=0.5, zeta=0.8 should be stable."""
        result = analyze_at_speed(yacht, U=2.0, omega_n=0.5, zeta=0.8)
        assert result.poles.is_stable
        assert all(np.real(p) < 0 for p in result.poles.poles)

    def test_damping_near_design(self, yacht: YachtConfig) -> None:
        """Dominant poles should have damping near the design zeta."""
        result = analyze_at_speed(yacht, U=2.0, omega_n=0.5, zeta=0.8)
        # Find dominant poles (largest natural frequency matching omega_n)
        dominant_mask = result.poles.natural_frequencies > 0.1
        if np.any(dominant_mask):
            dominant_zeta = result.poles.damping_ratios[dominant_mask]
            # At least one pole pair near design zeta (within tolerance)
            assert np.any(np.abs(dominant_zeta - 0.8) < 0.5)

    def test_gains_match_pole_placement(self, yacht: YachtConfig) -> None:
        """Verify gains from analyze_at_speed match _compute_gains."""
        U = 3.0
        nomoto = estimate_nomoto_params(yacht, U)
        Kp, Kd, Ki = _compute_gains(nomoto.K, nomoto.T, 0.5, 0.8)
        result = analyze_at_speed(yacht, U, omega_n=0.5, zeta=0.8)
        assert abs(result.Kp - Kp) < 1e-10
        assert abs(result.Kd - Kd) < 1e-10
        assert abs(result.Ki - Ki) < 1e-10


# -------------------------------------------------------------------
# Speed sweep
# -------------------------------------------------------------------

class TestSpeedSweep:
    def test_sweep_length(self, yacht: YachtConfig) -> None:
        U_range = np.linspace(1.0, 6.0, 6)
        result = sweep_speed(yacht, U_range, omega_n=0.5, zeta=0.8)
        assert len(result.speeds) == 6
        assert len(result.K_values) == 6
        assert len(result.poles_list) == 6

    def test_all_stable_in_range(self, yacht: YachtConfig) -> None:
        """Default yacht should be stable from 1 to 6 m/s."""
        U_range = np.linspace(1.0, 6.0, 6)
        result = sweep_speed(yacht, U_range, omega_n=0.5, zeta=0.8)
        assert np.all(result.is_stable)

    def test_margins_positive_in_range(self, yacht: YachtConfig) -> None:
        U_range = np.linspace(1.0, 6.0, 6)
        result = sweep_speed(yacht, U_range, omega_n=0.5, zeta=0.8)
        assert np.all(result.gain_margins_db > 0)
        assert np.all(result.phase_margins_deg > 0)

    def test_K_varies_with_speed(self, yacht: YachtConfig) -> None:
        """Nomoto K depends on speed — values should differ."""
        U_range = np.array([1.0, 3.0, 6.0])
        result = sweep_speed(yacht, U_range, omega_n=0.5, zeta=0.8)
        assert result.K_values[0] != result.K_values[1]
        assert result.K_values[1] != result.K_values[2]


# -------------------------------------------------------------------
# Describing function
# -------------------------------------------------------------------

class TestDescribingFunction:
    def test_no_clipping(self) -> None:
        """If A*omega < rate_limit, N = 1."""
        N = describing_function_rate_limiter(
            rate_limit=5.0, amplitude=1.0, omega=2.0,
        )
        assert complex(1.0, 0.0) == N

    def test_at_boundary(self) -> None:
        """If A*omega == rate_limit, N = 1."""
        N = describing_function_rate_limiter(
            rate_limit=5.0, amplitude=1.0, omega=5.0,
        )
        assert complex(1.0, 0.0) == N

    def test_clipping_reduces_gain(self) -> None:
        """If A*omega > rate_limit, |N| < 1."""
        N = describing_function_rate_limiter(
            rate_limit=1.0, amplitude=1.0, omega=10.0,
        )
        assert abs(N) < 1.0

    def test_zero_amplitude(self) -> None:
        N = describing_function_rate_limiter(
            rate_limit=5.0, amplitude=0.0, omega=1.0,
        )
        assert complex(1.0, 0.0) == N

    def test_zero_omega(self) -> None:
        N = describing_function_rate_limiter(
            rate_limit=5.0, amplitude=1.0, omega=0.0,
        )
        assert complex(1.0, 0.0) == N


# -------------------------------------------------------------------
# Full analysis integration
# -------------------------------------------------------------------

class TestAnalyzeAtSpeed:
    def test_returns_complete_result(self, yacht: YachtConfig) -> None:
        result = analyze_at_speed(yacht, U=3.0)
        assert isinstance(result, LinearAnalysisResult)
        assert isinstance(result.margins, StabilityMargins)
        assert isinstance(result.poles, ClosedLoopPoles)
        assert result.U == 3.0
        assert result.plant_tf is not None
        assert result.controller_tf is not None
        assert result.open_loop_tf is not None
        assert result.closed_loop_tf is not None

    def test_different_speeds_different_results(self, yacht: YachtConfig) -> None:
        r1 = analyze_at_speed(yacht, U=2.0)
        r2 = analyze_at_speed(yacht, U=5.0)
        assert r1.nomoto.K != r2.nomoto.K
        assert r1.Kp != r2.Kp
