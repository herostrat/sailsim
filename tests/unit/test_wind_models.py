"""Tests for wind environment models."""

from __future__ import annotations

import numpy as np
import pytest

from sailsim.environment.wind import ConstantWind, GustWind, ShiftingWind


class TestConstantWind:
    """ConstantWind should return the same state regardless of time."""

    def test_returns_configured_values(self):
        wind = ConstantWind(speed=6.0, direction=1.0)
        state = wind.get(0.0)
        assert state.speed == 6.0
        assert state.direction == 1.0

    def test_constant_over_time(self):
        wind = ConstantWind(speed=5.0, direction=0.5)
        for t in [0.0, 10.0, 100.0, 1000.0]:
            state = wind.get(t)
            assert state.speed == 5.0
            assert state.direction == 0.5


class TestGustWind:
    """GustWind should vary speed over time with mean-reverting noise."""

    def test_varies_over_time(self):
        """Speed should not be constant."""
        wind = GustWind(base_speed=8.0, direction=0.0, gust_intensity=3.0, seed=42)
        speeds = [wind.get(t).speed for t in np.arange(0, 60, 0.5)]
        assert max(speeds) > min(speeds), "Gust wind should vary"

    def test_reproducible_with_seed(self):
        """Same seed should produce identical results."""
        kwargs = dict(base_speed=8.0, direction=0.0, gust_intensity=2.0, gust_tau=10.0, seed=123)
        wind1 = GustWind(**kwargs)
        wind2 = GustWind(**kwargs)

        times = np.arange(0, 30, 1.0)
        for t in times:
            s1 = wind1.get(t).speed
            s2 = wind2.get(t).speed
            assert s1 == pytest.approx(s2), f"Mismatch at t={t}"

    def test_mean_near_base_speed(self):
        """Mean speed over long period should be close to base_speed."""
        wind = GustWind(base_speed=8.0, direction=0.0, gust_intensity=2.0, gust_tau=10.0, seed=99)
        speeds = [wind.get(t).speed for t in np.arange(0, 300, 0.1)]
        mean_speed = np.mean(speeds)
        assert abs(mean_speed - 8.0) < 1.5, f"Mean {mean_speed:.1f} too far from base 8.0"

    def test_speed_never_negative(self):
        """Wind speed must never go below zero."""
        wind = GustWind(base_speed=3.0, direction=0.0, gust_intensity=5.0, gust_tau=5.0, seed=7)
        for t in np.arange(0, 120, 0.1):
            assert wind.get(t).speed >= 0.0

    def test_direction_constant(self):
        """Direction should remain constant for gust model."""
        wind = GustWind(base_speed=8.0, direction=1.5, gust_intensity=3.0, seed=42)
        for t in [0, 10, 50]:
            assert wind.get(t).direction == 1.5


class TestShiftingWind:
    """ShiftingWind should change direction while keeping speed constant."""

    def test_speed_constant(self):
        """Speed should remain constant."""
        wind = ShiftingWind(speed=7.0, base_direction=0.0, amplitude=0.5, period=60.0)
        for t in np.arange(0, 120, 1.0):
            assert wind.get(t).speed == 7.0

    def test_linear_grows(self):
        """Linear mode: direction should increase over time."""
        rate = 0.01  # rad/s
        wind = ShiftingWind(speed=5.0, base_direction=0.0, mode="linear", rate=rate)
        d0 = wind.get(0.0).direction
        d10 = wind.get(10.0).direction
        # After 10s at 0.01 rad/s → direction should have increased by ~0.1 rad
        assert d10 > d0

    def test_linear_rate(self):
        """Linear mode: direction change matches rate."""
        rate = 0.05
        wind = ShiftingWind(speed=5.0, base_direction=0.0, mode="linear", rate=rate)
        d = wind.get(10.0).direction
        assert d == pytest.approx(rate * 10.0, abs=0.001)

    def test_sinusoidal_oscillates(self):
        """Sinusoidal mode: direction should oscillate around base."""
        period = 60.0
        amplitude = 0.3
        wind = ShiftingWind(
            speed=5.0,
            base_direction=1.0,
            mode="sinusoidal",
            amplitude=amplitude,
            period=period,
        )
        directions = [wind.get(t).direction for t in np.arange(0, 120, 0.5)]
        assert max(directions) > 1.0  # goes above base
        assert min(directions) < 1.0  # goes below base
        assert max(directions) <= 1.0 + amplitude + 0.01
        assert min(directions) >= 1.0 - amplitude - 0.01

    def test_sinusoidal_returns_to_base(self):
        """After one full period, direction should return near base."""
        period = 60.0
        wind = ShiftingWind(
            speed=5.0,
            base_direction=0.5,
            mode="sinusoidal",
            amplitude=0.2,
            period=period,
        )
        d_start = wind.get(0.0).direction
        d_full = wind.get(period).direction
        assert d_start == pytest.approx(d_full, abs=0.01)

    def test_direction_normalized(self):
        """Direction should always be in [-pi, pi]."""
        wind = ShiftingWind(speed=5.0, base_direction=3.0, mode="linear", rate=0.1)
        for t in [0, 10, 50, 100]:
            d = wind.get(t).direction
            assert -np.pi <= d <= np.pi


class TestBuildWindModel:
    """Test the factory function."""

    def test_constant_default(self):
        from sailsim.core.config import WindConfig
        from sailsim.environment import build_wind_model

        cfg = WindConfig()
        model = build_wind_model(cfg)
        assert isinstance(model, ConstantWind)

    def test_gust_model(self):
        from sailsim.core.config import WindConfig
        from sailsim.environment import build_wind_model

        cfg = WindConfig(model="gust", speed=8.0, gust_intensity=3.0, gust_seed=42)
        model = build_wind_model(cfg)
        assert isinstance(model, GustWind)
        state = model.get(0.0)
        assert state.speed > 0

    def test_shifting_model(self):
        from sailsim.core.config import WindConfig
        from sailsim.environment import build_wind_model

        cfg = WindConfig(
            model="shifting", shift_mode="sinusoidal", shift_amplitude=0.2, shift_period=60.0
        )
        model = build_wind_model(cfg)
        assert isinstance(model, ShiftingWind)
