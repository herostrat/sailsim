"""Tests for ocean current environment models."""

from __future__ import annotations

import numpy as np
import pytest

from sailsim.core.types import CurrentState
from sailsim.environment.current import ConstantCurrent, NoCurrent, TidalCurrent


class TestCurrentState:
    """Tests for CurrentState dataclass."""

    def test_velocity_ned_north(self):
        """Current flowing north: vn > 0, ve ≈ 0."""
        c = CurrentState(speed=1.0, direction=0.0)
        vn, ve = c.velocity_ned
        assert vn == pytest.approx(1.0)
        assert ve == pytest.approx(0.0, abs=1e-10)

    def test_velocity_ned_east(self):
        """Current flowing east: vn ≈ 0, ve > 0."""
        c = CurrentState(speed=2.0, direction=np.pi / 2)
        vn, ve = c.velocity_ned
        assert vn == pytest.approx(0.0, abs=1e-10)
        assert ve == pytest.approx(2.0)

    def test_zero_speed(self):
        c = CurrentState(speed=0.0, direction=1.0)
        vn, ve = c.velocity_ned
        assert vn == pytest.approx(0.0)
        assert ve == pytest.approx(0.0)


class TestNoCurrent:
    def test_always_zero(self):
        model = NoCurrent()
        for t in [0, 10, 100]:
            state = model.get(t)
            assert state.speed == 0.0
            assert state.direction == 0.0


class TestConstantCurrent:
    def test_constant_values(self):
        model = ConstantCurrent(speed=0.5, direction=1.2)
        for t in [0, 50, 1000]:
            state = model.get(t)
            assert state.speed == 0.5
            assert state.direction == 1.2


class TestTidalCurrent:
    def test_varies_over_time(self):
        model = TidalCurrent(base_speed=0.5, amplitude=0.3, period=100.0, direction=0.0)
        speeds = [model.get(t).speed for t in np.arange(0, 200, 1.0)]
        assert max(speeds) > min(speeds)

    def test_direction_constant(self):
        model = TidalCurrent(base_speed=0.5, amplitude=0.3, period=100.0, direction=1.5)
        for t in [0, 25, 50, 75]:
            assert model.get(t).direction == 1.5

    def test_sinusoidal_pattern(self):
        period = 100.0
        model = TidalCurrent(base_speed=1.0, amplitude=0.5, period=period, direction=0.0)
        # At t=0, phase=0: sin(0) = 0 → speed = base
        assert model.get(0.0).speed == pytest.approx(1.0)
        # At t=period/4: sin(π/2) = 1 → speed = base + amplitude
        assert model.get(period / 4).speed == pytest.approx(1.5)

    def test_speed_never_negative(self):
        model = TidalCurrent(base_speed=0.3, amplitude=0.5, period=60.0, direction=0.0)
        for t in np.arange(0, 120, 0.5):
            assert model.get(t).speed >= 0.0


class TestBuildCurrentModel:
    def test_none_default(self):
        from sailsim.core.config import CurrentConfig
        from sailsim.environment import build_current_model

        cfg = CurrentConfig()
        model = build_current_model(cfg)
        assert isinstance(model, NoCurrent)

    def test_constant(self):
        from sailsim.core.config import CurrentConfig
        from sailsim.environment import build_current_model

        cfg = CurrentConfig(model="constant", speed=0.5, direction=1.0)
        model = build_current_model(cfg)
        assert isinstance(model, ConstantCurrent)
        assert model.get(0).speed == 0.5

    def test_tidal(self):
        from sailsim.core.config import CurrentConfig
        from sailsim.environment import build_current_model

        cfg = CurrentConfig(model="tidal", speed=0.5, tidal_amplitude=0.3, tidal_period=100.0)
        model = build_current_model(cfg)
        assert isinstance(model, TidalCurrent)
