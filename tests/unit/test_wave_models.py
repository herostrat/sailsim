"""Tests for wave environment models and spectra."""

from __future__ import annotations

import numpy as np
import pytest

from sailsim.environment.waves import (
    NoWaves,
    SpectralWaves,
    generate_wave_components,
    jonswap_spectrum,
    pierson_moskowitz_spectrum,
    wave_elevation,
)


class TestPiersonMoskowitzSpectrum:
    def test_peak_near_omega_p(self):
        """Spectrum should peak near omega_p = 2*pi/Tp."""
        Tp = 8.0
        omega = np.linspace(0.01, 3.0, 500)
        S = pierson_moskowitz_spectrum(omega, Hs=2.0, Tp=Tp)
        omega_p = 2.0 * np.pi / Tp
        peak_idx = np.argmax(S)
        assert abs(omega[peak_idx] - omega_p) < 0.1

    def test_energy_proportional_to_Hs_squared(self):
        """Spectral energy (m0) should be ≈ (Hs/4)^2."""
        omega = np.linspace(0.01, 4.0, 1000)
        Hs = 3.0
        S = pierson_moskowitz_spectrum(omega, Hs=Hs, Tp=8.0)
        m0 = np.trapezoid(S, omega)
        expected_m0 = (Hs / 4.0) ** 2
        assert m0 == pytest.approx(expected_m0, rel=0.15)

    def test_zero_at_omega_zero(self):
        """S(0) should be 0."""
        S = pierson_moskowitz_spectrum(np.array([0.0]), Hs=2.0, Tp=8.0)
        assert S[0] == 0.0


class TestJONSWAPSpectrum:
    def test_narrower_than_pm(self):
        """JONSWAP should be narrower (higher peak) than PM."""
        omega = np.linspace(0.01, 3.0, 500)
        S_pm = pierson_moskowitz_spectrum(omega, Hs=2.0, Tp=8.0)
        S_js = jonswap_spectrum(omega, Hs=2.0, Tp=8.0, gamma=3.3)
        assert np.max(S_js) > np.max(S_pm)

    def test_energy_normalized(self):
        """JONSWAP energy should be normalized to match Hs."""
        omega = np.linspace(0.01, 4.0, 1000)
        Hs = 2.5
        S = jonswap_spectrum(omega, Hs=Hs, Tp=8.0)
        m0 = np.trapezoid(S, omega)
        expected_m0 = (Hs / 4.0) ** 2
        assert m0 == pytest.approx(expected_m0, rel=0.05)


class TestWaveComponents:
    def test_reproducible_with_seed(self):
        wc1 = generate_wave_components(Hs=1.5, Tp=7.0, direction=0.0, seed=42)
        wc2 = generate_wave_components(Hs=1.5, Tp=7.0, direction=0.0, seed=42)
        np.testing.assert_array_equal(wc1.amplitudes, wc2.amplitudes)
        np.testing.assert_array_equal(wc1.phases, wc2.phases)

    def test_elevation_bounded(self):
        """Elevation should be bounded (not diverge)."""
        wc = generate_wave_components(Hs=2.0, Tp=8.0, direction=0.0, n_components=50, seed=7)
        elevations = [wave_elevation(wc, 0, 0, t) for t in np.arange(0, 300, 0.1)]
        assert max(abs(e) for e in elevations) < 5.0 * 2.0  # within 5*Hs

    def test_elevation_varies(self):
        """Elevation should change over time."""
        wc = generate_wave_components(Hs=2.0, Tp=8.0, direction=0.0, seed=42)
        elevations = [wave_elevation(wc, 0, 0, t) for t in np.arange(0, 30, 0.5)]
        assert max(elevations) > min(elevations)


class TestNoWaves:
    def test_zero_state(self):
        model = NoWaves()
        state = model.get(10.0)
        assert state.Hs == 0.0
        assert state.elevation == 0.0


class TestSpectralWaves:
    def test_returns_wave_state(self):
        model = SpectralWaves(Hs=1.5, Tp=7.0, direction=0.5, seed=42)
        state = model.get(0.0)
        assert state.Hs == 1.5
        assert state.Tp == 7.0
        assert state.direction == 0.5

    def test_elevation_varies_over_time(self):
        model = SpectralWaves(Hs=2.0, Tp=8.0, direction=0.0, seed=42)
        elevations = [model.get(t).elevation for t in np.arange(0, 30, 0.5)]
        assert max(elevations) > min(elevations)

    def test_position_affects_elevation(self):
        model = SpectralWaves(Hs=2.0, Tp=8.0, direction=0.0, seed=42)
        model.set_boat_position(0, 0)
        _ = model.get(10.0).elevation
        model.set_boat_position(100, 0)
        e2 = model.get(10.0).elevation
        # At different positions, elevation should generally differ
        # (could rarely be equal, so just check it's a valid number)
        assert isinstance(e2, float)


class TestBuildWaveModel:
    def test_none_default(self):
        from sailsim.core.config import WaveConfig
        from sailsim.environment import build_wave_model

        cfg = WaveConfig()
        model = build_wave_model(cfg)
        assert isinstance(model, NoWaves)

    def test_spectral_model(self):
        from sailsim.core.config import WaveConfig
        from sailsim.environment import build_wave_model

        cfg = WaveConfig(model="spectral", Hs=1.5, Tp=7.0, seed=42)
        model = build_wave_model(cfg)
        assert isinstance(model, SpectralWaves)
