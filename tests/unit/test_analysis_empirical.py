"""Tests for data-driven empirical stability analysis."""

from __future__ import annotations

import numpy as np

from sailsim.analysis.empirical import (
    RudderActivityAnalysis,
    SpectralEstimate,
    StepResponseMetrics,
    analyze_rudder_activity,
    estimate_transfer_function,
    extract_step_responses,
)
from sailsim.core.types import (
    ControlCommand,
    SensorData,
    VesselState,
    WindState,
)
from sailsim.recording.recorder import Recorder, TimeStep

# -------------------------------------------------------------------
# Helpers to build synthetic recorders
# -------------------------------------------------------------------

def _make_recorder(
    duration: float = 60.0,
    dt: float = 0.1,
    heading_fn=None,
    rudder_fn=None,
    target_fn=None,
    speed: float = 3.0,
) -> Recorder:
    """Build a synthetic recorder with custom heading/rudder signals."""
    rec = Recorder()
    t = np.arange(0, duration, dt)

    for ti in t:
        heading = heading_fn(ti) if heading_fn else 0.0
        rudder = rudder_fn(ti) if rudder_fn else 0.0
        target = target_fn(ti) if target_fn else heading

        eta = np.zeros(6)
        eta[5] = heading
        nu = np.zeros(6)
        nu[0] = speed

        rec.steps.append(TimeStep(
            t=ti,
            state=VesselState(eta=eta.copy(), nu=nu.copy()),
            sensors=SensorData(
                heading=heading,
                speed_through_water=speed,
                yaw_rate=0.0,
            ),
            control=ControlCommand(rudder_angle=rudder),
            wind=WindState(speed=5.0, direction=np.pi / 2),
            target_heading=target,
        ))

    return rec


def _make_step_recorder(
    step_time: float = 10.0,
    step_deg: float = 30.0,
    rise_time: float = 5.0,
    overshoot_frac: float = 0.1,
    duration: float = 60.0,
    dt: float = 0.1,
) -> Recorder:
    """Build a recorder with a known step response."""
    step_rad = np.radians(step_deg)

    def heading_fn(t: float) -> float:
        if t < step_time:
            return 0.0
        elapsed = t - step_time
        # Simple second-order response approximation
        wn = 1.8 / rise_time  # approximate for 10-90% rise time
        zeta = -np.log(overshoot_frac) / np.sqrt(np.pi**2 + np.log(overshoot_frac)**2)
        wd = wn * np.sqrt(1 - zeta**2)
        if wd > 0:
            envelope = np.exp(-zeta * wn * elapsed)
            response = 1.0 - envelope * (
                np.cos(wd * elapsed) + (zeta * wn / wd) * np.sin(wd * elapsed)
            )
        else:
            response = 1.0 - np.exp(-wn * elapsed)
        return step_rad * response

    def target_fn(t: float) -> float:
        return step_rad if t >= step_time else 0.0

    def rudder_fn(t: float) -> float:
        if t < step_time:
            return 0.0
        elapsed = t - step_time
        # Decaying rudder input
        return np.radians(15.0) * np.exp(-elapsed / rise_time)

    return _make_recorder(
        duration=duration, dt=dt,
        heading_fn=heading_fn, rudder_fn=rudder_fn, target_fn=target_fn,
    )


# -------------------------------------------------------------------
# Step response tests
# -------------------------------------------------------------------

class TestExtractStepResponses:
    def test_detects_step(self) -> None:
        rec = _make_step_recorder(step_time=10.0, step_deg=30.0)
        steps = extract_step_responses(rec, min_step_deg=5.0)
        assert len(steps) == 1
        assert isinstance(steps[0], StepResponseMetrics)

    def test_step_magnitude(self) -> None:
        rec = _make_step_recorder(step_deg=30.0)
        steps = extract_step_responses(rec, min_step_deg=5.0)
        assert abs(steps[0].step_magnitude_deg - 30.0) < 1.0

    def test_rise_time_reasonable(self) -> None:
        rec = _make_step_recorder(rise_time=5.0, duration=60.0)
        steps = extract_step_responses(rec, min_step_deg=5.0)
        # Rise time should be within 2x of designed value
        assert 1.0 < steps[0].rise_time_s < 15.0

    def test_overshoot_detected(self) -> None:
        rec = _make_step_recorder(overshoot_frac=0.2)
        steps = extract_step_responses(rec, min_step_deg=5.0)
        assert steps[0].overshoot_pct > 0

    def test_small_step_ignored(self) -> None:
        rec = _make_step_recorder(step_deg=2.0)
        steps = extract_step_responses(rec, min_step_deg=5.0)
        assert len(steps) == 0

    def test_peak_rudder(self) -> None:
        rec = _make_step_recorder()
        steps = extract_step_responses(rec, min_step_deg=5.0)
        assert steps[0].peak_rudder_deg > 0

    def test_no_steps_empty_list(self) -> None:
        rec = _make_recorder(heading_fn=lambda t: 0.0, target_fn=lambda t: 0.0)
        steps = extract_step_responses(rec, min_step_deg=5.0)
        assert len(steps) == 0


# -------------------------------------------------------------------
# Spectral estimation tests
# -------------------------------------------------------------------

class TestEstimateTransferFunction:
    def test_returns_spectral_estimate(self) -> None:
        rec = _make_recorder(
            duration=30.0, dt=0.05,
            heading_fn=lambda t: 0.01 * np.sin(0.5 * t),
            rudder_fn=lambda t: 0.02 * np.sin(0.5 * t),
        )
        result = estimate_transfer_function(rec, nperseg=64)
        assert isinstance(result, SpectralEstimate)
        assert len(result.frequencies) > 0
        assert len(result.magnitude_db) == len(result.frequencies)
        assert len(result.phase_deg) == len(result.frequencies)
        assert len(result.coherence) == len(result.frequencies)

    def test_coherence_high_for_linear(self) -> None:
        """Pure sinusoidal I/O should give high coherence at that frequency."""
        freq = 0.5  # Hz
        omega = 2 * np.pi * freq
        rec = _make_recorder(
            duration=60.0, dt=0.05,
            heading_fn=lambda t: 0.5 * np.sin(omega * t),
            rudder_fn=lambda t: 0.1 * np.sin(omega * t),
        )
        result = estimate_transfer_function(rec, nperseg=128)
        # Find coherence near the test frequency
        f_idx = np.argmin(np.abs(result.frequencies - freq))
        assert result.coherence[f_idx] > 0.8

    def test_short_signal_handled(self) -> None:
        """Very short signals should not crash."""
        rec = _make_recorder(duration=2.0, dt=0.1)
        result = estimate_transfer_function(rec, nperseg=256)
        assert len(result.frequencies) > 0


# -------------------------------------------------------------------
# Rudder activity tests
# -------------------------------------------------------------------

class TestAnalyzeRudderActivity:
    def test_returns_analysis(self) -> None:
        rec = _make_recorder(
            rudder_fn=lambda t: np.radians(10.0) * np.sin(0.3 * t),
        )
        result = analyze_rudder_activity(rec)
        assert isinstance(result, RudderActivityAnalysis)

    def test_sinusoidal_psd_peak(self) -> None:
        """Sinusoidal rudder should have PSD peak at that frequency."""
        freq_hz = 0.3
        omega = 2 * np.pi * freq_hz
        rec = _make_recorder(
            duration=60.0, dt=0.05,
            rudder_fn=lambda t: np.radians(10.0) * np.sin(omega * t),
        )
        result = analyze_rudder_activity(rec)
        # Dominant frequency should be near the input frequency
        assert abs(result.dominant_freq_hz - freq_hz) < 0.1

    def test_rms_nonzero_for_active_rudder(self) -> None:
        rec = _make_recorder(
            rudder_fn=lambda t: np.radians(10.0) * np.sin(0.5 * t),
        )
        result = analyze_rudder_activity(rec)
        assert result.total_rms_deg > 0

    def test_saturation_detected(self) -> None:
        """Rudder at limit should report high fraction_at_limit."""
        rec = _make_recorder(
            rudder_fn=lambda t: np.radians(30.0),  # constant at limit
        )
        result = analyze_rudder_activity(rec, position_limit_deg=30.0)
        assert result.fraction_at_limit > 0.9

    def test_no_saturation_for_small_rudder(self) -> None:
        rec = _make_recorder(
            rudder_fn=lambda t: np.radians(5.0) * np.sin(0.3 * t),
        )
        result = analyze_rudder_activity(rec, position_limit_deg=30.0)
        assert result.fraction_at_limit < 0.05

    def test_rate_saturation(self) -> None:
        """Fast oscillations should trigger rate limiting."""
        # High frequency oscillation: rate = A*omega = 10*2pi*2 ≈ 125 deg/s
        rec = _make_recorder(
            duration=30.0, dt=0.05,
            rudder_fn=lambda t: np.radians(10.0) * np.sin(2 * np.pi * 2.0 * t),
        )
        result = analyze_rudder_activity(rec, rate_limit_deg_s=5.0)
        assert result.fraction_rate_limited > 0.5
