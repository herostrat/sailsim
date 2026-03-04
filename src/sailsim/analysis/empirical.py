"""Data-driven stability analysis from recorded simulations.

Works with any autopilot type (Nomoto, pypilot, SignalK) by analysing
the recorded time-series data from ``Recorder``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import signal as sig

if TYPE_CHECKING:
    from sailsim.recording.recorder import Recorder


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class StepResponseMetrics:
    """Metrics for a single heading step response."""

    step_time_s: float
    step_magnitude_deg: float
    rise_time_s: float
    settling_time_s: float
    overshoot_pct: float
    steady_state_error_deg: float
    peak_rudder_deg: float


@dataclass
class SpectralEstimate:
    """Estimated transfer function from I/O data (Welch method)."""

    frequencies: np.ndarray
    magnitude_db: np.ndarray
    phase_deg: np.ndarray
    coherence: np.ndarray


@dataclass
class RudderActivityAnalysis:
    """Rudder activity spectral and saturation analysis."""

    frequencies_hz: np.ndarray
    psd: np.ndarray
    dominant_freq_hz: float
    total_rms_deg: float
    fraction_at_limit: float
    fraction_rate_limited: float


# ---------------------------------------------------------------------------
# Signal extraction
# ---------------------------------------------------------------------------


def _extract_signals(
    recorder: Recorder,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract time, heading, rudder, yaw_rate, target, speed from recorder."""
    t = np.array([s.t for s in recorder.steps])
    heading = np.array([s.sensors.heading for s in recorder.steps])
    rudder = np.array([s.control.rudder_angle for s in recorder.steps])
    yaw_rate = np.array([s.sensors.yaw_rate for s in recorder.steps])
    target = np.array(
        [
            s.target_heading if s.target_heading is not None else s.sensors.heading
            for s in recorder.steps
        ]
    )
    speed = np.array([s.sensors.speed_through_water for s in recorder.steps])
    return t, heading, rudder, yaw_rate, target, speed


def _wrap_error(error: np.ndarray) -> np.ndarray:
    """Wrap angle error to [-180, 180] degrees."""
    return (error + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------------------
# Step response analysis
# ---------------------------------------------------------------------------


def extract_step_responses(
    recorder: Recorder,
    min_step_deg: float = 5.0,
) -> list[StepResponseMetrics]:
    """Detect setpoint steps in target_heading and measure response.

    Args:
        recorder: Recorded simulation data.
        min_step_deg: Minimum step size to consider [deg].

    Returns:
        List of metrics for each detected step.
    """
    t, heading, rudder, _, target, _ = _extract_signals(recorder)

    # Convert to degrees
    heading_deg = np.degrees(heading)
    rudder_deg = np.degrees(rudder)
    target_deg = np.degrees(target)

    # Unwrap target to detect steps
    target_uw = np.unwrap(target_deg, period=360.0)

    # Find steps: large changes between consecutive samples
    dtarget = np.diff(target_uw)
    step_indices = np.where(np.abs(dtarget) >= min_step_deg)[0]

    results: list[StepResponseMetrics] = []

    for si in step_indices:
        step_time = t[si + 1]
        step_mag = dtarget[si]
        new_target = target_uw[si + 1]

        # Window: from step to next step or end
        end_idx = len(t)
        mask = step_indices > si
        if np.any(mask):
            end_idx = step_indices[mask][0] + 1

        t_win = t[si + 1 : end_idx] - step_time
        heading_win = np.unwrap(heading_deg[si + 1 : end_idx], period=360.0)
        rudder_win = rudder_deg[si + 1 : end_idx]

        if len(t_win) < 3:
            continue

        error_win = _wrap_error(new_target - heading_win)

        # Rise time: time to reach 90% of step
        threshold_90 = 0.1 * abs(step_mag)
        reached = np.where(np.abs(error_win) <= threshold_90)[0]
        rise_time = t_win[reached[0]] if len(reached) > 0 else t_win[-1]

        # Settling time: time after which error stays within 5% of step
        threshold_5 = 0.05 * abs(step_mag)
        settled_mask = np.abs(error_win) <= max(threshold_5, 1.0)
        settling_time = t_win[-1]
        # Find last index where NOT settled, settling = time after that
        not_settled = np.where(~settled_mask)[0]
        if len(not_settled) > 0 and not_settled[-1] + 1 < len(t_win):
            settling_time = t_win[not_settled[-1] + 1]
        elif len(not_settled) == 0:
            settling_time = 0.0

        # Overshoot
        if abs(step_mag) > 0:
            if step_mag > 0:
                overshoot = np.max(heading_win - heading_win[0]) - abs(step_mag)
            else:
                overshoot = abs(step_mag) - np.min(heading_win - heading_win[0])
            overshoot_pct = max(0.0, overshoot / abs(step_mag) * 100.0)
        else:
            overshoot_pct = 0.0

        # Steady-state error (last 20% of window)
        tail_start = int(0.8 * len(error_win))
        ss_error = float(np.mean(np.abs(error_win[tail_start:])))

        # Peak rudder
        peak_rudder = float(np.max(np.abs(rudder_win)))

        results.append(
            StepResponseMetrics(
                step_time_s=step_time,
                step_magnitude_deg=float(step_mag),
                rise_time_s=float(rise_time),
                settling_time_s=float(settling_time),
                overshoot_pct=float(overshoot_pct),
                steady_state_error_deg=ss_error,
                peak_rudder_deg=peak_rudder,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Spectral transfer function estimation
# ---------------------------------------------------------------------------


def estimate_transfer_function(
    recorder: Recorder,
    nperseg: int = 256,
) -> SpectralEstimate:
    """Estimate H(f) = Sxy/Sxx via Welch cross-spectral method.

    Input: rudder_angle, Output: heading.
    Coherence gamma^2 = |Sxy|^2 / (Sxx * Syy) indicates linearity.
    """
    t, heading, rudder, _, _, _ = _extract_signals(recorder)

    dt = np.median(np.diff(t))
    fs = 1.0 / dt

    # Ensure nperseg doesn't exceed signal length
    nperseg = min(nperseg, len(rudder) // 2)
    if nperseg < 8:
        nperseg = 8

    # Remove mean
    rudder_dm = rudder - np.mean(rudder)
    heading_dm = heading - np.mean(heading)

    # Auto-spectral density of input
    f_xx, Sxx = sig.welch(rudder_dm, fs=fs, nperseg=nperseg)
    # Cross-spectral density
    _f_xy, Sxy = sig.csd(rudder_dm, heading_dm, fs=fs, nperseg=nperseg)
    # Auto-spectral density of output
    _f_yy, Syy = sig.welch(heading_dm, fs=fs, nperseg=nperseg)

    # Transfer function estimate H(f) = Sxy / Sxx
    H = Sxy / (Sxx + 1e-30)
    magnitude_db = 20.0 * np.log10(np.abs(H) + 1e-30)
    phase_deg = np.degrees(np.angle(H))

    # Coherence
    coherence = np.abs(Sxy) ** 2 / (Sxx * Syy + 1e-30)
    coherence = np.clip(coherence, 0.0, 1.0)

    return SpectralEstimate(
        frequencies=f_xx,
        magnitude_db=magnitude_db,
        phase_deg=phase_deg,
        coherence=coherence,
    )


# ---------------------------------------------------------------------------
# Rudder activity analysis
# ---------------------------------------------------------------------------


def analyze_rudder_activity(
    recorder: Recorder,
    position_limit_deg: float = 30.0,
    rate_limit_deg_s: float = 5.0,
) -> RudderActivityAnalysis:
    """Analyse rudder activity: PSD, dominant frequency, saturation.

    Args:
        recorder: Recorded simulation data.
        position_limit_deg: Rudder position limit [deg].
        rate_limit_deg_s: Rudder rate limit [deg/s].
    """
    t, _, rudder, _, _, _ = _extract_signals(recorder)

    rudder_deg = np.degrees(rudder)
    dt = np.median(np.diff(t))
    fs = 1.0 / dt

    # PSD via Welch
    nperseg = min(256, len(rudder_deg) // 2)
    if nperseg < 8:
        nperseg = 8
    freq, psd = sig.welch(rudder_deg - np.mean(rudder_deg), fs=fs, nperseg=nperseg)

    # Dominant frequency (excluding DC)
    if len(freq) > 1:
        psd_no_dc = psd[1:]
        dominant_idx = np.argmax(psd_no_dc) + 1
        dominant_freq = float(freq[dominant_idx])
    else:
        dominant_freq = 0.0

    # RMS
    total_rms = float(np.sqrt(np.mean(rudder_deg**2)))

    # Position saturation: fraction of time at or beyond limit
    at_limit = np.abs(rudder_deg) >= position_limit_deg * 0.95
    fraction_at_limit = float(np.mean(at_limit))

    # Rate saturation: fraction of time rudder rate exceeds limit
    rudder_rate = np.abs(np.diff(rudder_deg) / dt)
    rate_limited = rudder_rate >= rate_limit_deg_s * 0.95
    fraction_rate_limited = float(np.mean(rate_limited))

    return RudderActivityAnalysis(
        frequencies_hz=freq,
        psd=psd,
        dominant_freq_hz=dominant_freq,
        total_rms_deg=total_rms,
        fraction_at_limit=fraction_at_limit,
        fraction_rate_limited=fraction_rate_limited,
    )
