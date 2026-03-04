"""Model-based linear stability analysis using Nomoto transfer functions.

Builds continuous-time transfer functions from yacht hydrodynamics and
autopilot gains, then computes classical stability metrics (gain/phase
margins, poles, bandwidth) and their sensitivity to speed.

All transfer functions use ``scipy.signal.TransferFunction`` in the
continuous-time (Laplace) domain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy import signal

from sailsim.autopilot.nomoto import (
    NomotoParams,
    _compute_gains,
    estimate_nomoto_params,
)

if TYPE_CHECKING:
    from sailsim.core.config import YachtConfig


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class StabilityMargins:
    """Classical gain and phase margins of the open-loop system."""

    gain_margin_db: float
    phase_margin_deg: float
    delay_margin_s: float
    bandwidth_rad_s: float
    gain_crossover_freq: float
    phase_crossover_freq: float


@dataclass
class ClosedLoopPoles:
    """Closed-loop pole analysis."""

    poles: np.ndarray
    damping_ratios: np.ndarray
    natural_frequencies: np.ndarray
    is_stable: bool


@dataclass
class LinearAnalysisResult:
    """Complete linear analysis at a single operating point."""

    U: float
    nomoto: NomotoParams
    Kp: float
    Kd: float
    Ki: float
    plant_tf: signal.TransferFunction
    controller_tf: signal.TransferFunction
    open_loop_tf: signal.TransferFunction
    closed_loop_tf: signal.TransferFunction
    margins: StabilityMargins
    poles: ClosedLoopPoles


@dataclass
class SpeedSweepResult:
    """Stability margins and poles across a speed range."""

    speeds: np.ndarray
    K_values: np.ndarray
    T_values: np.ndarray
    Kp_values: np.ndarray
    Kd_values: np.ndarray
    Ki_values: np.ndarray
    gain_margins_db: np.ndarray
    phase_margins_deg: np.ndarray
    bandwidths: np.ndarray
    poles_list: list[np.ndarray] = field(default_factory=list)
    is_stable: np.ndarray = field(default_factory=lambda: np.array([]))


# ---------------------------------------------------------------------------
# Transfer function builders
# ---------------------------------------------------------------------------


def build_plant_tf(K: float, T: float) -> signal.TransferFunction:
    """1st-order Nomoto heading transfer function.

    G(s) = K / (s * (T*s + 1)) = K / (T*s^2 + s)

    Maps rudder angle delta to heading psi (includes 1/s integrator
    from yaw rate r to heading).
    """
    return signal.TransferFunction([K], [T, 1.0, 0.0])


def build_plant_tf_2nd(
    K: float,
    T1: float,
    T2: float,
    T3: float,
) -> signal.TransferFunction:
    """2nd-order Nomoto heading transfer function.

    G(s) = K * (T3*s + 1) / (s * (T1*s + 1) * (T2*s + 1))
    """
    num = np.polymul([K * T3, K], [1.0])  # K*(T3*s + 1)
    # den = s * (T1*s+1) * (T2*s+1)
    d12 = np.polymul([T1, 1.0], [T2, 1.0])  # (T1*s+1)*(T2*s+1)
    den = np.polymul(d12, [1.0, 0.0])  # * s
    return signal.TransferFunction(num, den)


def build_controller_tf(
    Kp: float,
    Kd: float,
    Ki: float,
) -> signal.TransferFunction:
    """PID controller transfer function.

    C(s) = Kp + Ki/s + Kd*s = (Kd*s^2 + Kp*s + Ki) / s
    """
    return signal.TransferFunction([Kd, Kp, Ki], [1.0, 0.0])


# ---------------------------------------------------------------------------
# Margin computation
# ---------------------------------------------------------------------------


def compute_margins(L: signal.TransferFunction) -> StabilityMargins:
    """Compute gain/phase margins from the open-loop transfer function.

    Uses frequency sweep to find crossover frequencies and margins.
    """
    w = np.logspace(-3, 2, 5000)
    _, H = signal.freqresp(L, w)

    mag = np.abs(H)
    phase = np.unwrap(np.angle(H))

    # --- Gain crossover: |L(jw)| = 1 ---
    gc_freq = 0.0
    pm_deg = 0.0
    log_mag = 20.0 * np.log10(mag + 1e-30)
    crossings_gc = np.where(np.diff(np.sign(log_mag)))[0]
    if len(crossings_gc) > 0:
        # Interpolate for last gain crossover
        idx = crossings_gc[-1]
        frac = -log_mag[idx] / (log_mag[idx + 1] - log_mag[idx] + 1e-30)
        gc_freq = w[idx] + frac * (w[idx + 1] - w[idx])
        phase_at_gc = phase[idx] + frac * (phase[idx + 1] - phase[idx])
        pm_deg = 180.0 + np.degrees(phase_at_gc)

    # --- Phase crossover: phase = -180 deg ---
    pc_freq = 0.0
    gm_db = float("inf")
    phase_shifted = phase + np.pi  # look for zero crossings
    crossings_pc = np.where(np.diff(np.sign(phase_shifted)))[0]
    if len(crossings_pc) > 0:
        idx = crossings_pc[0]
        frac = -phase_shifted[idx] / (phase_shifted[idx + 1] - phase_shifted[idx] + 1e-30)
        pc_freq = w[idx] + frac * (w[idx + 1] - w[idx])
        mag_at_pc = mag[idx] + frac * (mag[idx + 1] - mag[idx])
        gm_db = -20.0 * np.log10(mag_at_pc + 1e-30)

    # --- Bandwidth: |T(jw)| = -3dB (closed-loop) ---
    # T(s) = L / (1+L)
    T_mag = mag / np.abs(1.0 + H)
    T_db = 20.0 * np.log10(T_mag + 1e-30)
    bw = 0.0
    cross_bw = np.where(np.diff(np.sign(T_db - T_db[0] + 3.0)))[0]
    if len(cross_bw) > 0:
        bw = w[cross_bw[0]]

    # --- Delay margin ---
    delay_margin = np.radians(pm_deg) / gc_freq if gc_freq > 0 else float("inf")

    return StabilityMargins(
        gain_margin_db=float(gm_db),
        phase_margin_deg=float(pm_deg),
        delay_margin_s=float(delay_margin),
        bandwidth_rad_s=float(bw),
        gain_crossover_freq=float(gc_freq),
        phase_crossover_freq=float(pc_freq),
    )


def _analyze_poles(tf: signal.TransferFunction) -> ClosedLoopPoles:
    """Extract poles, damping ratios, and natural frequencies."""
    poles = np.roots(tf.den)
    wn = np.abs(poles)
    # Damping ratio: for pole at -sigma +/- j*wd,  zeta = sigma / wn
    zeta = -np.real(poles) / (wn + 1e-30)
    is_stable = bool(np.all(np.real(poles) < 0))
    return ClosedLoopPoles(
        poles=poles,
        damping_ratios=zeta,
        natural_frequencies=wn,
        is_stable=is_stable,
    )


# ---------------------------------------------------------------------------
# Main analysis functions
# ---------------------------------------------------------------------------


def analyze_at_speed(
    yacht: YachtConfig,
    U: float,
    omega_n: float = 0.5,
    zeta: float = 0.8,
) -> LinearAnalysisResult:
    """Full linear analysis at a single speed (frozen-point).

    Estimates Nomoto parameters, computes controller gains via pole
    placement, builds transfer functions, and computes stability margins
    and closed-loop poles.
    """
    nomoto = estimate_nomoto_params(yacht, U)
    Kp, Kd, Ki = _compute_gains(nomoto.K, nomoto.T, omega_n, zeta)

    G = build_plant_tf(nomoto.K, nomoto.T)
    C = build_controller_tf(Kp, Kd, Ki)

    # Open-loop: L = C * G
    L_num = np.polymul(C.num, G.num)
    L_den = np.polymul(C.den, G.den)
    L = signal.TransferFunction(L_num, L_den)

    # Closed-loop: T = L / (1 + L)
    T_num = L_num
    T_den = np.polyadd(L_den, L_num)
    T_cl = signal.TransferFunction(T_num, T_den)

    margins = compute_margins(L)
    poles = _analyze_poles(T_cl)

    return LinearAnalysisResult(
        U=U,
        nomoto=nomoto,
        Kp=Kp,
        Kd=Kd,
        Ki=Ki,
        plant_tf=G,
        controller_tf=C,
        open_loop_tf=L,
        closed_loop_tf=T_cl,
        margins=margins,
        poles=poles,
    )


def sweep_speed(
    yacht: YachtConfig,
    U_range: np.ndarray,
    omega_n: float = 0.5,
    zeta: float = 0.8,
) -> SpeedSweepResult:
    """Frozen-point stability analysis across a speed range."""
    n = len(U_range)
    result = SpeedSweepResult(
        speeds=np.array(U_range),
        K_values=np.zeros(n),
        T_values=np.zeros(n),
        Kp_values=np.zeros(n),
        Kd_values=np.zeros(n),
        Ki_values=np.zeros(n),
        gain_margins_db=np.zeros(n),
        phase_margins_deg=np.zeros(n),
        bandwidths=np.zeros(n),
        poles_list=[],
        is_stable=np.zeros(n, dtype=bool),
    )

    for i, U in enumerate(U_range):
        r = analyze_at_speed(yacht, float(U), omega_n, zeta)
        result.K_values[i] = r.nomoto.K
        result.T_values[i] = r.nomoto.T
        result.Kp_values[i] = r.Kp
        result.Kd_values[i] = r.Kd
        result.Ki_values[i] = r.Ki
        result.gain_margins_db[i] = r.margins.gain_margin_db
        result.phase_margins_deg[i] = r.margins.phase_margin_deg
        result.bandwidths[i] = r.margins.bandwidth_rad_s
        result.poles_list.append(r.poles.poles)
        result.is_stable[i] = r.poles.is_stable

    return result


# ---------------------------------------------------------------------------
# Describing function for rate limiter
# ---------------------------------------------------------------------------


def describing_function_rate_limiter(
    rate_limit: float,
    amplitude: float,
    omega: float,
) -> complex:
    """Sinusoidal-input describing function (SIDF) of a rate limiter.

    For sinusoidal input delta(t) = A*sin(wt) with rate limit R:
    - If A*w <= R: no clipping, N = 1
    - If A*w > R: gain < 1 with phase lag

    Args:
        rate_limit: Maximum rate R [units/s].
        amplitude: Sinusoid amplitude A [units].
        omega: Sinusoid frequency w [rad/s].

    Returns:
        Complex describing function N(A, w).
    """
    if amplitude <= 0 or omega <= 0:
        return complex(1.0, 0.0)

    max_rate = amplitude * omega
    if max_rate <= rate_limit:
        return complex(1.0, 0.0)

    # Ratio of rate limit to max input rate
    d = rate_limit / max_rate  # 0 < d < 1

    # SIDF for ideal rate limiter (Gelb & Vander Velde, 1968)
    # N = (2/pi) * (arcsin(d) + d*sqrt(1-d^2)) - j*(2/pi)*d^2 ...
    # Simplified form for first harmonic:
    phi = np.arcsin(d)
    gain = (2.0 / np.pi) * (phi + d * np.sqrt(1.0 - d**2))
    phase = -(2.0 / np.pi) * (1.0 - np.sqrt(1.0 - d**2))

    return complex(gain, phase)
