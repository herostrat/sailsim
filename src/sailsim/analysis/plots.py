"""Visualisation functions for stability analysis results.

All plot functions return the matplotlib ``Figure`` for optional saving.
Uses the ``Agg`` backend by default; call ``plt.show()`` explicitly
if interactive display is needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import signal

if TYPE_CHECKING:
    from sailsim.analysis.empirical import (
        RudderActivityAnalysis,
        SpectralEstimate,
        StepResponseMetrics,
    )
    from sailsim.analysis.linear import (
        LinearAnalysisResult,
        SpeedSweepResult,
    )
    from sailsim.recording.recorder import Recorder


def _import_plt() -> Any:
    import matplotlib.pyplot as plt

    return plt


# ===================================================================
# Model-based plots
# ===================================================================

def plot_bode(
    result: LinearAnalysisResult,
    which: str = "open_loop",
) -> Any:
    """Bode plot (magnitude + phase) of the specified transfer function.

    Args:
        result: Analysis result from ``analyze_at_speed``.
        which: ``"open_loop"``, ``"closed_loop"``, ``"plant"``, or
               ``"controller"``.
    """
    plt = _import_plt()

    tf_map = {
        "open_loop": result.open_loop_tf,
        "closed_loop": result.closed_loop_tf,
        "plant": result.plant_tf,
        "controller": result.controller_tf,
    }
    sys = tf_map[which]

    w = np.logspace(-3, 2, 2000)
    _, H = signal.freqresp(sys, w)
    mag_db = 20.0 * np.log10(np.abs(H) + 1e-30)
    phase_deg = np.degrees(np.unwrap(np.angle(H)))

    fig, (ax_mag, ax_ph) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    fig.suptitle(f"Bode — {which.replace('_', ' ').title()} (U={result.U:.1f} m/s)")

    ax_mag.semilogx(w, mag_db, "b-", linewidth=1.5)
    ax_mag.axhline(0, color="k", linestyle="--", linewidth=0.5)
    ax_mag.set_ylabel("Magnitude [dB]")
    ax_mag.grid(True, which="both", alpha=0.3)

    ax_ph.semilogx(w, phase_deg, "r-", linewidth=1.5)
    ax_ph.axhline(-180, color="k", linestyle="--", linewidth=0.5)
    ax_ph.set_ylabel("Phase [deg]")
    ax_ph.set_xlabel("Frequency [rad/s]")
    ax_ph.grid(True, which="both", alpha=0.3)

    # Annotate margins for open-loop
    if which == "open_loop":
        m = result.margins
        if m.gain_crossover_freq > 0:
            ax_ph.axvline(
                m.gain_crossover_freq, color="g", linestyle=":",
                label=f"PM={m.phase_margin_deg:.1f} deg",
            )
            ax_ph.legend(loc="best", fontsize=8)
        if m.phase_crossover_freq > 0:
            ax_mag.axvline(
                m.phase_crossover_freq, color="m", linestyle=":",
                label=f"GM={m.gain_margin_db:.1f} dB",
            )
            ax_mag.legend(loc="best", fontsize=8)
        elif np.isinf(m.gain_margin_db):
            ax_mag.text(
                0.98, 0.95, "GM = inf",
                transform=ax_mag.transAxes, ha="right", va="top",
                fontsize=9, color="m",
                bbox={"boxstyle": "round,pad=0.3", "fc": "white", "alpha": 0.8},
            )

    fig.tight_layout()
    return fig


def plot_nyquist(
    result: LinearAnalysisResult,
    show_describing_function: bool = False,
    rate_limit: float = 0.087,
) -> Any:
    """Nyquist plot of the open-loop transfer function.

    Args:
        result: Analysis result.
        show_describing_function: If True, overlay -1/N(A,w) locus.
        rate_limit: Rudder rate limit [rad/s] for describing function.
    """
    plt = _import_plt()
    from sailsim.analysis.linear import describing_function_rate_limiter

    w = np.logspace(-3, 2, 3000)
    _, H = signal.freqresp(result.open_loop_tf, w)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_title(f"Nyquist — Open Loop (U={result.U:.1f} m/s)")

    ax.plot(np.real(H), np.imag(H), "b-", linewidth=1.5, label="L(jw)")
    ax.plot(np.real(H), -np.imag(H), "b--", linewidth=0.8, alpha=0.5)
    ax.plot(-1, 0, "rx", markersize=12, markeredgewidth=2, label="(-1, 0)")

    if show_describing_function:
        amplitudes = np.linspace(0.01, 0.5, 50)
        for w_test in [0.1, 0.3, 0.5, 1.0]:
            neg_inv_N = []
            for A in amplitudes:
                N = describing_function_rate_limiter(rate_limit, A, w_test)
                if abs(N) > 1e-10:
                    neg_inv_N.append(-1.0 / N)
            if neg_inv_N:
                pts = np.array(neg_inv_N)
                ax.plot(
                    np.real(pts), np.imag(pts), "--",
                    linewidth=1, label=f"-1/N (w={w_test})",
                )

    # Auto-zoom around the critical point (-1, 0).
    # Find where the contour is closest to (-1, 0) and set limits
    # to show that region with comfortable margin.
    dist_to_crit = np.abs(H + 1.0)
    closest_mag = np.abs(H[np.argmin(dist_to_crit)])
    # Show a region that includes the critical point and the nearby contour
    radius = max(closest_mag * 1.5, 3.0)
    ax.set_xlim(-1 - radius, -1 + radius)
    ax.set_ylim(-radius, radius)

    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def plot_root_locus(
    result: LinearAnalysisResult,
    gain_range: tuple[float, float] = (0.1, 10.0),
    n_points: int = 200,
) -> Any:
    """Root locus: pole trajectories as gain varies.

    Varies the proportional gain Kp from gain_range[0]*Kp to gain_range[1]*Kp.
    """
    plt = _import_plt()
    from sailsim.analysis.linear import build_controller_tf, build_plant_tf

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_title(f"Root Locus (U={result.U:.1f} m/s)")

    gains = np.linspace(gain_range[0], gain_range[1], n_points) * result.Kp
    all_poles = []

    for Kp_var in gains:
        # Scale Kd and Ki proportionally
        scale = Kp_var / result.Kp
        Kd_var = result.Kd * scale
        Ki_var = result.Ki * scale
        C = build_controller_tf(Kp_var, Kd_var, Ki_var)
        G = build_plant_tf(result.nomoto.K, result.nomoto.T)
        L_num = np.polymul(C.num, G.num)
        L_den = np.polymul(C.den, G.den)
        T_den = np.polyadd(L_den, L_num)
        poles = np.roots(T_den)
        all_poles.append(poles)

    all_poles_arr = np.array(all_poles)
    for j in range(all_poles_arr.shape[1]):
        ax.plot(
            np.real(all_poles_arr[:, j]),
            np.imag(all_poles_arr[:, j]),
            ".", markersize=2,
        )

    # Mark nominal poles
    for p in result.poles.poles:
        ax.plot(np.real(p), np.imag(p), "kx", markersize=10, markeredgewidth=2)

    ax.axvline(0, color="k", linewidth=0.5)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_pole_zero_map(
    result: LinearAnalysisResult,
) -> Any:
    """Pole-zero map with damping lines and natural frequency circles."""
    plt = _import_plt()

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_title(f"Pole-Zero Map (U={result.U:.1f} m/s)")

    poles = result.poles.poles
    ax.plot(np.real(poles), np.imag(poles), "bx", markersize=12, markeredgewidth=2,
            label="Poles")

    # Zeros of closed-loop
    zeros = np.roots(result.closed_loop_tf.num)
    if len(zeros) > 0:
        ax.plot(np.real(zeros), np.imag(zeros), "ro", markersize=10,
                fillstyle="none", markeredgewidth=2, label="Zeros")

    # Damping ratio lines
    wn_max = max(np.abs(poles).max() * 1.3, 1.0)
    for z in [0.2, 0.4, 0.6, 0.8]:
        theta = np.arccos(z)
        ax.plot(
            [-wn_max * np.cos(theta), 0], [wn_max * np.sin(theta), 0],
            "k--", alpha=0.2, linewidth=0.8,
        )
        ax.plot(
            [-wn_max * np.cos(theta), 0], [-wn_max * np.sin(theta), 0],
            "k--", alpha=0.2, linewidth=0.8,
        )
        ax.text(
            -wn_max * np.cos(theta) * 0.5, wn_max * np.sin(theta) * 0.5,
            f"z={z}", fontsize=7, alpha=0.4,
        )

    # Natural frequency circles
    for wn in np.arange(0.2, wn_max, 0.2):
        circle = plt.Circle((0, 0), wn, fill=False, linestyle=":",
                             color="gray", alpha=0.2)
        ax.add_patch(circle)

    ax.axvline(0, color="k", linewidth=0.5)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_speed_sensitivity(
    sweep: SpeedSweepResult,
) -> Any:
    """4-panel plot: K/T vs U, gains vs U, GM/PM vs U, pole trajectories."""
    plt = _import_plt()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Speed Sensitivity Analysis")

    # Panel 1: Nomoto K and T
    ax = axes[0, 0]
    ax.plot(sweep.speeds, sweep.K_values, "b-o", markersize=3, label="K")
    ax.set_ylabel("K [1/s]", color="b")
    ax.tick_params(axis="y", labelcolor="b")
    ax2 = ax.twinx()
    ax2.plot(sweep.speeds, sweep.T_values, "r-s", markersize=3, label="T")
    ax2.set_ylabel("T [s]", color="r")
    ax2.tick_params(axis="y", labelcolor="r")
    ax.set_xlabel("Speed [m/s]")
    ax.set_title("Nomoto Parameters")
    ax.grid(True, alpha=0.3)

    # Panel 2: Controller gains
    ax = axes[0, 1]
    ax.plot(sweep.speeds, sweep.Kp_values, "b-o", markersize=3, label="Kp")
    ax.plot(sweep.speeds, sweep.Kd_values, "r-s", markersize=3, label="Kd")
    ax.plot(sweep.speeds, sweep.Ki_values, "g-^", markersize=3, label="Ki")
    ax.set_xlabel("Speed [m/s]")
    ax.set_ylabel("Gain")
    ax.set_title("Controller Gains")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: GM and PM
    ax = axes[1, 0]
    gm = np.clip(sweep.gain_margins_db, -50, 100)
    ax.plot(sweep.speeds, gm, "b-o", markersize=3, label="GM")
    ax.set_ylabel("Gain Margin [dB]", color="b")
    ax.tick_params(axis="y", labelcolor="b")
    ax3 = ax.twinx()
    ax3.plot(sweep.speeds, sweep.phase_margins_deg, "r-s", markersize=3, label="PM")
    ax3.set_ylabel("Phase Margin [deg]", color="r")
    ax3.tick_params(axis="y", labelcolor="r")
    ax.set_xlabel("Speed [m/s]")
    ax.set_title("Stability Margins")
    ax.grid(True, alpha=0.3)

    # Panel 4: Pole trajectories
    ax = axes[1, 1]
    for pole_set in sweep.poles_list:
        ax.plot(np.real(pole_set), np.imag(pole_set), "b.", markersize=4)
    ax.axvline(0, color="k", linewidth=0.5)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    ax.set_title("Pole Trajectories")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# ===================================================================
# Empirical plots
# ===================================================================

def plot_step_response(
    steps: list[StepResponseMetrics],
    recorder: Recorder | None = None,
) -> Any:
    """Step response plot with annotations."""
    plt = _import_plt()

    fig, axes = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    fig.suptitle("Step Response Analysis")

    if recorder is not None:
        t = np.array([s.t for s in recorder.steps])
        heading_deg = np.degrees(np.array([s.sensors.heading for s in recorder.steps]))
        rudder_deg = np.degrees(np.array([s.control.rudder_angle for s in recorder.steps]))
        target_deg = np.degrees(np.array([
            s.target_heading if s.target_heading is not None else s.sensors.heading
            for s in recorder.steps
        ]))

        axes[0].plot(t, heading_deg, "b-", linewidth=1, label="Heading")
        axes[0].plot(t, target_deg, "k--", linewidth=1, label="Target")
        axes[1].plot(t, rudder_deg, "r-", linewidth=1, label="Rudder")

    # Annotate steps
    for s in steps:
        axes[0].axvline(s.step_time_s, color="g", linestyle=":", alpha=0.5)
        axes[0].annotate(
            f"RT={s.rise_time_s:.1f}s\nOS={s.overshoot_pct:.0f}%\n"
            f"ST={s.settling_time_s:.1f}s",
            xy=(s.step_time_s, 0), xycoords=("data", "axes fraction"),
            fontsize=7, va="bottom",
        )

    axes[0].set_ylabel("Heading [deg]")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_ylabel("Rudder [deg]")
    axes[1].set_xlabel("Time [s]")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_estimated_bode(
    estimate: SpectralEstimate,
    model_result: LinearAnalysisResult | None = None,
) -> Any:
    """Empirical Bode plot with coherence panel and optional model overlay."""
    plt = _import_plt()

    fig, (ax_mag, ax_ph, ax_coh) = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
    fig.suptitle("Empirical Transfer Function Estimate")

    f = estimate.frequencies
    mask = f > 0  # skip DC

    ax_mag.semilogx(f[mask], estimate.magnitude_db[mask], "b-", linewidth=1.5,
                    label="Empirical")
    ax_ph.semilogx(f[mask], estimate.phase_deg[mask], "r-", linewidth=1.5,
                   label="Empirical")
    ax_coh.semilogx(f[mask], estimate.coherence[mask], "g-", linewidth=1.5)

    if model_result is not None:
        w = f[mask] * 2 * np.pi  # Hz to rad/s
        _, H = signal.freqresp(
            model_result.closed_loop_tf if hasattr(model_result, "plant_tf")
            else model_result.plant_tf,
            w,
        )
        ax_mag.semilogx(f[mask], 20.0 * np.log10(np.abs(H) + 1e-30),
                        "b--", alpha=0.6, label="Model")
        ax_ph.semilogx(f[mask], np.degrees(np.unwrap(np.angle(H))),
                       "r--", alpha=0.6, label="Model")

    ax_mag.set_ylabel("Magnitude [dB]")
    ax_mag.legend(fontsize=8)
    ax_mag.grid(True, which="both", alpha=0.3)

    ax_ph.set_ylabel("Phase [deg]")
    ax_ph.legend(fontsize=8)
    ax_ph.grid(True, which="both", alpha=0.3)

    ax_coh.set_ylabel("Coherence")
    ax_coh.set_xlabel("Frequency [Hz]")
    ax_coh.set_ylim(0, 1.05)
    ax_coh.axhline(0.8, color="k", linestyle="--", linewidth=0.5, alpha=0.5)
    ax_coh.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    return fig


def plot_rudder_spectrum(
    analysis: RudderActivityAnalysis,
) -> Any:
    """Rudder PSD with dominant frequency annotation."""
    plt = _import_plt()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title("Rudder Activity Spectrum")

    mask = analysis.frequencies_hz > 0
    ax.semilogy(analysis.frequencies_hz[mask], analysis.psd[mask], "b-", linewidth=1.5)

    if analysis.dominant_freq_hz > 0:
        ax.axvline(
            analysis.dominant_freq_hz, color="r", linestyle="--",
            label=f"Dominant: {analysis.dominant_freq_hz:.3f} Hz",
        )

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD [deg^2/Hz]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_saturation_summary(
    analysis: RudderActivityAnalysis,
) -> Any:
    """Bar chart of rudder saturation statistics."""
    plt = _import_plt()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_title("Rudder Saturation Summary")

    labels = ["At position limit", "At rate limit"]
    values = [analysis.fraction_at_limit * 100, analysis.fraction_rate_limited * 100]
    colors = ["#2196F3", "#FF5722"]

    bars = ax.bar(labels, values, color=colors, width=0.5)
    for bar, val in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{val:.1f}%", ha="center", fontsize=10,
        )

    ax.set_ylabel("Time at limit [%]")
    ax.set_ylim(0, max(max(values) * 1.3, 10))
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig
