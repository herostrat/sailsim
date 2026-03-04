"""Text reports and summaries for stability analysis results."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sailsim.analysis.empirical import RudderActivityAnalysis, StepResponseMetrics
    from sailsim.analysis.linear import LinearAnalysisResult, SpeedSweepResult


def summarize_linear(result: LinearAnalysisResult) -> str:
    """Human-readable summary of a linear analysis result."""
    lines = [
        f"Linear Analysis at U = {result.U:.2f} m/s",
        "=" * 50,
        f"Nomoto:  K = {result.nomoto.K:.4f}  T = {result.nomoto.T:.2f} s",
        f"Gains:   Kp = {result.Kp:.3f}  Kd = {result.Kd:.3f}  Ki = {result.Ki:.4f}",
        "",
        "Stability Margins:",
        f"  Gain Margin:  {result.margins.gain_margin_db:.1f} dB",
        f"  Phase Margin: {result.margins.phase_margin_deg:.1f} deg",
        f"  Delay Margin: {result.margins.delay_margin_s:.3f} s",
        f"  Bandwidth:    {result.margins.bandwidth_rad_s:.3f} rad/s",
        "",
        "Closed-Loop Poles:",
    ]

    for i, p in enumerate(result.poles.poles):
        z = result.poles.damping_ratios[i]
        wn = result.poles.natural_frequencies[i]
        if np.imag(p) != 0:
            lines.append(
                f"  {np.real(p):+.4f} +/- {abs(np.imag(p)):.4f}j (zeta={z:.3f}, wn={wn:.3f})"
            )
        else:
            lines.append(f"  {np.real(p):+.4f} (zeta={z:.3f}, wn={wn:.3f})")

    status = "STABLE" if result.poles.is_stable else "UNSTABLE"
    lines.append(f"\nSystem: {status}")
    return "\n".join(lines)


def summarize_speed_sweep(sweep: SpeedSweepResult) -> str:
    """Tabulated sweep results with critical-value markers."""
    lines = [
        "Speed Sweep Analysis",
        "=" * 75,
        f"{'U [m/s]':>8} {'K':>8} {'Kp':>8} {'Kd':>8} {'GM [dB]':>8} {'PM [deg]':>8} {'Stable':>7}",
        "-" * 75,
    ]

    for i, U in enumerate(sweep.speeds):
        gm = sweep.gain_margins_db[i]
        pm = sweep.phase_margins_deg[i]
        stable = "YES" if sweep.is_stable[i] else " NO*"

        # Mark critical values
        gm_str = f"{gm:8.1f}"
        pm_str = f"{pm:8.1f}"
        if gm < 6.0:
            gm_str += "!"
        if pm < 30.0:
            pm_str += "!"

        lines.append(
            f"{U:8.2f} {sweep.K_values[i]:8.4f} {sweep.Kp_values[i]:8.3f} "
            f"{sweep.Kd_values[i]:8.3f} {gm_str:>9} {pm_str:>9} {stable:>7}"
        )

    lines.append("-" * 75)
    n_stable = int(np.sum(sweep.is_stable))
    lines.append(f"Stable: {n_stable}/{len(sweep.speeds)} operating points")
    return "\n".join(lines)


def summarize_empirical(
    steps: list[StepResponseMetrics],
    rudder_analysis: RudderActivityAnalysis | None = None,
) -> str:
    """Text report of empirical analysis results."""
    lines = [
        "Empirical Analysis",
        "=" * 60,
    ]

    if steps:
        lines.append(f"\nStep Responses ({len(steps)} detected):")
        lines.append(
            f"{'t [s]':>8} {'Step [deg]':>10} {'Rise [s]':>9} "
            f"{'Settle [s]':>10} {'OS [%]':>7} {'SS err':>7} {'Pk rud':>7}"
        )
        lines.append("-" * 60)
        for s in steps:
            lines.append(
                f"{s.step_time_s:8.1f} {s.step_magnitude_deg:10.1f} "
                f"{s.rise_time_s:9.2f} {s.settling_time_s:10.2f} "
                f"{s.overshoot_pct:7.1f} {s.steady_state_error_deg:7.2f} "
                f"{s.peak_rudder_deg:7.1f}"
            )
    else:
        lines.append("\nNo step responses detected.")

    if rudder_analysis is not None:
        lines.extend(
            [
                "",
                "Rudder Activity:",
                f"  RMS:             {rudder_analysis.total_rms_deg:.2f} deg",
                f"  Dominant freq:   {rudder_analysis.dominant_freq_hz:.4f} Hz",
                f"  At position limit: {rudder_analysis.fraction_at_limit * 100:.1f}%",
                f"  At rate limit:     {rudder_analysis.fraction_rate_limited * 100:.1f}%",
            ]
        )

    return "\n".join(lines)
