#!/usr/bin/env python3
"""Control-theoretic stability analysis for autopilots.

Modes:
  analytical  — Model-based analysis from yacht config (no simulation needed)
  empirical   — Data-driven analysis from recorded JSON files
  full        — Run simulation, then analyse both analytically and empirically

Examples:
  python scripts/analyze_autopilot.py analytical --yacht default --speed 3.0
  python scripts/analyze_autopilot.py analytical --yacht default --sweep
  python scripts/analyze_autopilot.py empirical recording.json
  python scripts/analyze_autopilot.py empirical nomoto.json pypilot.json
  python scripts/analyze_autopilot.py full --scenario calm_heading_hold
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def cmd_analytical(args: argparse.Namespace) -> None:
    """Model-based linear stability analysis."""
    from sailsim.analysis.linear import analyze_at_speed, sweep_speed
    from sailsim.analysis.report import summarize_linear, summarize_speed_sweep
    from sailsim.core.config import load_autopilot, load_yacht

    yacht = load_yacht(args.yacht)
    ap = load_autopilot(args.autopilot)
    omega_n = args.omega_n or ap.omega_n
    zeta = args.zeta or ap.zeta

    if args.sweep:
        U_range = np.linspace(0.5, 8.0, 16)
        sweep = sweep_speed(yacht, U_range, omega_n, zeta)
        print(summarize_speed_sweep(sweep))

        if not args.no_plot:
            import matplotlib.pyplot as plt

            from sailsim.analysis.plots import plot_speed_sensitivity

            plot_speed_sensitivity(sweep)
            if args.save_dir:
                save = Path(args.save_dir)
                save.mkdir(parents=True, exist_ok=True)
                plt.savefig(save / "speed_sensitivity.png", dpi=150)
                print(f"\nSaved to {save / 'speed_sensitivity.png'}")
            else:
                plt.show()
    else:
        result = analyze_at_speed(yacht, args.speed, omega_n, zeta)
        print(summarize_linear(result))

        if not args.no_plot:
            import matplotlib.pyplot as plt

            from sailsim.analysis.plots import (
                plot_bode,
                plot_nyquist,
                plot_pole_zero_map,
            )

            plot_bode(result, which="open_loop")
            plot_nyquist(result)
            plot_pole_zero_map(result)
            if args.save_dir:
                save = Path(args.save_dir)
                save.mkdir(parents=True, exist_ok=True)
                plt.figure(1)
                plt.savefig(save / "bode.png", dpi=150)
                plt.figure(2)
                plt.savefig(save / "nyquist.png", dpi=150)
                plt.figure(3)
                plt.savefig(save / "pole_zero.png", dpi=150)
                print(f"\nPlots saved to {save}")
            else:
                plt.show()


def cmd_empirical(args: argparse.Namespace) -> None:
    """Data-driven analysis from recorded JSON files."""
    from sailsim.analysis.empirical import (
        analyze_rudder_activity,
        estimate_transfer_function,
        extract_step_responses,
    )
    from sailsim.analysis.report import summarize_empirical
    from sailsim.recording.recorder import Recorder

    for json_path in args.files:
        path = Path(json_path)
        if not path.exists():
            print(f"File not found: {path}", file=sys.stderr)
            continue

        print(f"\n{'=' * 60}")
        print(f"Analysing: {path.name}")
        print("=" * 60)

        rec = Recorder.from_json(path)

        steps = extract_step_responses(rec)
        rudder = analyze_rudder_activity(
            rec,
            rate_limit_deg_s=args.rate_limit,
        )
        spectral = estimate_transfer_function(rec)

        print(summarize_empirical(steps, rudder))

        if not args.no_plot:
            import matplotlib.pyplot as plt

            from sailsim.analysis.plots import (
                plot_estimated_bode,
                plot_rudder_spectrum,
                plot_saturation_summary,
                plot_step_response,
            )

            plot_step_response(steps, recorder=rec)
            plot_estimated_bode(spectral)
            plot_rudder_spectrum(rudder)
            plot_saturation_summary(rudder)

            if args.save_dir:
                save = Path(args.save_dir)
                save.mkdir(parents=True, exist_ok=True)
                stem = path.stem
                for i, name in enumerate(
                    ["step_response", "bode_empirical", "rudder_psd", "saturation"],
                    start=1,
                ):
                    plt.figure(i)
                    plt.savefig(save / f"{stem}_{name}.png", dpi=150)
                print(f"Plots saved to {save}")
            else:
                plt.show()


def cmd_full(args: argparse.Namespace) -> None:
    """Run simulation, then analyse analytically and empirically."""
    import tempfile

    from sailsim.analysis.empirical import (
        analyze_rudder_activity,
        extract_step_responses,
    )
    from sailsim.analysis.linear import analyze_at_speed
    from sailsim.analysis.report import summarize_empirical, summarize_linear
    from sailsim.autopilot.factory import create_autopilot
    from sailsim.core.config import load_autopilot, load_scenario
    from sailsim.core.runner import run_scenario

    config = load_scenario(args.scenario)
    ap_config = load_autopilot(getattr(args, "autopilot", "heading_hold"))
    autopilot = create_autopilot(ap_config, config.yacht)
    rec = run_scenario(config, autopilot)

    # Save recording
    if args.save_dir:
        save = Path(args.save_dir)
        save.mkdir(parents=True, exist_ok=True)
        json_path = save / f"{args.scenario}_recording.json"
    else:
        json_path = Path(tempfile.mktemp(suffix=".json"))
    rec.to_json(json_path)
    print(f"Recording saved to {json_path}")

    # Analytical
    U = 3.0  # nominal speed
    if rec.steps:
        speeds = [s.sensors.speed_through_water for s in rec.steps]
        U = float(np.mean(speeds)) if speeds else 3.0
    result = analyze_at_speed(
        config.yacht,
        U,
        ap_config.omega_n,
        ap_config.zeta,
    )
    print("\n" + summarize_linear(result))

    # Empirical
    steps = extract_step_responses(rec)
    rudder = analyze_rudder_activity(rec)
    print("\n" + summarize_empirical(steps, rudder))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autopilot stability analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- analytical ---
    p_ana = sub.add_parser("analytical", help="Model-based analysis")
    p_ana.add_argument("--yacht", default="default", help="Yacht config name/path")
    p_ana.add_argument("--autopilot", default="heading_hold", help="Autopilot config name/path")
    p_ana.add_argument("--speed", type=float, default=3.0, help="Speed [m/s]")
    p_ana.add_argument("--omega-n", type=float, default=None, help="Natural frequency override")
    p_ana.add_argument("--zeta", type=float, default=None, help="Damping ratio override")
    p_ana.add_argument("--sweep", action="store_true", help="Speed sweep instead of single point")
    p_ana.add_argument("--save-dir", help="Directory to save plots")
    p_ana.add_argument("--no-plot", action="store_true", help="Skip plots")
    p_ana.set_defaults(func=cmd_analytical)

    # --- empirical ---
    p_emp = sub.add_parser("empirical", help="Data-driven analysis")
    p_emp.add_argument("files", nargs="+", help="JSON recording files")
    p_emp.add_argument("--rate-limit", type=float, default=5.0, help="Rudder rate limit [deg/s]")
    p_emp.add_argument("--save-dir", help="Directory to save plots")
    p_emp.add_argument("--no-plot", action="store_true", help="Skip plots")
    p_emp.set_defaults(func=cmd_empirical)

    # --- full ---
    p_full = sub.add_parser("full", help="Simulate + analyse")
    p_full.add_argument("--scenario", default="calm_heading_hold", help="Scenario name/path")
    p_full.add_argument("--autopilot", default="heading_hold", help="Autopilot config name/path")
    p_full.add_argument("--save-dir", help="Directory to save output")
    p_full.set_defaults(func=cmd_full)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
