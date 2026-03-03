"""Command-line interface for the sailing yacht simulator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from sailsim.autopilot.factory import create_autopilot
from sailsim.core.config import load_autopilot, load_scenario, load_yacht
from sailsim.core.runner import run_scenario
from sailsim.recording.analysis import evaluate_heading_hold
from sailsim.recording.recorder import Recorder


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sailsim",
        description="Sailing yacht autopilot simulator and test framework",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="calm_heading_hold",
        help="Scenario profile name or TOML path (default: 'calm_heading_hold')",
    )
    parser.add_argument(
        "--yacht",
        type=str,
        default="default",
        help="Yacht profile name or TOML path (default: 'default')",
    )
    parser.add_argument(
        "--autopilot",
        type=str,
        default="heading_hold",
        help="Autopilot profile name or TOML path (default: 'heading_hold')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output CSV file (optional)",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default=None,
        help="Save recording as JSON file",
    )
    parser.add_argument(
        "--view",
        nargs="*",
        metavar="JSON_FILE",
        help="Launch viewer. No args: run and view. With args: load and compare JSON files.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # View-only mode: load JSON files and display
    if args.view is not None and len(args.view) > 0:
        from sailsim.viewer import show

        runs = []
        for json_path in args.view:
            rec = Recorder.from_json(json_path)
            label = Path(json_path).stem
            runs.append((label, rec))
        show(runs)
        return

    # Load scenario
    config = load_scenario(args.scenario)

    # Load yacht and autopilot separately
    config.yacht = load_yacht(args.yacht)
    ap_config = load_autopilot(args.autopilot)
    autopilot = create_autopilot(ap_config, yacht=config.yacht)

    if not args.quiet:
        print(f"Scenario: {config.name}")
        print(f"  Yacht: {args.yacht}")
        if ap_config.type == "signalk":
            ap_info = f"signalk @ {ap_config.signalk_url}"
        else:
            ap_info = f"nomoto (omega_n={ap_config.omega_n}, zeta={ap_config.zeta})"
        print(f"  Autopilot: {ap_info} [{args.autopilot}]")
        print(f"  Duration: {config.duration_s}s, dt: {config.dt}s")
        print(f"  Wind: {config.wind.speed:.1f} m/s @ {np.degrees(config.wind.direction):.0f}°")
        print(f"  Target heading: {np.degrees(config.target_heading):.0f}°")
        print()

    # Run simulation
    if not args.quiet:
        print("Running simulation...")

    recorder = run_scenario(config, autopilot)

    if not args.quiet:
        print(f"Completed: {len(recorder.steps)} steps recorded")
        print()

    # Export CSV if requested
    if args.output:
        recorder.to_csv(args.output)
        if not args.quiet:
            print(f"Telemetry saved to: {args.output}")
            print()

    # Save JSON if requested
    if args.save_json:
        metadata = {"scenario_name": config.name, "dt": config.dt, "duration_s": config.duration_s}
        recorder.to_json(args.save_json, metadata=metadata)
        if not args.quiet:
            print(f"Recording saved to: {args.save_json}")
            print()

    # Evaluate quality gates
    target_heading_deg = np.degrees(config.target_heading)
    result = evaluate_heading_hold(
        recorder,
        target_heading_deg,
        config.quality_gates,
    )

    print(result.summary())

    # Launch viewer if --view with no args (run + view)
    if args.view is not None and len(args.view) == 0:
        from sailsim.viewer import show

        show([(config.name, recorder)])

    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
