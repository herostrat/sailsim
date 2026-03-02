"""Command-line interface for the sailing yacht simulator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from sailsim.autopilot.pid import PIDAutopilot
from sailsim.core.config import ScenarioConfig, load_scenario
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
        help="Path to scenario TOML file",
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
    if args.scenario:
        config = load_scenario(args.scenario)
    else:
        config = ScenarioConfig(
            name="default_heading_hold",
            duration_s=120.0,
            dt=0.05,
        )

    if not args.quiet:
        print(f"Scenario: {config.name}")
        print(f"Duration: {config.duration_s}s, dt: {config.dt}s")
        print(f"Wind: {config.wind.speed:.1f} m/s @ {np.degrees(config.wind.direction):.0f}°")
        print(f"Target heading: {np.degrees(config.autopilot.target_heading):.0f}°")
        print()

    # Create autopilot
    autopilot = PIDAutopilot(
        kp=config.autopilot.kp,
        ki=config.autopilot.ki,
        kd=config.autopilot.kd,
        auto_sail_trim=config.autopilot.auto_sail_trim,
    )

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
    target_heading_deg = np.degrees(config.autopilot.target_heading)
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
