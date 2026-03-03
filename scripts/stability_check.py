#!/usr/bin/env python3
"""Quick stability check for all yacht profiles in 3-DOF and 6-DOF modes.

For each yacht profile, runs a 30-second simulation with moderate wind
(5 m/s TWS, 60 deg TWD) and reports whether the simulation diverges (NaN)
and the final speed, roll, and heading.
"""

from __future__ import annotations

import sys
import tomllib
import warnings
from pathlib import Path

import numpy as np

# Suppress overflow warnings during the run (we detect divergence explicitly)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Ensure package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sailsim.autopilot.factory import create_autopilot  # noqa: E402
from sailsim.core.config import AutopilotConfig, ScenarioConfig, YachtConfig  # noqa: E402
from sailsim.core.runner import run_scenario  # noqa: E402

YACHT_DIR = Path(__file__).resolve().parents[1] / "configs" / "yachts"
PROFILES = ["mini650", "j24", "dehler34", "swan45", "hr62", "imoca60"]

# Moderate wind conditions: 5 m/s from 60 deg (close reach)
WIND_SPEED = 5.0
WIND_DIR = np.radians(60.0)
DURATION = 30.0
DT = 0.05
INITIAL_U = 1.5  # small initial surge so boat is moving


def load_yacht_profile(name: str) -> dict:
    """Load a yacht TOML profile and return as dict."""
    path = YACHT_DIR / f"{name}.toml"
    with path.open("rb") as f:
        return tomllib.load(f)


def build_config(profile_name: str, dof: int) -> ScenarioConfig:
    """Build a ScenarioConfig for the given yacht profile and DOF mode."""
    yacht_data = load_yacht_profile(profile_name)
    yacht_data["dof"] = dof

    return ScenarioConfig(
        name=f"stability_{profile_name}_{dof}dof",
        duration_s=DURATION,
        dt=DT,
        target_heading=0.0,
        yacht=YachtConfig(**yacht_data),
        wind={
            "speed": WIND_SPEED,
            "direction": WIND_DIR,
            "model": "constant",
        },
        initial_state={"u": INITIAL_U, "psi": 0.0},
    )


def check_diverged(recorder) -> tuple[bool, float | None]:
    """Check if any state variable diverged to NaN or Inf.

    Returns (diverged, diverge_time) — time of first NaN/Inf, or None.
    """
    for step in recorder.steps:
        if np.any(np.isnan(step.state.eta)) or np.any(np.isinf(step.state.eta)):
            return True, step.t
        if np.any(np.isnan(step.state.nu)) or np.any(np.isinf(step.state.nu)):
            return True, step.t
    return False, None


def extract_max_roll(recorder) -> float:
    """Get the maximum absolute roll angle during the simulation."""
    return max(abs(step.state.phi) for step in recorder.steps)


def extract_max_speed(recorder) -> float:
    """Get the maximum speed during the simulation."""
    return max(
        float(np.sqrt(step.state.nu[0] ** 2 + step.state.nu[1] ** 2))
        for step in recorder.steps
        if not np.any(np.isnan(step.state.nu))
    )


def time_series_snapshot(recorder, times_s: list[float]) -> list[dict]:
    """Extract state snapshots at the given approximate times."""
    snapshots = []
    step_idx = 0
    for target_t in times_s:
        # Find closest step
        while step_idx < len(recorder.steps) - 1:
            if recorder.steps[step_idx].t >= target_t:
                break
            step_idx += 1
        step = recorder.steps[step_idx]
        s = step.state
        if np.any(np.isnan(s.eta)):
            snapshots.append({
                "t": step.t,
                "speed_kn": float("nan"),
                "heading_deg": float("nan"),
                "roll_deg": float("nan"),
                "pitch_deg": float("nan"),
                "rudder_deg": float("nan"),
            })
        else:
            speed_ms = float(np.sqrt(s.nu[0] ** 2 + s.nu[1] ** 2))
            snapshots.append({
                "t": step.t,
                "speed_kn": speed_ms * 1.94384,
                "heading_deg": float(np.degrees(s.eta[5])) % 360,
                "roll_deg": float(np.degrees(s.eta[3])),
                "pitch_deg": float(np.degrees(s.eta[4])),
                "rudder_deg": float(np.degrees(step.control.rudder_angle)),
            })
    return snapshots


def run_check(profile_name: str, dof: int) -> dict:
    """Run a single stability check and return results."""
    config = build_config(profile_name, dof)
    ap_config = AutopilotConfig(kp=1.0, ki=0.05, kd=2.0, auto_sail_trim=True)
    autopilot = create_autopilot(ap_config)
    try:
        recorder = run_scenario(config, autopilot)
    except Exception as e:
        return {
            "profile": profile_name,
            "dof": dof,
            "status": f"ERROR: {e}",
            "speed_kn": None,
            "roll_deg": None,
            "max_roll_deg": None,
            "max_speed_kn": None,
            "heading_deg": None,
            "heading_error_deg": None,
            "pitch_deg": None,
            "diverge_time": None,
            "snapshots": [],
        }

    diverged, diverge_time = check_diverged(recorder)
    final = recorder.steps[-1].state

    speed_ms = float(np.sqrt(final.nu[0] ** 2 + final.nu[1] ** 2))
    speed_kn = speed_ms * 1.94384
    roll_deg = np.degrees(final.phi)
    pitch_deg = np.degrees(final.eta[4])
    heading_deg = np.degrees(final.eta[5]) % 360
    heading_error = np.degrees(
        np.arctan2(
            np.sin(final.eta[5] - 0.0),
            np.cos(final.eta[5] - 0.0),
        )
    )
    max_roll = np.degrees(extract_max_roll(recorder))
    max_speed = extract_max_speed(recorder) * 1.94384

    # Time series at 5s intervals
    snap_times = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    snapshots = time_series_snapshot(recorder, snap_times)

    # Classify status
    if diverged:
        status = "DIVERGED"
    elif max_roll > 45:
        status = "CAPSIZE"
    elif max_speed > 15:
        status = "UNSTABLE"
    elif abs(heading_error) > 30:
        status = "DRIFTING"
    else:
        status = "OK"

    return {
        "profile": profile_name,
        "dof": dof,
        "status": status,
        "speed_kn": speed_kn,
        "roll_deg": roll_deg,
        "max_roll_deg": max_roll,
        "max_speed_kn": max_speed,
        "heading_deg": heading_deg,
        "heading_error_deg": heading_error,
        "pitch_deg": pitch_deg,
        "diverge_time": diverge_time,
        "snapshots": snapshots,
    }


def print_summary_table(results: list[dict], dof: int):
    """Print the summary table for a DOF mode."""
    print(f"\n{'=' * 100}")
    print(f"  {dof}-DOF MODE SUMMARY")
    print(f"{'=' * 100}")
    print(
        f"{'Profile':<12} {'Status':<10} {'Speed[kn]':>10} {'MaxSpd[kn]':>11} "
        f"{'Roll[deg]':>10} {'MaxRoll':>8} {'Pitch[deg]':>11} "
        f"{'HdgErr[deg]':>12}"
    )
    print("-" * 100)

    for r in results:
        if r["speed_kn"] is not None and not np.isnan(r["speed_kn"]):
            print(
                f"{r['profile']:<12} {r['status']:<10} "
                f"{r['speed_kn']:>10.2f} {r['max_speed_kn']:>11.2f} "
                f"{r['roll_deg']:>10.2f} {r['max_roll_deg']:>8.1f} "
                f"{r['pitch_deg']:>11.2f} {r['heading_error_deg']:>12.2f}"
            )
        elif r.get("diverge_time") is not None:
            dt_str = f"NaN @ {r['diverge_time']:.1f}s"
            print(
                f"{r['profile']:<12} {r['status']:<10} "
                f"{'--- NaN ---':>10} {'':>11} "
                f"{'':>10} {r['max_roll_deg']:>8.1f} "
                f"{'':>11} {dt_str:>12}"
            )
        else:
            print(f"{r['profile']:<12} {r['status']}")


def print_time_series(results: list[dict], dof: int):
    """Print time-series snapshots for each profile."""
    print(f"\n{'=' * 100}")
    print(f"  {dof}-DOF TIME SERIES (snapshots every 5s)")
    print(f"{'=' * 100}")

    for r in results:
        if not r["snapshots"]:
            continue
        print(f"\n  --- {r['profile']} ({r['status']}) ---")
        print(f"  {'t[s]':>6} {'Speed[kn]':>10} {'Heading':>10} {'Roll':>10} {'Pitch':>10} {'Rudder':>10}")
        for snap in r["snapshots"]:
            if np.isnan(snap["speed_kn"]):
                print(f"  {snap['t']:>6.1f}      NaN        NaN        NaN        NaN        NaN")
            else:
                print(
                    f"  {snap['t']:>6.1f} {snap['speed_kn']:>10.2f} "
                    f"{snap['heading_deg']:>10.1f} {snap['roll_deg']:>10.2f} "
                    f"{snap['pitch_deg']:>10.2f} {snap['rudder_deg']:>10.1f}"
                )


def main():
    print("=" * 100)
    print("SAILING AUTOPILOT SIMULATOR — YACHT PROFILE STABILITY CHECK")
    print(f"Wind: {WIND_SPEED} m/s TWS, {np.degrees(WIND_DIR):.0f} deg TWD (constant)")
    print(f"Duration: {DURATION}s, dt={DT}s, initial speed: {INITIAL_U} m/s")
    print("Autopilot: PID (kp=1.0, ki=0.05, kd=2.0), target heading = 0 deg, auto_sail_trim = True")
    print("=" * 100)

    all_results = {}

    for dof in [3, 6]:
        results = []
        for profile in PROFILES:
            result = run_check(profile, dof)
            results.append(result)
        all_results[dof] = results

    # Print summary tables
    for dof in [3, 6]:
        print_summary_table(all_results[dof], dof)

    # Print time series
    for dof in [3, 6]:
        print_time_series(all_results[dof], dof)

    # Final assessment
    print(f"\n{'=' * 100}")
    print("ASSESSMENT")
    print("=" * 100)
    print()
    print("Status codes:")
    print("  OK       = Stable simulation, realistic values")
    print("  DRIFTING = Heading error > 30 deg (autopilot not holding course)")
    print("  UNSTABLE = Max speed > 15 kn (unrealistic for 5 m/s wind)")
    print("  CAPSIZE  = Max roll > 45 deg (boat tipping over)")
    print("  DIVERGED = NaN/Inf in state (numerical blow-up)")
    print("  ERROR    = Exception during simulation")
    print()

    for dof in [3, 6]:
        ok_count = sum(1 for r in all_results[dof] if r["status"] == "OK")
        total = len(all_results[dof])
        print(f"  {dof}-DOF: {ok_count}/{total} profiles stable")
        for r in all_results[dof]:
            if r["status"] != "OK":
                detail = ""
                if r["diverge_time"] is not None:
                    detail = f" (NaN at t={r['diverge_time']:.1f}s)"
                elif r["max_roll_deg"] is not None:
                    detail = f" (max roll={r['max_roll_deg']:.1f} deg)"
                print(f"    {r['profile']:12} -> {r['status']}{detail}")
    print()


if __name__ == "__main__":
    main()
