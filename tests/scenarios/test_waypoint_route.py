"""Scenario test: waypoint route following."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sailsim.autopilot.nomoto import NomotoAutopilot
from sailsim.core.config import load_scenario, load_yacht
from sailsim.core.runner import run_scenario
from sailsim.core.types import Waypoint
from sailsim.recording.analysis import evaluate_waypoint_route

SCENARIOS_DIR = Path(__file__).resolve().parents[2] / "configs" / "scenarios"


def test_waypoint_triangle():
    """Boat should reach all 3 waypoints in the triangle scenario."""
    config = load_scenario(SCENARIOS_DIR / "waypoint_triangle.toml")
    config.yacht = load_yacht("default")

    autopilot = NomotoAutopilot(
        yacht=config.yacht,
        omega_n=0.6,
        zeta=0.7,
        rudder_rate_max=np.radians(10),
        auto_sail_trim=True,
    )
    recorder = run_scenario(config, autopilot)

    waypoints = [Waypoint(x=wp.x, y=wp.y, tolerance=wp.tolerance) for wp in config.route.waypoints]
    result = evaluate_waypoint_route(recorder, waypoints)

    assert result.all_reached, (
        f"Not all waypoints reached: {result.waypoints_reached}/{result.total_waypoints}\n"
        f"{result.summary()}"
    )
