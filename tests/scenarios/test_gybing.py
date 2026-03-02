"""Scenario test: gybing (starboard to port)."""

import numpy as np

from sailsim.autopilot.pid import PIDAutopilot
from sailsim.core.config import load_scenario
from sailsim.core.runner import run_scenario
from sailsim.recording.analysis import evaluate_maneuver


def test_gybe_starboard_to_port():
    """Boat should complete a gybe from starboard to port."""
    config = load_scenario("configs/scenarios/gybe_starboard_to_port.toml")
    autopilot = PIDAutopilot(
        kp=config.autopilot.kp,
        ki=config.autopilot.ki,
        kd=config.autopilot.kd,
        auto_sail_trim=config.autopilot.auto_sail_trim,
    )
    recorder = run_scenario(config, autopilot)

    # Evaluate the gybe maneuver at t=30s
    step = config.maneuvers.steps[0]
    target_deg = np.degrees(step.target_heading)
    result = evaluate_maneuver(
        recorder,
        maneuver_time_s=step.time_s,
        target_heading_deg=target_deg,
        threshold_deg=10.0,
        window_s=60.0,
    )

    assert result.completed, (
        f"Gybe not completed within window. Completion time: {result.completion_time_s:.1f}s"
    )
    assert result.completion_time_s < 45.0, f"Gybe too slow: {result.completion_time_s:.1f}s"
