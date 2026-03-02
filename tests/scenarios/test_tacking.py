"""Scenario test: tacking (port to starboard)."""

import numpy as np

from sailsim.autopilot.pid import PIDAutopilot
from sailsim.core.config import load_scenario
from sailsim.core.runner import run_scenario
from sailsim.recording.analysis import evaluate_maneuver


def test_tack_port_to_starboard():
    """Boat should complete a tack from port to starboard."""
    config = load_scenario("configs/scenarios/tack_port_to_starboard.toml")
    autopilot = PIDAutopilot(
        kp=config.autopilot.kp,
        ki=config.autopilot.ki,
        kd=config.autopilot.kd,
        auto_sail_trim=config.autopilot.auto_sail_trim,
    )
    recorder = run_scenario(config, autopilot)

    # Evaluate the tack maneuver at t=30s
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
        f"Tack not completed within window. Completion time: {result.completion_time_s:.1f}s"
    )
    assert result.completion_time_s < 30.0, f"Tack too slow: {result.completion_time_s:.1f}s"

    # Verify heading crossed through the wind (0°)
    headings = [np.degrees(s.sensors.heading) for s in recorder.steps]
    maneuver_start_idx = next(i for i, s in enumerate(recorder.steps) if s.t >= step.time_s)
    headings_during = headings[maneuver_start_idx:]
    near_zero = any(abs(h) < 15 for h in headings_during)
    assert near_zero, "Heading should cross near head-to-wind during tack"
