"""Scenario-based test: heading hold with PID autopilot."""

import numpy as np

from sailsim.autopilot.pid import PIDAutopilot
from sailsim.core.config import load_scenario
from sailsim.core.runner import run_scenario
from sailsim.recording.analysis import evaluate_heading_hold


def test_calm_heading_hold_scenario():
    """The calm heading hold scenario should PASS all quality gates."""
    config = load_scenario("configs/scenarios/calm_heading_hold.toml")
    autopilot = PIDAutopilot(
        kp=config.autopilot.kp,
        ki=config.autopilot.ki,
        kd=config.autopilot.kd,
    )
    recorder = run_scenario(config, autopilot)

    target_deg = np.degrees(config.autopilot.target_heading)
    result = evaluate_heading_hold(recorder, target_deg, config.quality_gates)

    assert result.passed, f"Scenario failed:\n{result.summary()}"
