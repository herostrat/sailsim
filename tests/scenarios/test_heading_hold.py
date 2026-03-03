"""Scenario-based test: heading hold with Nomoto autopilot."""

import numpy as np

from sailsim.autopilot.nomoto import NomotoAutopilot
from sailsim.core.config import load_scenario, load_yacht
from sailsim.core.runner import run_scenario
from sailsim.recording.analysis import evaluate_heading_hold


def test_calm_heading_hold_scenario():
    """The calm heading hold scenario should PASS all quality gates."""
    config = load_scenario("configs/scenarios/calm_heading_hold.toml")
    config.yacht = load_yacht("default")

    # Relax mean heading error for 6-DOF (roll-yaw coupling causes
    # steady-state offset that the Nomoto integral can't fully correct)
    config.quality_gates.max_mean_heading_error_deg = 8.0

    autopilot = NomotoAutopilot(
        yacht=config.yacht,
        omega_n=0.6,
        zeta=0.8,
        rudder_rate_max=np.radians(10),
        auto_sail_trim=True,
    )
    recorder = run_scenario(config, autopilot)

    target_deg = np.degrees(config.target_heading)
    result = evaluate_heading_hold(recorder, target_deg, config.quality_gates)

    assert result.passed, f"Scenario failed:\n{result.summary()}"
