"""Integration tests for recorder data completeness."""

from __future__ import annotations

from sailsim.autopilot.pid import PIDAutopilot
from sailsim.core.config import (
    CurrentConfig,
    ScenarioConfig,
    WaveConfig,
    WindConfig,
)
from sailsim.core.runner import run_scenario


def test_runner_records_waves_and_target():
    """Runner should record waves and target_heading in each step."""
    config = ScenarioConfig(
        duration_s=1.0,
        dt=0.1,
        waves=WaveConfig(model="spectral", Hs=1.0, Tp=6.0, seed=42),
        current=CurrentConfig(model="constant", speed=0.3, direction=0.5),
        wind=WindConfig(speed=5.0, direction=1.0),
    )
    autopilot = PIDAutopilot()
    recorder = run_scenario(config, autopilot)

    for step in recorder.steps:
        assert step.waves is not None, "WaveState should be recorded"
        assert step.waves.Hs == 1.0
        assert step.target_heading is not None, "target_heading should be recorded"
        assert step.target_heading == config.autopilot.target_heading
        assert step.current is not None, "CurrentState should be recorded"
