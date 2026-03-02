"""Tests for maneuver analysis."""

from __future__ import annotations

import numpy as np
import pytest

from sailsim.core.types import ControlCommand, SensorData, VesselState, WindState
from sailsim.recording.analysis import evaluate_maneuver
from sailsim.recording.recorder import Recorder, TimeStep


def _make_recorder_with_heading_profile(
    times: list[float],
    headings_deg: list[float],
    speed: float = 2.0,
) -> Recorder:
    """Create a recorder with synthetic heading data."""
    rec = Recorder()
    for t, hdg_deg in zip(times, headings_deg, strict=False):
        hdg_rad = np.radians(hdg_deg)
        eta = np.zeros(6)
        eta[5] = hdg_rad
        nu = np.zeros(6)
        nu[0] = speed
        rec.steps.append(
            TimeStep(
                t=t,
                state=VesselState(eta=eta, nu=nu),
                sensors=SensorData(heading=hdg_rad, speed_through_water=speed),
                control=ControlCommand(),
                wind=WindState(speed=5.0, direction=0.0),
            )
        )
    return rec


class TestEvaluateManeuver:
    def test_instant_completion(self):
        """If already at target, completion time should be 0."""
        times = [0, 1, 2, 3, 4, 5]
        headings = [0, 0, 45, 45, 45, 45]
        rec = _make_recorder_with_heading_profile(times, headings)
        result = evaluate_maneuver(rec, maneuver_time_s=2.0, target_heading_deg=45.0)
        assert result.completed
        assert result.completion_time_s == pytest.approx(0.0)

    def test_gradual_turn(self):
        """Heading ramps to target — should detect completion time."""
        times = list(np.arange(0, 20, 0.5))
        headings = []
        for t in times:
            if t < 5:
                headings.append(0.0)
            elif t < 15:
                headings.append(min(4.5 * (t - 5), 45.0))
            else:
                headings.append(45.0)

        rec = _make_recorder_with_heading_profile(times, headings)
        result = evaluate_maneuver(rec, maneuver_time_s=5.0, target_heading_deg=45.0)
        assert result.completed
        assert result.completion_time_s > 0
        assert result.completion_time_s < 15.0

    def test_overshoot_detected(self):
        """Overshoot past target should be measured."""
        times = list(np.arange(0, 20, 0.5))
        headings = []
        for t in times:
            if t < 5:
                headings.append(0.0)
            elif t < 10:
                headings.append(9.0 * (t - 5))  # ramp to 45 + overshoot
            elif t < 15:
                headings.append(55.0 - 2.0 * (t - 10))  # settle back
            else:
                headings.append(45.0)

        rec = _make_recorder_with_heading_profile(times, headings)
        result = evaluate_maneuver(rec, maneuver_time_s=5.0, target_heading_deg=45.0)
        assert result.completed
        assert result.overshoot_deg > 5.0

    def test_speed_loss_measured(self):
        """Speed drop during maneuver should be captured."""
        times = list(np.arange(0, 10, 0.5))
        headings = [45.0] * len(times)
        speeds = [2.0 if t < 5 else 1.0 for t in times]

        rec = Recorder()
        for t, hdg_deg, spd in zip(times, headings, speeds, strict=False):
            hdg_rad = np.radians(hdg_deg)
            eta = np.zeros(6)
            eta[5] = hdg_rad
            nu = np.zeros(6)
            nu[0] = spd
            rec.steps.append(
                TimeStep(
                    t=t,
                    state=VesselState(eta=eta, nu=nu),
                    sensors=SensorData(heading=hdg_rad, speed_through_water=spd),
                    control=ControlCommand(),
                    wind=WindState(speed=5.0, direction=0.0),
                )
            )

        result = evaluate_maneuver(rec, maneuver_time_s=0.0, target_heading_deg=45.0)
        assert result.speed_loss_pct == pytest.approx(50.0)

    def test_not_completed(self):
        """If heading never reaches target, completed=False."""
        times = list(np.arange(0, 10, 0.5))
        headings = [10.0] * len(times)  # never reaches 45
        rec = _make_recorder_with_heading_profile(times, headings)
        result = evaluate_maneuver(rec, maneuver_time_s=0.0, target_heading_deg=45.0)
        assert not result.completed
