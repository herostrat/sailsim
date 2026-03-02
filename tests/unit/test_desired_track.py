"""Tests for desired track (Soll-Linie) computation."""

from __future__ import annotations

import numpy as np

from sailsim.core.types import ControlCommand, SensorData, VesselState, WindState
from sailsim.recording.recorder import Recorder


def test_desired_track_constant_heading():
    """Constant heading + constant speed -> straight line north."""
    rec = Recorder()
    heading = 0.0  # due North
    speed = 2.0
    for i in range(11):
        t = i * 1.0
        state = VesselState(
            eta=np.array([t * speed, 0.0, 0.0, 0.0, 0.0, heading]),
            nu=np.array([speed, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )
        sensors = SensorData(heading=heading, speed_through_water=speed)
        rec.record(
            t,
            state,
            sensors,
            ControlCommand(),
            WindState(speed=5.0, direction=1.0),
            target_heading=heading,
        )

    xs, ys = rec.desired_track()
    assert len(xs) == 11
    assert abs(xs[-1] - 20.0) < 0.01  # 10 steps * 1s * 2m/s * cos(0)
    assert abs(ys[-1]) < 0.01  # no east displacement


def test_desired_track_heading_change():
    """90-degree heading change produces an L-shaped track."""
    rec = Recorder()
    speed = 1.0
    for i in range(20):
        t = i * 1.0
        heading = 0.0 if t < 10.0 else np.pi / 2
        state = VesselState(nu=np.array([speed, 0.0, 0.0, 0.0, 0.0, 0.0]))
        sensors = SensorData(heading=heading, speed_through_water=speed)
        rec.record(
            t,
            state,
            sensors,
            ControlCommand(),
            WindState(speed=5.0, direction=1.0),
            target_heading=heading,
        )

    xs, ys = rec.desired_track()
    # First 10s: moves North at 1 m/s
    assert xs[10] > 9.0
    assert abs(ys[10]) < 0.01
    # After heading change: moves East
    assert ys[-1] > 8.0


def test_desired_track_empty_recorder():
    """Empty recorder returns empty lists."""
    rec = Recorder()
    xs, ys = rec.desired_track()
    assert xs == []
    assert ys == []


def test_desired_track_no_target_heading_fallback():
    """When target_heading is None, falls back to actual heading."""
    rec = Recorder()
    for i in range(5):
        t = i * 1.0
        state = VesselState(nu=np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        sensors = SensorData(heading=0.5, speed_through_water=2.0)
        rec.record(t, state, sensors, ControlCommand(), WindState(speed=5.0, direction=1.0))

    xs, ys = rec.desired_track()
    assert len(xs) == 5
    # cos(0.5) > 0, sin(0.5) > 0
    assert xs[-1] > 0
    assert ys[-1] > 0
