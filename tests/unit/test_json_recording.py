"""Tests for JSON recording serialization."""

import tempfile
from pathlib import Path

import numpy as np

from sailsim.core.types import (
    ControlCommand,
    ForceData,
    SensorData,
    VesselState,
    WaveState,
    WindState,
)
from sailsim.recording.recorder import Recorder


def _make_recorder_without_forces() -> Recorder:
    """Create a recorder with a few steps, no force data."""
    rec = Recorder()
    for i in range(5):
        t = i * 0.1
        state = VesselState(
            eta=np.array([t, 0.0, 0.0, 0.0, 0.0, 0.1 * i]),
            nu=np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.01 * i]),
        )
        sensors = SensorData(heading=0.1 * i, speed_through_water=2.0)
        control = ControlCommand(rudder_angle=0.05 * i)
        wind = WindState(speed=5.0, direction=1.0)
        rec.record(t, state, sensors, control, wind)
    return rec


def _make_recorder_with_forces() -> Recorder:
    """Create a recorder with a few steps, including force data."""
    rec = Recorder()
    for i in range(5):
        t = i * 0.1
        state = VesselState(
            eta=np.array([t, 0.0, 0.0, 0.0, 0.0, 0.1 * i]),
            nu=np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.01 * i]),
        )
        sensors = SensorData(heading=0.1 * i, speed_through_water=2.0)
        control = ControlCommand(rudder_angle=0.05 * i)
        wind = WindState(speed=5.0, direction=1.0)
        forces = ForceData(
            sail=np.array([100.0 + i, -200.0, -50.0]),
            rudder=np.array([-5.0, 30.0 * i, -100.0 * i]),
            keel=np.array([-10.0, 180.0, -40.0]),
        )
        rec.record(t, state, sensors, control, wind, forces=forces)
    return rec


def test_roundtrip_without_forces():
    """JSON save/load roundtrip without force data."""
    rec = _make_recorder_without_forces()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"
        rec.to_json(path)
        loaded = Recorder.from_json(path)

    assert len(loaded.steps) == len(rec.steps)
    for orig, load in zip(rec.steps, loaded.steps, strict=False):
        assert abs(orig.t - load.t) < 1e-10
        assert np.allclose(orig.state.eta, load.state.eta)
        assert np.allclose(orig.state.nu, load.state.nu)
        assert abs(orig.sensors.heading - load.sensors.heading) < 1e-10
        assert abs(orig.control.rudder_angle - load.control.rudder_angle) < 1e-10
        assert abs(orig.wind.speed - load.wind.speed) < 1e-10
        assert load.forces is None


def test_roundtrip_with_forces():
    """JSON save/load roundtrip with force data."""
    rec = _make_recorder_with_forces()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"
        rec.to_json(path)
        loaded = Recorder.from_json(path)

    assert len(loaded.steps) == len(rec.steps)
    for orig, load in zip(rec.steps, loaded.steps, strict=False):
        assert load.forces is not None
        assert np.allclose(orig.forces.sail, load.forces.sail)
        assert np.allclose(orig.forces.rudder, load.forces.rudder)
        assert np.allclose(orig.forces.keel, load.forces.keel)


def test_json_version_field():
    """JSON output should contain version field."""
    import json

    rec = _make_recorder_without_forces()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"
        rec.to_json(path)
        with open(path) as f:
            doc = json.load(f)

    assert doc["version"] == 2
    assert "steps" in doc
    assert "metadata" in doc


def test_json_metadata():
    """Metadata should be preserved in JSON."""
    import json

    rec = _make_recorder_without_forces()
    metadata = {"scenario_name": "test", "dt": 0.05}

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"
        rec.to_json(path, metadata=metadata)
        with open(path) as f:
            doc = json.load(f)

    assert doc["metadata"]["scenario_name"] == "test"
    assert doc["metadata"]["dt"] == 0.05


def test_roundtrip_with_waves_and_target():
    """JSON roundtrip preserves waves and target_heading."""
    rec = Recorder()
    for i in range(3):
        t = i * 0.1
        state = VesselState(
            eta=np.array([t, 0.0, 0.0, 0.0, 0.0, 0.0]),
            nu=np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )
        sensors = SensorData(heading=0.0, speed_through_water=2.0)
        control = ControlCommand()
        wind = WindState(speed=5.0, direction=1.0)
        waves = WaveState(Hs=1.5, Tp=6.0, direction=0.5, elevation=0.3 * i)
        rec.record(t, state, sensors, control, wind, waves=waves, target_heading=0.7)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"
        rec.to_json(path)
        loaded = Recorder.from_json(path)

    for orig, load in zip(rec.steps, loaded.steps, strict=False):
        assert load.waves is not None
        assert abs(load.waves.Hs - orig.waves.Hs) < 1e-10
        assert abs(load.waves.elevation - orig.waves.elevation) < 1e-10
        assert load.target_heading is not None
        assert abs(load.target_heading - 0.7) < 1e-10


def test_backward_compat_no_waves_or_target():
    """Loading JSON without waves/target_heading fields should give None."""
    rec = _make_recorder_without_forces()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"
        rec.to_json(path)
        loaded = Recorder.from_json(path)

    for step in loaded.steps:
        assert step.waves is None
        assert step.target_heading is None


def test_roundtrip_with_rudder_torque():
    """JSON roundtrip preserves rudder_torque."""
    rec = Recorder()
    for i in range(3):
        t = i * 0.1
        state = VesselState(
            eta=np.array([t, 0.0, 0.0, 0.0, 0.0, 0.0]),
            nu=np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )
        sensors = SensorData(heading=0.0, speed_through_water=2.0)
        control = ControlCommand()
        wind = WindState(speed=5.0, direction=1.0)
        forces = ForceData(
            sail=np.array([100.0, -50.0, -20.0]),
            rudder=np.array([-5.0, 30.0, -80.0]),
            keel=np.array([-10.0, 100.0, -30.0]),
        )
        rec.record(t, state, sensors, control, wind, forces=forces, rudder_torque=-2.4 * i)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"
        rec.to_json(path)
        loaded = Recorder.from_json(path)

    for orig, load in zip(rec.steps, loaded.steps, strict=False):
        assert load.rudder_torque is not None or orig.rudder_torque == 0.0
        assert abs((load.rudder_torque or 0.0) - (orig.rudder_torque or 0.0)) < 1e-10


def test_backward_compat_no_rudder_torque():
    """Loading JSON without rudder_torque should give None."""
    rec = _make_recorder_without_forces()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"
        rec.to_json(path)
        loaded = Recorder.from_json(path)

    for step in loaded.steps:
        assert step.rudder_torque is None
