"""Tests for rudder steering effort (torque + cumulative energy)."""

from __future__ import annotations

import numpy as np

from sailsim.core.types import (
    ControlCommand,
    ForceData,
    SensorData,
    VesselState,
    WindState,
)
from sailsim.recording.analysis import SteeringEffortResult, evaluate_steering_effort
from sailsim.recording.recorder import Recorder


def _make_recorder_with_torque(
    rudder_forces_y: list[float],
    rudder_angles: list[float],
    cp_offset: float = 0.08,
    dt: float = 0.1,
) -> Recorder:
    """Create a recorder with known rudder torque values."""
    rec = Recorder()
    for i, (fy, angle) in enumerate(zip(rudder_forces_y, rudder_angles, strict=True)):
        t = i * dt
        state = VesselState(
            eta=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            nu=np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )
        sensors = SensorData(heading=0.0, speed_through_water=2.0)
        control = ControlCommand(rudder_angle=angle)
        wind = WindState(speed=5.0, direction=1.0)
        forces = ForceData(
            sail=np.array([100.0, -50.0, -20.0]),
            rudder=np.array([-5.0, fy, -80.0]),
            keel=np.array([-10.0, 100.0, -30.0]),
        )
        torque = float(-fy * cp_offset)
        rec.record(t, state, sensors, control, wind, forces=forces, rudder_torque=torque)
    return rec


def test_torque_sign():
    """Positive rudder lateral force → negative torque (Newton III)."""
    rec = _make_recorder_with_torque(
        rudder_forces_y=[100.0],
        rudder_angles=[0.1],
        cp_offset=0.08,
    )
    assert rec.steps[0].rudder_torque is not None
    assert rec.steps[0].rudder_torque < 0  # -100 * 0.08 = -8.0


def test_torque_proportional_to_cp_offset():
    """Torque scales linearly with cp_offset."""
    rec1 = _make_recorder_with_torque(
        rudder_forces_y=[100.0],
        rudder_angles=[0.1],
        cp_offset=0.08,
    )
    rec2 = _make_recorder_with_torque(
        rudder_forces_y=[100.0],
        rudder_angles=[0.1],
        cp_offset=0.16,
    )
    t1 = rec1.steps[0].rudder_torque
    t2 = rec2.steps[0].rudder_torque
    assert t1 is not None and t2 is not None
    assert abs(t2 / t1 - 2.0) < 1e-10


def test_evaluate_steering_effort_known_values():
    """evaluate_steering_effort with known torques."""
    rec = _make_recorder_with_torque(
        rudder_forces_y=[0.0, -100.0, -200.0, -100.0],
        rudder_angles=[0.0, 0.1, 0.2, 0.1],
        cp_offset=0.1,
    )
    # Torques: [0, 10, 20, 10]  (negated * 0.1)
    result = evaluate_steering_effort(rec)
    assert isinstance(result, SteeringEffortResult)
    assert abs(result.peak_torque_nm - 20.0) < 1e-10
    assert abs(result.mean_torque_nm - 10.0) < 1e-10


def test_evaluate_steering_effort_no_torque_data():
    """evaluate_steering_effort with no torque data returns zeros."""
    rec = Recorder()
    for i in range(3):
        t = i * 0.1
        state = VesselState(
            eta=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            nu=np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )
        sensors = SensorData(heading=0.0, speed_through_water=2.0)
        control = ControlCommand()
        wind = WindState(speed=5.0, direction=1.0)
        rec.record(t, state, sensors, control, wind)

    result = evaluate_steering_effort(rec)
    assert result.peak_torque_nm == 0.0
    assert result.mean_torque_nm == 0.0
    assert result.total_energy_j == 0.0
    assert result.energy_rate_j_per_s == 0.0


def test_cumulative_energy_hand_calculation():
    """Cumulative energy matches hand calculation.

    Steps:  t=0.0  angle=0.0   T=0
            t=0.1  angle=0.1   T=10   E += |10| * |0.1-0.0| = 1.0
            t=0.2  angle=0.2   T=20   E += |20| * |0.2-0.1| = 2.0
            t=0.3  angle=0.1   T=10   E += |10| * |0.1-0.2| = 1.0
    Total energy = 4.0
    Duration = 0.3
    Rate = 4.0 / 0.3 ≈ 13.333
    """
    rec = _make_recorder_with_torque(
        rudder_forces_y=[0.0, -100.0, -200.0, -100.0],
        rudder_angles=[0.0, 0.1, 0.2, 0.1],
        cp_offset=0.1,
    )
    result = evaluate_steering_effort(rec)
    assert abs(result.total_energy_j - 4.0) < 1e-10
    assert abs(result.energy_rate_j_per_s - 4.0 / 0.3) < 1e-6
