"""Test PID autopilot behavior."""

import numpy as np
import pytest

from sailsim.autopilot.pid import PIDAutopilot
from sailsim.core.types import SensorData


def test_zero_error_zero_output():
    """When on target heading with no yaw rate, rudder should be ~0."""
    pid = PIDAutopilot(kp=1.0, ki=0.0, kd=1.0)
    pid.set_target_heading(0.0)
    sensors = SensorData(heading=0.0, yaw_rate=0.0)
    cmd = pid.compute(sensors, 0.05)
    assert cmd.rudder_angle == pytest.approx(0.0)


def test_proportional_response():
    """Heading error should produce proportional rudder output."""
    pid = PIDAutopilot(kp=1.0, ki=0.0, kd=0.0)
    pid.set_target_heading(0.0)
    sensors = SensorData(heading=0.1, yaw_rate=0.0)
    cmd = pid.compute(sensors, 0.05)
    # Error = 0 - 0.1 = -0.1, rudder = kp * error = -0.1
    assert cmd.rudder_angle == pytest.approx(-0.1, abs=0.01)


def test_derivative_opposes_yaw_rate():
    """Positive yaw rate should produce counter-rudder."""
    pid = PIDAutopilot(kp=0.0, ki=0.0, kd=1.0)
    pid.set_target_heading(0.0)
    sensors = SensorData(heading=0.0, yaw_rate=0.5)
    cmd = pid.compute(sensors, 0.05)
    # Derivative = -yaw_rate, rudder = kd * (-0.5) = -0.5
    assert cmd.rudder_angle < 0


def test_rudder_saturation():
    """Rudder should be clamped to physical limits."""
    pid = PIDAutopilot(kp=10.0, ki=0.0, kd=0.0, rudder_max=0.52)
    pid.set_target_heading(0.0)
    sensors = SensorData(heading=1.0, yaw_rate=0.0)
    cmd = pid.compute(sensors, 0.05)
    assert abs(cmd.rudder_angle) <= 0.52


def test_heading_wrapping():
    """Error should wrap correctly across 0/360 boundary."""
    pid = PIDAutopilot(kp=1.0, ki=0.0, kd=0.0)
    pid.set_target_heading(np.radians(350))
    # Heading at 10° -> shortest path is -20° (go left)
    sensors = SensorData(heading=np.radians(10), yaw_rate=0.0)
    cmd = pid.compute(sensors, 0.05)
    # Error should be -20° = -0.349 rad, rudder should be negative
    assert cmd.rudder_angle < 0
    assert cmd.rudder_angle == pytest.approx(-np.radians(20), abs=0.01)
