"""Tests for the SignalK autopilot adapter."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from sailsim.autopilot.signalk import _SENSOR_PATHS, SignalKAutopilot
from sailsim.core.types import ControlCommand, SensorData


@pytest.fixture()
def sensors() -> SensorData:
    """Sample sensor data for testing."""
    return SensorData(
        heading=1.0,
        speed_through_water=3.5,
        speed_over_ground=3.8,
        course_over_ground=1.1,
        yaw_rate=0.02,
        roll=0.05,
        apparent_wind_speed=8.0,
        apparent_wind_angle=-0.5,
    )


class TestBuildDelta:
    """Tests for delta message construction."""

    def test_build_delta_structure(self, sensors: SensorData):
        """Delta message has correct SignalK structure."""
        delta = SignalKAutopilot._build_delta(sensors)
        assert "updates" in delta
        assert len(delta["updates"]) == 1
        values = delta["updates"][0]["values"]
        assert len(values) == len(_SENSOR_PATHS)

    def test_build_delta_paths(self, sensors: SensorData):
        """Delta contains all expected SignalK paths."""
        delta = SignalKAutopilot._build_delta(sensors)
        paths = {v["path"] for v in delta["updates"][0]["values"]}
        expected = {sk_path for _, sk_path in _SENSOR_PATHS}
        assert paths == expected

    def test_build_delta_values(self, sensors: SensorData):
        """Delta values match sensor data."""
        delta = SignalKAutopilot._build_delta(sensors)
        values_by_path = {v["path"]: v["value"] for v in delta["updates"][0]["values"]}
        assert values_by_path["navigation.headingMagnetic"] == 1.0
        assert values_by_path["navigation.speedThroughWater"] == 3.5
        assert values_by_path["environment.wind.speedApparent"] == 8.0
        assert values_by_path["environment.wind.angleApparent"] == -0.5


class TestCompute:
    """Tests for compute() with mocked HTTP."""

    def _mock_urlopen(self, rudder_value: float = 0.1):
        """Create a mock urlopen that returns a rudder value on GET."""

        def urlopen_side_effect(req, timeout=None):
            if req.method == "GET":
                body = json.dumps({"value": rudder_value}).encode()
                resp = MagicMock()
                resp.read.return_value = body
                resp.__enter__ = lambda s: s
                resp.__exit__ = MagicMock(return_value=False)
                return resp
            else:
                # POST — just return success
                resp = MagicMock()
                resp.__enter__ = lambda s: s
                resp.__exit__ = MagicMock(return_value=False)
                return resp

        return urlopen_side_effect

    @patch("sailsim.autopilot.signalk.urllib.request.urlopen")
    def test_compute_returns_rudder(self, mock_urlopen, sensors: SensorData):
        """compute() reads rudder angle from SignalK server."""
        mock_urlopen.side_effect = self._mock_urlopen(rudder_value=0.15)
        ap = SignalKAutopilot(url="http://test:3000")
        cmd = ap.compute(sensors, dt=0.05)
        assert isinstance(cmd, ControlCommand)
        assert cmd.rudder_angle == pytest.approx(0.15)

    @patch("sailsim.autopilot.signalk.urllib.request.urlopen")
    def test_compute_publishes_sensors(self, mock_urlopen, sensors: SensorData):
        """compute() sends sensor data via POST."""
        mock_urlopen.side_effect = self._mock_urlopen()
        ap = SignalKAutopilot(url="http://test:3000")
        ap.compute(sensors, dt=0.05)

        # First call should be POST (sensor publish)
        post_call = mock_urlopen.call_args_list[0]
        req = post_call[0][0]
        assert req.method == "POST"

    @patch("sailsim.autopilot.signalk.urllib.request.urlopen")
    def test_compute_graceful_on_timeout_after_connected(self, mock_urlopen, sensors: SensorData):
        """After initial connection, timeout uses last known rudder value."""
        import urllib.error

        # First call succeeds
        mock_urlopen.side_effect = self._mock_urlopen(rudder_value=0.2)
        ap = SignalKAutopilot(url="http://test:3000")
        cmd1 = ap.compute(sensors, dt=0.05)
        assert cmd1.rudder_angle == pytest.approx(0.2)

        # Second call times out
        mock_urlopen.side_effect = urllib.error.URLError("timeout")
        cmd2 = ap.compute(sensors, dt=0.05)
        assert cmd2.rudder_angle == pytest.approx(0.2)  # last known value

    @patch("sailsim.autopilot.signalk.urllib.request.urlopen")
    def test_compute_raises_on_first_failure(self, mock_urlopen, sensors: SensorData):
        """First compute() failure raises ConnectionError."""
        import urllib.error

        mock_urlopen.side_effect = urllib.error.URLError("connection refused")
        ap = SignalKAutopilot(url="http://test:3000")
        with pytest.raises(ConnectionError, match="Cannot reach SignalK"):
            ap.compute(sensors, dt=0.05)


class TestSetTargetHeading:
    """Tests for set_target_heading()."""

    @patch("sailsim.autopilot.signalk.urllib.request.urlopen")
    def test_set_target_heading(self, mock_urlopen):
        """set_target_heading stores value and publishes to SignalK."""
        mock_urlopen.return_value.__enter__ = lambda s: s
        mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)

        ap = SignalKAutopilot(url="http://test:3000")
        ap.set_target_heading(1.5)
        assert ap._target_heading == 1.5
        # Should have called urlopen for the PUT
        assert mock_urlopen.called

    def test_set_target_heading_tolerates_offline(self):
        """set_target_heading does not raise if server is unreachable."""
        ap = SignalKAutopilot(url="http://unreachable:9999", timeout=0.01)
        # Should not raise
        ap.set_target_heading(1.0)
        assert ap._target_heading == 1.0
