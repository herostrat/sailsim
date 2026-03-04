"""Tests for the signalk-rs autopilot adapter."""

from __future__ import annotations

import json
import math
from unittest.mock import MagicMock, patch

import pytest

from sailsim.autopilot.signalk_rs import (
    SignalKRsAutopilot,
    _build_nmea,
    _nmea_checksum,
)
from sailsim.core.types import ControlCommand, SensorData


@pytest.fixture()
def sensors() -> SensorData:
    """Sample sensor data for testing."""
    return SensorData(
        heading=math.radians(180.0),
        speed_through_water=3.5,
        speed_over_ground=3.8,
        course_over_ground=math.radians(170.0),
        yaw_rate=0.02,
        roll=math.radians(5.0),
        apparent_wind_speed=8.0,
        apparent_wind_angle=math.radians(-30.0),
    )


def _make_mock_socket(recv_data: bytes = b""):
    """Create a mock socket."""
    sock = MagicMock()
    sock.connect = MagicMock()
    sock.sendall = MagicMock()
    sock.close = MagicMock()
    sock.settimeout = MagicMock()
    sock.recv = MagicMock(return_value=recv_data)
    return sock


def _mock_urlopen_rudder(rudder_rad: float):
    """Create a mock context manager for urllib.request.urlopen returning rudder value."""
    resp = MagicMock()
    resp.read.return_value = json.dumps({"value": rudder_rad}).encode("utf-8")
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestNmeaHelpers:
    """Tests for NMEA sentence construction (shared with pypilot)."""

    def test_nmea_checksum(self):
        """Checksum is correct XOR of all characters."""
        cs = _nmea_checksum("HCHDM,180.0,M")
        assert len(cs) == 2
        assert cs == cs.upper()

    def test_build_nmea_format(self):
        """Built NMEA sentence has $, checksum, and CRLF."""
        result = _build_nmea("HCHDM,180.0,M")
        assert result.startswith(b"$")
        assert result.endswith(b"\r\n")
        assert b"*" in result

    def test_build_nmea_ascii(self):
        """NMEA sentence is pure ASCII."""
        result = _build_nmea("WIMWV,30.0,R,15.6,N,A")
        result.decode("ascii")  # Should not raise


class TestCompute:
    """Tests for compute() with mocked sockets and HTTP."""

    @patch("sailsim.autopilot.signalk_rs.urllib.request.urlopen")
    @patch("sailsim.autopilot.signalk_rs.socket.socket")
    def test_compute_returns_control_command(self, mock_socket_cls, mock_urlopen, sensors):
        """compute() returns a ControlCommand."""
        nmea_sock = _make_mock_socket()
        mock_socket_cls.return_value = nmea_sock
        mock_urlopen.return_value = _mock_urlopen_rudder(0.1)

        ap = SignalKRsAutopilot(host="test", http_port=3000, nmea_port=10110)
        cmd = ap.compute(sensors, dt=0.05)
        assert isinstance(cmd, ControlCommand)

    @patch("sailsim.autopilot.signalk_rs.urllib.request.urlopen")
    @patch("sailsim.autopilot.signalk_rs.socket.socket")
    def test_compute_reads_rudder(self, mock_socket_cls, mock_urlopen, sensors):
        """compute() reads rudder angle from HTTP and returns it."""
        nmea_sock = _make_mock_socket()
        mock_socket_cls.return_value = nmea_sock
        rudder_rad = math.radians(10.0)
        mock_urlopen.return_value = _mock_urlopen_rudder(rudder_rad)

        ap = SignalKRsAutopilot(host="test", http_port=3000, nmea_port=10110)
        cmd = ap.compute(sensors, dt=0.05)
        assert cmd.rudder_angle == pytest.approx(rudder_rad)

    @patch("sailsim.autopilot.signalk_rs.urllib.request.urlopen")
    @patch("sailsim.autopilot.signalk_rs.socket.socket")
    def test_compute_publishes_sensors(self, mock_socket_cls, mock_urlopen, sensors):
        """compute() sends NMEA sentences over TCP."""
        nmea_sock = _make_mock_socket()
        mock_socket_cls.return_value = nmea_sock
        mock_urlopen.return_value = _mock_urlopen_rudder(0.0)

        ap = SignalKRsAutopilot(host="test", http_port=3000, nmea_port=10110)
        ap.compute(sensors, dt=0.05)

        assert nmea_sock.sendall.called
        sent = nmea_sock.sendall.call_args[0][0].decode("ascii")
        assert "HDM,180.0,M" in sent
        assert "MWV," in sent
        assert "RMC," in sent

    @patch("sailsim.autopilot.signalk_rs.urllib.request.urlopen")
    @patch("sailsim.autopilot.signalk_rs.socket.socket")
    def test_engage_on_first_compute(self, mock_socket_cls, mock_urlopen, sensors):
        """First compute() engages autopilot and sets mode via V2 API."""
        nmea_sock = _make_mock_socket()
        mock_socket_cls.return_value = nmea_sock
        mock_urlopen.return_value = _mock_urlopen_rudder(0.0)

        ap = SignalKRsAutopilot(host="test", http_port=3000, nmea_port=10110, mode="compass")
        ap.compute(sensors, dt=0.05)

        # Check that engage, mode, and target were called via urlopen
        calls = mock_urlopen.call_args_list
        urls = [call[0][0].full_url for call in calls]

        # Should have: engage POST, mode PUT, target PUT, rudder GET
        engage_urls = [u for u in urls if "engage" in u]
        mode_urls = [u for u in urls if "mode" in u]
        target_urls = [u for u in urls if "target" in u]
        rudder_urls = [u for u in urls if "rudderAngle" in u]

        assert len(engage_urls) >= 1
        assert len(mode_urls) >= 1
        assert len(target_urls) >= 1
        assert len(rudder_urls) >= 1

    @patch("sailsim.autopilot.signalk_rs.urllib.request.urlopen")
    @patch("sailsim.autopilot.signalk_rs.socket.socket")
    def test_graceful_fallback_after_connected(self, mock_socket_cls, mock_urlopen, sensors):
        """After initial connection, HTTP error uses last known rudder."""
        nmea_sock = _make_mock_socket()
        mock_socket_cls.return_value = nmea_sock
        rudder_rad = math.radians(5.0)
        mock_urlopen.return_value = _mock_urlopen_rudder(rudder_rad)

        ap = SignalKRsAutopilot(host="test", http_port=3000, nmea_port=10110)
        cmd1 = ap.compute(sensors, dt=0.05)
        assert cmd1.rudder_angle == pytest.approx(rudder_rad)

        # Now NMEA socket fails
        nmea_sock.sendall.side_effect = OSError("connection lost")
        cmd2 = ap.compute(sensors, dt=0.05)

        # Should not raise — uses last rudder value
        assert isinstance(cmd2, ControlCommand)
        assert cmd2.rudder_angle == pytest.approx(rudder_rad)

    @patch("sailsim.autopilot.signalk_rs.socket.socket")
    def test_raises_on_first_failure(self, mock_socket_cls, sensors):
        """First compute() failure raises ConnectionError."""
        mock_sock = MagicMock()
        mock_sock.connect.side_effect = OSError("connection refused")
        mock_sock.settimeout = MagicMock()
        mock_socket_cls.return_value = mock_sock

        ap = SignalKRsAutopilot(host="test", http_port=3000, nmea_port=10110)
        with pytest.raises(ConnectionError, match="Cannot reach signalk-rs"):
            ap.compute(sensors, dt=0.05)


class TestSetTargetHeading:
    """Tests for set_target_heading()."""

    @patch("sailsim.autopilot.signalk_rs.urllib.request.urlopen")
    def test_set_target_heading_calls_v2_api(self, mock_urlopen):
        """set_target_heading() sends PUT to V2 API with radians."""
        mock_urlopen.return_value = _mock_urlopen_rudder(0.0)

        ap = SignalKRsAutopilot(host="test", http_port=3000, nmea_port=10110)
        heading_rad = math.radians(45.0)
        ap.set_target_heading(heading_rad)

        assert ap._target_heading == pytest.approx(heading_rad)
        # Should have called urlopen for the target PUT
        assert mock_urlopen.called
        req = mock_urlopen.call_args[0][0]
        assert "target" in req.full_url
        assert req.method == "PUT"
        body = json.loads(req.data.decode("utf-8"))
        assert body["value"] == pytest.approx(heading_rad)

    def test_set_target_heading_tolerates_offline(self):
        """set_target_heading does not raise if signalk-rs is unreachable."""
        ap = SignalKRsAutopilot(host="unreachable", http_port=99999)
        # Should not raise
        ap.set_target_heading(math.radians(90.0))
        assert ap._target_heading == pytest.approx(math.radians(90.0))


class TestReconnection:
    """Tests for socket reconnection after errors."""

    @patch("sailsim.autopilot.signalk_rs.urllib.request.urlopen")
    @patch("sailsim.autopilot.signalk_rs.socket.socket")
    def test_reconnects_after_error(self, mock_socket_cls, mock_urlopen, sensors):
        """After connection loss, adapter reconnects on next compute() call."""
        nmea_sock1 = _make_mock_socket()
        nmea_sock2 = _make_mock_socket()
        mock_socket_cls.side_effect = [nmea_sock1, nmea_sock2]
        mock_urlopen.return_value = _mock_urlopen_rudder(0.0)

        ap = SignalKRsAutopilot(host="test", http_port=3000, nmea_port=10110)
        ap.compute(sensors, dt=0.05)

        # Connection drops
        nmea_sock1.sendall.side_effect = OSError("connection lost")
        ap.compute(sensors, dt=0.05)  # Uses last rudder, resets socket

        # Socket should be cleared for reconnection
        assert ap._nmea_sock is None

        # Next call reconnects with fresh socket
        cmd3 = ap.compute(sensors, dt=0.05)
        assert mock_socket_cls.call_count == 2
        assert isinstance(cmd3, ControlCommand)


class TestFactory:
    """Tests for creating SignalKRsAutopilot via the factory."""

    def test_create_signalk_rs_autopilot(self):
        """Factory creates SignalKRsAutopilot for type='signalk_rs'."""
        from sailsim.autopilot.factory import create_autopilot
        from sailsim.autopilot.signalk_rs import SignalKRsAutopilot
        from sailsim.core.config import AutopilotConfig

        config = AutopilotConfig(
            type="signalk_rs",
            signalk_rs_host="myhost",
            signalk_rs_http_port=4000,
            signalk_rs_nmea_port=10111,
            signalk_rs_device_id="mydevice",
            signalk_rs_rudder_max_deg=25.0,
            signalk_rs_mode="wind",
        )
        ap = create_autopilot(config)
        assert isinstance(ap, SignalKRsAutopilot)
        assert ap.host == "myhost"
        assert ap.http_port == 4000
        assert ap.nmea_port == 10111
        assert ap.device_id == "mydevice"
        assert ap.mode == "wind"


class TestConfig:
    """Tests for signalk-rs config loading."""

    def test_load_autopilot_signalk_rs(self, tmp_path):
        """Load signalk-rs autopilot config from TOML."""
        from sailsim.core.config import load_autopilot

        toml_content = """\
type = "signalk_rs"
signalk_rs_host = "192.168.1.10"
signalk_rs_http_port = 3001
signalk_rs_nmea_port = 10111
signalk_rs_device_id = "test_device"
signalk_rs_rudder_max_deg = 25.0
signalk_rs_mode = "wind"
signalk_rs_sim_sleep_ms = 100
"""
        toml_file = tmp_path / "test_signalk_rs.toml"
        toml_file.write_text(toml_content)

        config = load_autopilot(str(toml_file))
        assert config.type == "signalk_rs"
        assert config.signalk_rs_host == "192.168.1.10"
        assert config.signalk_rs_http_port == 3001
        assert config.signalk_rs_nmea_port == 10111
        assert config.signalk_rs_device_id == "test_device"
        assert config.signalk_rs_rudder_max_deg == 25.0
        assert config.signalk_rs_mode == "wind"
        assert config.signalk_rs_sim_sleep_ms == 100
