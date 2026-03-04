"""Tests for the pypilot autopilot adapter."""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest

from sailsim.autopilot.pypilot import PypilotAutopilot, _build_nmea, _nmea_checksum
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


class TestNmeaHelpers:
    """Tests for NMEA sentence construction."""

    def test_nmea_checksum(self):
        """Checksum is correct XOR of all characters."""
        # Known sentence: HCHDM,180.0,M
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


class TestSensorNmea:
    """Tests for sensor data -> NMEA conversion."""

    @patch("sailsim.autopilot.pypilot.socket.socket")
    @patch("sailsim.autopilot.pypilot.select.select")
    def test_push_sensors_sends_hdm(self, mock_select, mock_socket_cls, sensors):
        """Heading is sent as NMEA HDM sentence."""
        json_sock = _make_mock_socket()
        nmea_sock = _make_mock_socket()
        mock_socket_cls.side_effect = [json_sock, nmea_sock]
        mock_select.return_value = ([], [], [])

        ap = PypilotAutopilot(host="test", json_port=23322, nmea_port=20220)
        ap.compute(sensors, dt=0.05)

        # NMEA socket should have sent data
        assert nmea_sock.sendall.called
        sent = nmea_sock.sendall.call_args[0][0].decode("ascii")
        assert "HDM,180.0,M" in sent

    @patch("sailsim.autopilot.pypilot.socket.socket")
    @patch("sailsim.autopilot.pypilot.select.select")
    def test_push_sensors_sends_mwv(self, mock_select, mock_socket_cls, sensors):
        """Wind data is sent as NMEA MWV sentence."""
        json_sock = _make_mock_socket()
        nmea_sock = _make_mock_socket()
        mock_socket_cls.side_effect = [json_sock, nmea_sock]
        mock_select.return_value = ([], [], [])

        ap = PypilotAutopilot(host="test", json_port=23322, nmea_port=20220)
        ap.compute(sensors, dt=0.05)

        sent = nmea_sock.sendall.call_args[0][0].decode("ascii")
        assert "MWV," in sent
        assert ",R," in sent  # Relative wind
        assert ",N,A" in sent  # Knots, valid

    @patch("sailsim.autopilot.pypilot.socket.socket")
    @patch("sailsim.autopilot.pypilot.select.select")
    def test_push_sensors_sends_rmc(self, mock_select, mock_socket_cls, sensors):
        """GPS data is sent as NMEA RMC sentence."""
        json_sock = _make_mock_socket()
        nmea_sock = _make_mock_socket()
        mock_socket_cls.side_effect = [json_sock, nmea_sock]
        mock_select.return_value = ([], [], [])

        ap = PypilotAutopilot(host="test", json_port=23322, nmea_port=20220)
        ap.compute(sensors, dt=0.05)

        sent = nmea_sock.sendall.call_args[0][0].decode("ascii")
        assert "RMC," in sent

    @patch("sailsim.autopilot.pypilot.socket.socket")
    @patch("sailsim.autopilot.pypilot.select.select")
    def test_push_sensors_converts_speed_to_knots(
        self, mock_select, mock_socket_cls, sensors
    ):
        """Speed values are converted from m/s to knots in NMEA."""
        json_sock = _make_mock_socket()
        nmea_sock = _make_mock_socket()
        mock_socket_cls.side_effect = [json_sock, nmea_sock]
        mock_select.return_value = ([], [], [])

        ap = PypilotAutopilot(host="test", json_port=23322, nmea_port=20220)
        ap.compute(sensors, dt=0.05)

        sent = nmea_sock.sendall.call_args[0][0].decode("ascii")
        # SOG 3.8 m/s ≈ 7.4 knots
        sog_knots = 3.8 * 1.94384
        assert f"{sog_knots:.1f}" in sent


class TestCompute:
    """Tests for compute() with mocked sockets."""

    @patch("sailsim.autopilot.pypilot.socket.socket")
    @patch("sailsim.autopilot.pypilot.select.select")
    def test_compute_returns_control_command(self, mock_select, mock_socket_cls, sensors):
        """compute() returns a ControlCommand."""
        json_sock = _make_mock_socket(b"servo.command=0.5\n")
        nmea_sock = _make_mock_socket()
        mock_socket_cls.side_effect = [json_sock, nmea_sock]
        mock_select.return_value = ([json_sock], [], [])

        ap = PypilotAutopilot(host="test", json_port=23322, nmea_port=20220)
        cmd = ap.compute(sensors, dt=0.05)
        assert isinstance(cmd, ControlCommand)

    @patch("sailsim.autopilot.pypilot.socket.socket")
    @patch("sailsim.autopilot.pypilot.select.select")
    def test_compute_maps_servo_to_rudder_position(
        self, mock_select, mock_socket_cls, sensors
    ):
        """servo.command is mapped (negated) to rudder position: angle = -cmd * rudder_max."""
        json_sock = _make_mock_socket(b"servo.command=0.5\n")
        nmea_sock = _make_mock_socket()
        mock_socket_cls.side_effect = [json_sock, nmea_sock]
        mock_select.return_value = ([json_sock], [], [])

        rudder_max = math.radians(30.0)
        ap = PypilotAutopilot(
            host="test", json_port=23322, nmea_port=20220,
            rudder_max=rudder_max,
        )

        cmd = ap.compute(sensors, dt=0.05)
        # Negated: pypilot positive = port, sim positive = starboard
        expected_angle = -0.5 * rudder_max
        assert cmd.rudder_angle == pytest.approx(expected_angle)

    @patch("sailsim.autopilot.pypilot.socket.socket")
    @patch("sailsim.autopilot.pypilot.select.select")
    def test_compute_full_servo_gives_full_rudder(self, mock_select, mock_socket_cls, sensors):
        """servo.command=1.0 maps to -rudder_max (full port)."""
        json_sock = _make_mock_socket(b"servo.command=1.0\n")
        nmea_sock = _make_mock_socket()
        mock_socket_cls.side_effect = [json_sock, nmea_sock]
        mock_select.return_value = ([json_sock], [], [])

        rudder_max = math.radians(30.0)
        ap = PypilotAutopilot(
            host="test", json_port=23322, nmea_port=20220,
            rudder_max=rudder_max,
        )

        cmd = ap.compute(sensors, dt=1.0)
        assert cmd.rudder_angle == pytest.approx(-rudder_max)

    @patch("sailsim.autopilot.pypilot.socket.socket")
    @patch("sailsim.autopilot.pypilot.select.select")
    def test_compute_updates_rudder_each_step(
        self, mock_select, mock_socket_cls, sensors
    ):
        """Rudder angle updates (not accumulates) each compute() call."""
        json_sock = _make_mock_socket(b"servo.command=0.5\n")
        nmea_sock = _make_mock_socket()
        mock_socket_cls.side_effect = [json_sock, nmea_sock]
        mock_select.return_value = ([json_sock], [], [])

        rudder_max = math.radians(30.0)
        ap = PypilotAutopilot(
            host="test", json_port=23322, nmea_port=20220,
            rudder_max=rudder_max,
        )

        ap.compute(sensors, dt=0.05)

        # Second call with same servo.command — rudder should be same, not accumulated
        json_sock.recv.return_value = b"servo.command=0.5\n"
        cmd2 = ap.compute(sensors, dt=0.05)

        expected = -0.5 * rudder_max
        assert cmd2.rudder_angle == pytest.approx(expected)

    @patch("sailsim.autopilot.pypilot.socket.socket")
    @patch("sailsim.autopilot.pypilot.select.select")
    def test_compute_graceful_after_connected(self, mock_select, mock_socket_cls, sensors):
        """After initial connection, socket error uses last known servo command."""
        json_sock = _make_mock_socket(b"servo.command=0.3\n")
        nmea_sock = _make_mock_socket()
        mock_socket_cls.side_effect = [json_sock, nmea_sock]
        mock_select.return_value = ([json_sock], [], [])

        ap = PypilotAutopilot(host="test", json_port=23322, nmea_port=20220)
        ap.compute(sensors, dt=0.05)

        # Now the NMEA socket fails
        nmea_sock.sendall.side_effect = OSError("connection lost")
        cmd2 = ap.compute(sensors, dt=0.05)

        # Should not raise — uses last servo command (0.3)
        assert isinstance(cmd2, ControlCommand)

    @patch("sailsim.autopilot.pypilot.socket.socket")
    def test_compute_raises_on_first_failure(self, mock_socket_cls, sensors):
        """First compute() failure raises ConnectionError."""
        mock_sock = MagicMock()
        mock_sock.connect.side_effect = OSError("connection refused")
        mock_sock.settimeout = MagicMock()
        mock_socket_cls.return_value = mock_sock

        ap = PypilotAutopilot(host="test", json_port=23322, nmea_port=20220)
        with pytest.raises(ConnectionError, match="Cannot reach pypilot"):
            ap.compute(sensors, dt=0.05)

    @patch("sailsim.autopilot.pypilot.socket.socket")
    @patch("sailsim.autopilot.pypilot.select.select")
    def test_compute_no_data_uses_last_command(self, mock_select, mock_socket_cls, sensors):
        """When no new servo data arrives, last known value is used."""
        json_sock = _make_mock_socket()
        nmea_sock = _make_mock_socket()
        mock_socket_cls.side_effect = [json_sock, nmea_sock]
        # No data ready (select returns empty)
        mock_select.return_value = ([], [], [])

        ap = PypilotAutopilot(host="test", json_port=23322, nmea_port=20220)
        cmd = ap.compute(sensors, dt=0.05)
        # Default last_servo_command is 0.0
        assert cmd.rudder_angle == pytest.approx(0.0)


class TestSetTargetHeading:
    """Tests for set_target_heading()."""

    @patch("sailsim.autopilot.pypilot.socket.socket")
    def test_set_target_heading_sends_command(self, mock_socket_cls):
        """set_target_heading sends ap.heading_command over JSON-TCP."""
        json_sock = _make_mock_socket()
        nmea_sock = _make_mock_socket()
        mock_socket_cls.side_effect = [json_sock, nmea_sock]

        ap = PypilotAutopilot(host="test", json_port=23322, nmea_port=20220)
        ap.set_target_heading(math.radians(45.0))

        assert ap._target_heading == pytest.approx(math.radians(45.0))
        # Should have connected and sent heading via JSON-TCP
        sent = b"".join(call[0][0] for call in json_sock.sendall.call_args_list)
        sent_str = sent.decode("ascii")
        assert "ap.heading_command=45.0" in sent_str

    def test_set_target_heading_tolerates_offline(self):
        """set_target_heading does not raise if pypilot is unreachable."""
        ap = PypilotAutopilot(host="unreachable", json_port=99999, timeout=0.01)
        # Should not raise
        ap.set_target_heading(math.radians(90.0))
        assert ap._target_heading == pytest.approx(math.radians(90.0))


class TestProtocol:
    """Tests for pypilot protocol format."""

    @patch("sailsim.autopilot.pypilot.socket.socket")
    @patch("sailsim.autopilot.pypilot.select.select")
    def test_watch_message_format(self, mock_select, mock_socket_cls):
        """Connection sends watch message for servo.command via JSON-TCP."""
        json_sock = _make_mock_socket()
        nmea_sock = _make_mock_socket()
        mock_socket_cls.side_effect = [json_sock, nmea_sock]
        mock_select.return_value = ([], [], [])

        ap = PypilotAutopilot(host="test", json_port=23322, nmea_port=20220)
        ap.compute(SensorData(), dt=0.05)

        sent = b"".join(call[0][0] for call in json_sock.sendall.call_args_list)
        sent_str = sent.decode("ascii")
        assert 'watch={"servo.command":true}' in sent_str

    @patch("sailsim.autopilot.pypilot.socket.socket")
    @patch("sailsim.autopilot.pypilot.select.select")
    def test_enables_autopilot_on_connect(self, mock_select, mock_socket_cls):
        """Connection enables autopilot and sets mode via JSON-TCP."""
        json_sock = _make_mock_socket()
        nmea_sock = _make_mock_socket()
        mock_socket_cls.side_effect = [json_sock, nmea_sock]
        mock_select.return_value = ([], [], [])

        ap = PypilotAutopilot(host="test", json_port=23322, nmea_port=20220, mode="compass")
        ap.compute(SensorData(), dt=0.05)

        sent = b"".join(call[0][0] for call in json_sock.sendall.call_args_list)
        sent_str = sent.decode("ascii")
        assert "ap.enabled=true" in sent_str
        assert 'ap.mode="compass"' in sent_str

    @patch("sailsim.autopilot.pypilot.socket.socket")
    @patch("sailsim.autopilot.pypilot.select.select")
    def test_connects_to_both_ports(self, mock_select, mock_socket_cls):
        """Connection opens both JSON-TCP and NMEA sockets."""
        json_sock = _make_mock_socket()
        nmea_sock = _make_mock_socket()
        mock_socket_cls.side_effect = [json_sock, nmea_sock]
        mock_select.return_value = ([], [], [])

        ap = PypilotAutopilot(host="test", json_port=23322, nmea_port=20220)
        ap.compute(SensorData(), dt=0.05)

        # Two sockets created
        assert mock_socket_cls.call_count == 2
        # JSON-TCP connects to json_port
        json_sock.connect.assert_called_once_with(("test", 23322))
        # NMEA connects to nmea_port
        nmea_sock.connect.assert_called_once_with(("test", 20220))
