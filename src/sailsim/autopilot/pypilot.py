"""pypilot autopilot adapter.

Implements :class:`AutopilotProtocol` by communicating with a pypilot instance:
- Sensor data pushed via NMEA-0183 sentences to pypilot's NMEA port (default 20220)
- Autopilot commands sent via JSON-TCP protocol (default port 23322)
- Rudder commands read from ``servo.command`` via JSON-TCP

pypilot's sensor values (imu.heading, gps.speed, wind.*) are read-only via
JSON-TCP; the only way to inject sensor data is through NMEA sentences.

Uses stdlib only: socket, select, math, logging.
"""

from __future__ import annotations

import contextlib
import logging
import math
import select
import socket
import time

from sailsim.core.types import ControlCommand, SensorData

logger = logging.getLogger(__name__)


def _nmea_checksum(sentence: str) -> str:
    """Compute NMEA-0183 XOR checksum (without leading '$')."""
    cs = 0
    for c in sentence:
        cs ^= ord(c)
    return f"{cs:02X}"


def _build_nmea(body: str) -> bytes:
    """Build a complete NMEA sentence with checksum and CRLF."""
    cs = _nmea_checksum(body)
    return f"${body}*{cs}\r\n".encode("ascii")


# Sensor mapping for NMEA sentence generation
# Each entry: (SensorData field, unit-conversion, NMEA builder function name)
# Conversions: radians -> degrees where needed, m/s -> knots for speed


def _ms_to_knots(v: float) -> float:
    return v * 1.94384


class PypilotAutopilot:
    """Autopilot adapter that communicates with pypilot.

    Sensor data is pushed via NMEA-0183 sentences to pypilot's NMEA TCP port.
    Autopilot control (enable, mode, heading command) and servo.command readback
    use pypilot's JSON-TCP protocol.

    Each simulation step:
    1. Push sensor data as NMEA sentences (heading, wind, speed)
    2. Read ``servo.command`` (-1..1) from JSON-TCP and map to rudder angle
    """

    def __init__(
        self,
        host: str = "localhost",
        json_port: int = 23322,
        nmea_port: int = 20220,
        rudder_max: float = math.radians(30.0),
        rudder_rate_max: float = math.radians(10.0),
        mode: str = "compass",
        timeout: float = 0.5,
        sim_sleep_ms: float = 0.0,
    ) -> None:
        self.host = host
        self.json_port = json_port
        self.nmea_port = nmea_port
        self.rudder_max = rudder_max
        self.rudder_rate_max = rudder_rate_max
        self.mode = mode
        self.timeout = timeout
        self._sim_sleep_s = sim_sleep_ms / 1000.0

        self._target_heading = 0.0
        self._rudder_angle = 0.0
        self._last_servo_command = 0.0
        self._connected = False
        self._json_sock: socket.socket | None = None
        self._nmea_sock: socket.socket | None = None
        self._buf = ""

    # -- public API (AutopilotProtocol) --

    def set_target_heading(self, heading: float) -> None:
        """Set desired heading [rad] and publish to pypilot."""
        self._target_heading = heading
        try:
            self._ensure_connected()
            self._json_send(f"ap.heading_command={math.degrees(heading):.4f}\n")
        except OSError:
            logger.debug("Could not publish target heading to pypilot")

    def compute(self, sensors: SensorData, dt: float) -> ControlCommand:
        """Push sensors via NMEA, read servo command, map to rudder angle."""
        try:
            self._ensure_connected()
            self._push_sensors_nmea(sensors)
            if self._sim_sleep_s > 0:
                time.sleep(self._sim_sleep_s)
            servo_cmd = self._read_servo_command()
            self._last_servo_command = servo_cmd
            self._connected = True
        except OSError as exc:
            if not self._connected:
                raise ConnectionError(
                    f"Cannot reach pypilot at {self.host}:{self.json_port}: {exc}"
                ) from exc
            logger.warning(
                "pypilot unreachable, using last servo command %.3f",
                self._last_servo_command,
            )
            servo_cmd = self._last_servo_command
            # Reset sockets so _ensure_connected will reconnect next call
            self._close_sockets()

        # Map servo command directly to rudder angle (position mode).
        # Negate: pypilot positive = port, simulator positive rudder = starboard.
        self._rudder_angle = -servo_cmd * self.rudder_max

        return ControlCommand(rudder_angle=self._rudder_angle, sail_trim=0.5)

    # -- connection management --

    def _ensure_connected(self) -> None:
        """Lazy-connect to both pypilot JSON-TCP and NMEA ports."""
        if self._json_sock is not None:
            return

        # JSON-TCP connection (control + servo readback)
        json_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        json_sock.settimeout(self.timeout)
        json_sock.connect((self.host, self.json_port))
        self._json_sock = json_sock
        self._buf = ""

        # NMEA connection (sensor data injection)
        nmea_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        nmea_sock.settimeout(self.timeout)
        nmea_sock.connect((self.host, self.nmea_port))
        self._nmea_sock = nmea_sock

        # Enable autopilot and set mode via JSON-TCP
        self._json_send("ap.enabled=true\n")
        self._json_send(f'ap.mode="{self.mode}"\n')
        self._json_send(f"ap.heading_command={math.degrees(self._target_heading):.4f}\n")
        # Watch servo command for rudder feedback
        self._json_send('watch={"servo.command":true}\n')

    def _json_send(self, msg: str) -> None:
        """Send a message over the JSON-TCP socket."""
        if self._json_sock is None:
            raise OSError("Not connected to pypilot")
        self._json_sock.sendall(msg.encode("ascii"))

    # -- NMEA sensor push --

    def _push_sensors_nmea(self, sensors: SensorData) -> None:
        """Push sensor values to pypilot via NMEA-0183 sentences."""
        if self._nmea_sock is None:
            raise OSError("NMEA socket not connected")

        sentences: list[bytes] = []

        # Heading: HDM (magnetic heading)
        heading_deg = math.degrees(sensors.heading) % 360.0
        sentences.append(_build_nmea(f"HCHDM,{heading_deg:.1f},M"))

        # Wind: MWV (apparent wind, relative)
        awa_deg = math.degrees(sensors.apparent_wind_angle) % 360.0
        aws_knots = _ms_to_knots(sensors.apparent_wind_speed)
        sentences.append(_build_nmea(f"WIMWV,{awa_deg:.1f},R,{aws_knots:.1f},N,A"))

        # Speed/course: RMC (GPS data)
        sog_knots = _ms_to_knots(sensors.speed_over_ground)
        cog_deg = math.degrees(sensors.course_over_ground) % 360.0
        sentences.append(
            _build_nmea(
                f"GPRMC,120000,A,0000.0,N,00000.0,E,{sog_knots:.1f},{cog_deg:.1f},010126,,,A"
            )
        )

        # Speed through water: VHW
        stw_knots = _ms_to_knots(sensors.speed_through_water)
        sentences.append(_build_nmea(f"VWVHW,,,{heading_deg:.1f},M,{stw_knots:.1f},N,,K"))

        self._nmea_sock.sendall(b"".join(sentences))

    # -- JSON-TCP servo readback --

    def _read_servo_command(self) -> float:
        """Read latest servo.command from pypilot (non-blocking drain).

        Returns the most recent value, or the last known value if no new
        data is available within the timeout.
        """
        if self._json_sock is None:
            raise OSError("Not connected to pypilot")

        servo_cmd = self._last_servo_command

        # Drain all available data (non-blocking)
        ready, _, _ = select.select([self._json_sock], [], [], 0.0)
        if ready:
            try:
                data = self._json_sock.recv(4096)
            except OSError:
                return servo_cmd
            if not data:
                raise OSError("pypilot connection closed")
            self._buf += data.decode("ascii", errors="replace")

            # Parse complete lines
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                line = line.strip()
                if line.startswith("servo.command="):
                    with contextlib.suppress(ValueError):
                        servo_cmd = float(line.split("=", 1)[1])

        return servo_cmd

    def _close_sockets(self) -> None:
        """Close both TCP connections and reset for reconnection."""
        for sock in (self._json_sock, self._nmea_sock):
            if sock is not None:
                with contextlib.suppress(OSError):
                    sock.close()
        self._json_sock = None
        self._nmea_sock = None
        self._buf = ""

    def close(self) -> None:
        """Close both TCP connections."""
        self._close_sockets()
