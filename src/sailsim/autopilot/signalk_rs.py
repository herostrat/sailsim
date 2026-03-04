"""signalk-rs autopilot adapter.

Implements :class:`AutopilotProtocol` by communicating with a signalk-rs instance
(Rust-based SignalK server with built-in PID autopilot).

This is distinct from the original SignalK reference implementation adapter
(``signalk.py``). signalk-rs includes its own autopilot provider plugin with
gain scheduling, anti-windup, heel compensation, and gust response.

Communication:
- Sensor data pushed via NMEA-0183 sentences to signalk-rs TCP port (default 10110)
- Autopilot lifecycle via V2 Autopilot API (engage/disengage/mode/target)
- Rudder readback via HTTP GET from V1 API

Uses stdlib only: socket, math, logging, urllib.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import socket
import time
import urllib.error
import urllib.request

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


def _ms_to_knots(v: float) -> float:
    return v * 1.94384


class SignalKRsAutopilot:
    """Autopilot adapter for signalk-rs (Rust-based SignalK server).

    Sensor data is pushed via NMEA-0183 sentences to signalk-rs's TCP port.
    Autopilot control uses the V2 Autopilot API (engage, mode, target).
    Rudder angle is read via HTTP GET from the V1 API.

    Each simulation step:
    1. Push sensor data as NMEA sentences (heading, wind, speed)
    2. Read rudder angle via HTTP GET
    3. Return ControlCommand with the current rudder angle
    """

    def __init__(
        self,
        host: str = "localhost",
        http_port: int = 3000,
        nmea_port: int = 10110,
        device_id: str = "default",
        rudder_max: float = math.radians(30.0),
        mode: str = "compass",
        sim_sleep_ms: float = 0.0,
    ) -> None:
        self.host = host
        self.http_port = http_port
        self.nmea_port = nmea_port
        self.device_id = device_id
        self.rudder_max = rudder_max
        self.mode = mode
        self._sim_sleep_s = sim_sleep_ms / 1000.0

        self._target_heading = 0.0
        self._rudder_angle = 0.0
        self._last_rudder = 0.0
        self._connected = False
        self._engaged = False
        self._nmea_sock: socket.socket | None = None

    @property
    def _base_url(self) -> str:
        return f"http://{self.host}:{self.http_port}"

    @property
    def _autopilot_url(self) -> str:
        return f"{self._base_url}/signalk/v2/api/vessels/self/autopilots/{self.device_id}"

    @property
    def _rudder_url(self) -> str:
        return f"{self._base_url}/signalk/v1/api/vessels/self/steering/rudderAngle"

    # -- public API (AutopilotProtocol) --

    def set_target_heading(self, heading: float) -> None:
        """Set desired heading [rad] and publish to signalk-rs."""
        self._target_heading = heading
        try:
            self._put_target(heading)
        except OSError:
            logger.debug("Could not publish target heading to signalk-rs")

    def compute(self, sensors: SensorData, dt: float) -> ControlCommand:
        """Push sensors via NMEA, read rudder via HTTP, return control command."""
        try:
            self._ensure_connected()
            self._push_sensors_nmea(sensors)
            if self._sim_sleep_s > 0:
                time.sleep(self._sim_sleep_s)
            rudder_rad = self._read_rudder()
            self._last_rudder = rudder_rad
            self._connected = True
        except OSError as exc:
            if not self._connected:
                raise ConnectionError(
                    f"Cannot reach signalk-rs at {self.host}:{self.http_port}: {exc}"
                ) from exc
            logger.warning(
                "signalk-rs unreachable, using last rudder %.3f rad",
                self._last_rudder,
            )
            rudder_rad = self._last_rudder
            self._close_sockets()

        self._rudder_angle = rudder_rad
        return ControlCommand(rudder_angle=self._rudder_angle, sail_trim=0.5)

    # -- connection management --

    def _ensure_connected(self) -> None:
        """Lazy-connect NMEA socket and engage autopilot via V2 API."""
        if self._nmea_sock is not None:
            return

        # NMEA connection (sensor data injection)
        nmea_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        nmea_sock.settimeout(2.0)
        nmea_sock.connect((self.host, self.nmea_port))
        self._nmea_sock = nmea_sock

        # Engage autopilot via V2 API
        if not self._engaged:
            self._post(f"{self._autopilot_url}/engage")
            self._put_json(f"{self._autopilot_url}/mode", {"value": self.mode})
            self._put_target(self._target_heading)
            self._engaged = True

    # -- V2 Autopilot API --

    def _put_target(self, heading_rad: float) -> None:
        """PUT target heading (radians) to V2 API."""
        self._put_json(f"{self._autopilot_url}/target", {"value": heading_rad})

    def _post(self, url: str) -> None:
        """Send a POST request (empty body)."""
        req = urllib.request.Request(url, data=b"", method="POST")
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=2) as resp:
                resp.read()
        except urllib.error.URLError as exc:
            raise OSError(f"POST {url} failed: {exc}") from exc

    def _put_json(self, url: str, data: dict) -> None:  # type: ignore[type-arg]
        """Send a PUT request with JSON body."""
        body = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="PUT")
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=2) as resp:
                resp.read()
        except urllib.error.URLError as exc:
            raise OSError(f"PUT {url} failed: {exc}") from exc

    # -- NMEA sensor push --

    def _push_sensors_nmea(self, sensors: SensorData) -> None:
        """Push sensor values to signalk-rs via NMEA-0183 sentences."""
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

    # -- HTTP rudder readback --

    def _read_rudder(self) -> float:
        """Read current rudder angle from signalk-rs V1 API.

        Returns rudder angle in radians. The V1 API returns the value
        in radians directly (SignalK convention).
        """
        try:
            req = urllib.request.Request(self._rudder_url)
            with urllib.request.urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise OSError(f"GET rudder failed: {exc}") from exc

        if isinstance(data, dict):
            return float(data.get("value", self._last_rudder))
        return float(data)

    def _close_sockets(self) -> None:
        """Close NMEA TCP connection and reset for reconnection."""
        if self._nmea_sock is not None:
            with contextlib.suppress(OSError):
                self._nmea_sock.close()
        self._nmea_sock = None

    def close(self) -> None:
        """Close connections and optionally disengage autopilot."""
        if self._engaged:
            with contextlib.suppress(OSError):
                self._post(f"{self._autopilot_url}/disengage")
            self._engaged = False
        self._close_sockets()
