"""SignalK autopilot adapter.

Implements :class:`AutopilotProtocol` by publishing sensor data to a SignalK
server and reading rudder commands back — all via HTTP (stdlib only).
"""

from __future__ import annotations

import json
import logging
import math
import urllib.error
import urllib.request
from typing import Any

from sailsim.core.types import ControlCommand, SensorData

logger = logging.getLogger(__name__)

# SignalK path mapping: SensorData field → (SignalK path, conversion)
_SENSOR_PATHS: list[tuple[str, str]] = [
    ("heading", "navigation.headingMagnetic"),
    ("speed_through_water", "navigation.speedThroughWater"),
    ("speed_over_ground", "navigation.speedOverGround"),
    ("course_over_ground", "navigation.courseOverGroundTrue"),
    ("yaw_rate", "navigation.rateOfTurn"),
    ("roll", "navigation.attitude.roll"),
    ("apparent_wind_speed", "environment.wind.speedApparent"),
    ("apparent_wind_angle", "environment.wind.angleApparent"),
]

_RUDDER_PATH = "steering.rudderAngle"
_TARGET_HEADING_PATH = "steering.autopilot.target.headingMagnetic"


class SignalKAutopilot:
    """Autopilot adapter that communicates via a SignalK server.

    Each simulation step:
    1. PUT sensor data to the server (one delta message)
    2. GET rudder angle from the server
    """

    def __init__(self, url: str = "http://localhost:3000", timeout: float = 0.5) -> None:
        self.base_url = url.rstrip("/")
        self.timeout = timeout
        self._target_heading = 0.0
        self._last_rudder = 0.0
        self._connected = False

    def set_target_heading(self, heading: float) -> None:
        """Set desired heading [rad] and publish to SignalK."""
        self._target_heading = heading
        try:
            self._put_value(_TARGET_HEADING_PATH, heading)
        except (urllib.error.URLError, OSError):
            logger.debug("Could not publish target heading to SignalK")

    def compute(self, sensors: SensorData, dt: float) -> ControlCommand:
        """Publish sensors, read rudder command from SignalK server."""
        try:
            self._publish_sensors(sensors)
            rudder = self._read_rudder()
            self._last_rudder = rudder
            self._connected = True
        except (urllib.error.URLError, OSError) as exc:
            if not self._connected:
                raise ConnectionError(
                    f"Cannot reach SignalK server at {self.base_url}: {exc}"
                ) from exc
            logger.warning("SignalK unreachable, using last rudder value %.3f", self._last_rudder)
            rudder = self._last_rudder

        return ControlCommand(rudder_angle=rudder, sail_trim=0.5)

    # -- internal helpers --

    def _publish_sensors(self, sensors: SensorData) -> None:
        """Send all sensor values to SignalK as a delta update."""
        delta = self._build_delta(sensors)
        data = json.dumps(delta).encode()
        req = urllib.request.Request(
            f"{self.base_url}/signalk/v1/api/",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=self.timeout)

    def _read_rudder(self) -> float:
        """Read current rudder angle [rad] from SignalK."""
        url = f"{self.base_url}/signalk/v1/api/vessels/self/{_RUDDER_PATH.replace('.', '/')}"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read())
        value = body.get("value", 0.0)
        return float(value) if value is not None else 0.0

    def _put_value(self, path: str, value: float) -> None:
        """PUT a single value to SignalK."""
        delta = {
            "updates": [
                {
                    "values": [{"path": path, "value": value}],
                }
            ]
        }
        data = json.dumps(delta).encode()
        req = urllib.request.Request(
            f"{self.base_url}/signalk/v1/api/",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=self.timeout)
    @staticmethod
    def _build_delta(sensors: SensorData) -> dict[str, Any]:
        """Build a SignalK delta message from sensor data."""
        values: list[dict[str, Any]] = []
        for attr, sk_path in _SENSOR_PATHS:
            val = getattr(sensors, attr, None)
            if val is not None and not (isinstance(val, float) and math.isnan(val)):
                values.append({"path": sk_path, "value": val})
        return {"updates": [{"values": values}]}
