"""Core data types for the sailing yacht simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class VesselState:
    """Full 6-DOF state of the vessel.

    Always carries 6-element vectors even when using reduced-DOF models,
    so downstream code (sensors, recorder, autopilot) never needs to change.

    eta = [x, y, z, phi, theta, psi]  (NED position + Euler angles)
    nu  = [u, v, w, p, q, r]          (body-fixed velocities + rotation rates)
    """

    eta: NDArray[np.float64] = field(default_factory=lambda: np.zeros(6))
    nu: NDArray[np.float64] = field(default_factory=lambda: np.zeros(6))

    @property
    def x(self) -> float:
        return float(self.eta[0])

    @property
    def y(self) -> float:
        return float(self.eta[1])

    @property
    def z(self) -> float:
        return float(self.eta[2])

    @property
    def phi(self) -> float:
        """Roll angle [rad]."""
        return float(self.eta[3])

    @property
    def theta(self) -> float:
        """Pitch angle [rad]."""
        return float(self.eta[4])

    @property
    def psi(self) -> float:
        """Heading / yaw angle [rad]."""
        return float(self.eta[5])

    @property
    def u(self) -> float:
        """Surge velocity [m/s]."""
        return float(self.nu[0])

    @property
    def v(self) -> float:
        """Sway velocity [m/s]."""
        return float(self.nu[1])

    @property
    def speed(self) -> float:
        """Speed through water [m/s]."""
        return float(np.sqrt(self.nu[0] ** 2 + self.nu[1] ** 2))

    @property
    def r(self) -> float:
        """Yaw rate [rad/s]."""
        return float(self.nu[5])


@dataclass
class WindState:
    """True wind conditions.

    speed: true wind speed [m/s]
    direction: true wind direction [rad], where the wind comes FROM in NED frame
                (0 = from North, pi/2 = from East)
    """

    speed: float = 0.0
    direction: float = 0.0


@dataclass
class CurrentState:
    """Ocean current conditions.

    speed: current speed [m/s]
    direction: direction current FLOWS TOWARDS [rad] in NED frame
               (0 = flows north, pi/2 = flows east)
    """

    speed: float = 0.0
    direction: float = 0.0

    @property
    def velocity_ned(self) -> tuple[float, float]:
        """Current velocity in NED frame (north, east) [m/s]."""
        return (
            self.speed * np.cos(self.direction),
            self.speed * np.sin(self.direction),
        )


@dataclass
class WaveState:
    """Sea state at the vessel's position.

    Hs: significant wave height [m]
    Tp: peak period [s]
    direction: wave propagation direction [rad] in NED
    elevation: instantaneous surface elevation at vessel [m]
    """

    Hs: float = 0.0
    Tp: float = 0.0
    direction: float = 0.0
    elevation: float = 0.0


@dataclass
class SensorData:
    """Sensor readings available to the autopilot.

    In Phase 1 these are perfect (no noise). Phase 2 adds noise models.
    """

    heading: float = 0.0  # magnetic heading [rad]
    speed_through_water: float = 0.0  # [m/s]
    x: float = 0.0  # NED north position [m]
    y: float = 0.0  # NED east position [m]
    roll: float = 0.0  # [rad]
    yaw_rate: float = 0.0  # [rad/s]
    apparent_wind_speed: float = 0.0  # [m/s]
    apparent_wind_angle: float = 0.0  # [rad], relative to bow
    speed_over_ground: float = 0.0  # [m/s]
    course_over_ground: float = 0.0  # [rad]

    @staticmethod
    def from_state(
        state: VesselState,
        wind: WindState,
        current: CurrentState | None = None,
    ) -> SensorData:
        """Extract perfect sensor readings from vessel state."""
        from sailsim.physics.aerodynamics import apparent_wind

        aw_speed, awa = apparent_wind(
            wind.speed,
            wind.direction,
            state.nu[0],
            state.nu[1],
            state.psi,
        )

        # Ground velocity = boat velocity in NED + current
        from sailsim.physics.dynamics import rotation_matrix_3dof

        R = rotation_matrix_3dof(state.psi)
        boat_ned = R @ np.array([state.nu[0], state.nu[1], state.nu[5]])
        vn = boat_ned[0]
        ve = boat_ned[1]

        if current is not None:
            cn, ce = current.velocity_ned
            vn += cn
            ve += ce

        sog = float(np.sqrt(vn**2 + ve**2))
        cog = float(np.arctan2(ve, vn))

        return SensorData(
            heading=state.psi,
            speed_through_water=state.speed,
            x=state.x,
            y=state.y,
            roll=state.phi,
            yaw_rate=state.r,
            apparent_wind_speed=aw_speed,
            apparent_wind_angle=awa,
            speed_over_ground=sog,
            course_over_ground=cog,
        )


@dataclass
class ForceData:
    """Per-component force breakdown in body frame.

    3-DOF: each field is a 3-element array [X, Y, N].
    6-DOF: each field is a 6-element array [X, Y, Z, K, M, N].
    X is always at [0], Y at [1]. N (yaw moment) is at [2] for 3-DOF, [5] for 6-DOF.
    """

    sail: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))
    rudder: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))
    keel: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))
    waves: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))

    @property
    def total(self) -> NDArray[np.float64]:
        return self.sail + self.rudder + self.keel + self.waves


@dataclass
class Waypoint:
    """A navigation waypoint in NED coordinates.

    x: North position [m]
    y: East position [m]
    tolerance: radius within which the waypoint is considered reached [m]
    """

    x: float = 0.0
    y: float = 0.0
    tolerance: float = 10.0


@dataclass
class ControlCommand:
    """Control outputs from the autopilot.

    rudder_angle: commanded rudder angle [rad], positive = starboard
    sail_trim: sail trim factor [0..1], 0 = fully eased, 1 = fully sheeted
               (Phase 1: not used, fixed at a reasonable default)
    """

    rudder_angle: float = 0.0
    sail_trim: float = 0.5
