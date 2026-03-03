"""Telemetry recorder for simulation data."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sailsim.core.types import (
    ControlCommand,
    CurrentState,
    ForceData,
    SensorData,
    VesselState,
    WaveState,
    Waypoint,
    WindState,
)


@dataclass
class TimeStep:
    """Single recorded time step."""

    t: float
    state: VesselState
    sensors: SensorData
    control: ControlCommand
    wind: WindState
    forces: ForceData | None = None
    current: CurrentState | None = None
    waves: WaveState | None = None
    target_heading: float | None = None
    active_waypoint_index: int | None = None
    rudder_torque: float | None = None


class Recorder:
    """Records simulation telemetry for analysis."""

    def __init__(self) -> None:
        self.steps: list[TimeStep] = []
        self.route: list[Waypoint] | None = None

    def record(
        self,
        t: float,
        state: VesselState,
        sensors: SensorData,
        control: ControlCommand,
        wind: WindState,
        forces: ForceData | None = None,
        current: CurrentState | None = None,
        waves: WaveState | None = None,
        target_heading: float | None = None,
        active_waypoint_index: int | None = None,
        rudder_torque: float | None = None,
    ) -> None:
        """Record a single time step."""
        self.steps.append(
            TimeStep(
                t=t,
                state=VesselState(eta=state.eta.copy(), nu=state.nu.copy()),
                sensors=sensors,
                control=control,
                wind=wind,
                forces=forces,
                current=current,
                waves=waves,
                target_heading=target_heading,
                active_waypoint_index=active_waypoint_index,
                rudder_torque=rudder_torque,
            )
        )

    def desired_track(self) -> tuple[list[float], list[float]]:
        """Compute dead-reckoned ideal trajectory from target headings.

        Uses target_heading as the direction and actual speed_through_water
        as the magnitude at each timestep.  Falls back to actual heading if
        no target_heading is recorded.

        Returns (xs, ys) lists of North and East positions.
        """
        if not self.steps:
            return [], []

        xs = [self.steps[0].state.x]
        ys = [self.steps[0].state.y]

        for i in range(1, len(self.steps)):
            dt = self.steps[i].t - self.steps[i - 1].t
            speed = self.steps[i - 1].sensors.speed_through_water
            heading = self.steps[i - 1].target_heading
            if heading is None:
                heading = self.steps[i - 1].sensors.heading

            xs.append(xs[-1] + speed * np.cos(heading) * dt)
            ys.append(ys[-1] + speed * np.sin(heading) * dt)

        return xs, ys

    def to_csv(self, path: str | Path) -> None:
        """Export telemetry to CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "t",
                    "x",
                    "y",
                    "psi_deg",
                    "u",
                    "v",
                    "r",
                    "heading_deg",
                    "speed",
                    "roll_deg",
                    "rudder_deg",
                    "aw_speed",
                    "aw_angle_deg",
                    "tw_speed",
                    "tw_dir_deg",
                ]
            )
            for step in self.steps:
                writer.writerow(
                    [
                        f"{step.t:.3f}",
                        f"{step.state.x:.2f}",
                        f"{step.state.y:.2f}",
                        f"{np.degrees(step.state.psi):.2f}",
                        f"{step.state.u:.3f}",
                        f"{step.state.v:.3f}",
                        f"{step.state.r:.4f}",
                        f"{np.degrees(step.sensors.heading):.2f}",
                        f"{step.sensors.speed_through_water:.3f}",
                        f"{np.degrees(step.sensors.roll):.2f}",
                        f"{np.degrees(step.control.rudder_angle):.2f}",
                        f"{step.sensors.apparent_wind_speed:.2f}",
                        f"{np.degrees(step.sensors.apparent_wind_angle):.2f}",
                        f"{step.wind.speed:.2f}",
                        f"{np.degrees(step.wind.direction):.2f}",
                    ]
                )

    @property
    def headings_deg(self) -> list[float]:
        """All recorded headings in degrees."""
        return [np.degrees(s.sensors.heading) for s in self.steps]

    @property
    def rudder_angles_deg(self) -> list[float]:
        """All recorded rudder angles in degrees."""
        return [np.degrees(s.control.rudder_angle) for s in self.steps]

    @property
    def times(self) -> list[float]:
        """All recorded timestamps."""
        return [s.t for s in self.steps]

    def to_json(self, path: str | Path, metadata: dict | None = None) -> None:
        """Save recording to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        steps_data = []
        for step in self.steps:
            step_dict = {
                "t": step.t,
                "state": {
                    "eta": step.state.eta.tolist(),
                    "nu": step.state.nu.tolist(),
                },
                "sensors": {
                    "heading": step.sensors.heading,
                    "speed_through_water": step.sensors.speed_through_water,
                    "x": step.sensors.x,
                    "y": step.sensors.y,
                    "roll": step.sensors.roll,
                    "yaw_rate": step.sensors.yaw_rate,
                    "apparent_wind_speed": step.sensors.apparent_wind_speed,
                    "apparent_wind_angle": step.sensors.apparent_wind_angle,
                    "speed_over_ground": step.sensors.speed_over_ground,
                    "course_over_ground": step.sensors.course_over_ground,
                },
                "control": {
                    "rudder_angle": step.control.rudder_angle,
                    "sail_trim": step.control.sail_trim,
                },
                "wind": {
                    "speed": step.wind.speed,
                    "direction": step.wind.direction,
                },
                "forces": None,
                "current": None,
            }
            if step.forces is not None:
                step_dict["forces"] = {
                    "sail": step.forces.sail.tolist(),
                    "rudder": step.forces.rudder.tolist(),
                    "keel": step.forces.keel.tolist(),
                    "waves": step.forces.waves.tolist(),
                }
            if step.current is not None:
                step_dict["current"] = {
                    "speed": step.current.speed,
                    "direction": step.current.direction,
                }
            step_dict["waves"] = None
            if step.waves is not None:
                step_dict["waves"] = {
                    "Hs": step.waves.Hs,
                    "Tp": step.waves.Tp,
                    "direction": step.waves.direction,
                    "elevation": step.waves.elevation,
                }
            step_dict["target_heading"] = step.target_heading
            step_dict["active_waypoint_index"] = step.active_waypoint_index
            step_dict["rudder_torque"] = step.rudder_torque
            steps_data.append(step_dict)

        doc = {
            "version": 2,
            "metadata": metadata or {},
            "route": None,
            "steps": steps_data,
        }
        if self.route:
            doc["route"] = [{"x": wp.x, "y": wp.y, "tolerance": wp.tolerance} for wp in self.route]
        with path.open("w") as f:
            json.dump(doc, f)

    @classmethod
    def from_json(cls, path: str | Path) -> Recorder:
        """Load recording from JSON file."""
        path = Path(path)
        with path.open() as f:
            doc = json.load(f)

        recorder = cls()

        route_data = doc.get("route")
        if route_data:
            recorder.route = [Waypoint(**wp) for wp in route_data]

        for s in doc["steps"]:
            forces = None
            if s.get("forces"):
                waves_data = s["forces"].get("waves", [0.0, 0.0, 0.0])
                forces = ForceData(
                    sail=np.array(s["forces"]["sail"]),
                    rudder=np.array(s["forces"]["rudder"]),
                    keel=np.array(s["forces"]["keel"]),
                    waves=np.array(waves_data),
                )
            current = None
            if s.get("current"):
                current = CurrentState(**s["current"])

            # Handle old JSON files missing new sensor fields
            sensor_data = s["sensors"]
            sensor_data.setdefault("speed_over_ground", 0.0)
            sensor_data.setdefault("course_over_ground", 0.0)

            waves = None
            if s.get("waves"):
                waves = WaveState(**s["waves"])

            target_heading = s.get("target_heading")
            active_waypoint_index = s.get("active_waypoint_index")
            rudder_torque = s.get("rudder_torque")

            recorder.steps.append(
                TimeStep(
                    t=s["t"],
                    state=VesselState(
                        eta=np.array(s["state"]["eta"]),
                        nu=np.array(s["state"]["nu"]),
                    ),
                    sensors=SensorData(**sensor_data),
                    control=ControlCommand(**s["control"]),
                    wind=WindState(**s["wind"]),
                    forces=forces,
                    current=current,
                    waves=waves,
                    target_heading=target_heading,
                    active_waypoint_index=active_waypoint_index,
                    rudder_torque=rudder_torque,
                )
            )
        return recorder
