"""Simulation runner - the main time-stepping loop."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from sailsim.core.types import ControlCommand, SensorData, VesselState, Waypoint
from sailsim.environment import build_current_model, build_wave_model, build_wind_model
from sailsim.recording.recorder import Recorder
from sailsim.vessel.yacht_3dof import Yacht3DOF, YachtParams
from sailsim.vessel.yacht_6dof import Yacht6DOF, YachtParams6DOF

if TYPE_CHECKING:
    from sailsim.autopilot.base import AutopilotProtocol
    from sailsim.core.config import ScenarioConfig


def _compute_waypoint_heading(
    boat_x: float,
    boat_y: float,
    wp_x: float,
    wp_y: float,
) -> float:
    """Compute heading from boat position to waypoint (line-of-sight).

    Returns heading [rad] in NED convention (0 = North, pi/2 = East).
    """
    dx = wp_x - boat_x  # North delta
    dy = wp_y - boat_y  # East delta
    return float(np.arctan2(dy, dx))


def _distance_to_waypoint(
    boat_x: float,
    boat_y: float,
    wp_x: float,
    wp_y: float,
) -> float:
    """Euclidean distance from boat to waypoint [m]."""
    return float(np.sqrt((wp_x - boat_x) ** 2 + (wp_y - boat_y) ** 2))


def run_scenario(config: ScenarioConfig, autopilot: AutopilotProtocol) -> Recorder:
    """Execute a complete simulation scenario.

    Args:
        config: scenario configuration
        autopilot: autopilot implementation to test

    Returns:
        Recorder with all telemetry data.
    """
    # Build yacht from config
    yacht_cfg = config.yacht.model_dump()
    dof = yacht_cfg.pop("dof", 3)
    yacht: Yacht6DOF | Yacht3DOF
    if dof == 6:
        # Filter to only fields that YachtParams6DOF accepts
        valid_fields = {f.name for f in YachtParams6DOF.__dataclass_fields__.values()}
        yacht = Yacht6DOF(
            YachtParams6DOF(**{k: v for k, v in yacht_cfg.items() if k in valid_fields})
        )
    else:
        # Filter to only fields that YachtParams accepts
        valid_fields = {f.name for f in YachtParams.__dataclass_fields__.values()}
        yacht = Yacht3DOF(YachtParams(**{k: v for k, v in yacht_cfg.items() if k in valid_fields}))

    # Set initial state
    initial_eta = np.zeros(6)
    initial_eta[0] = config.initial_state.x
    initial_eta[1] = config.initial_state.y
    initial_eta[5] = config.initial_state.psi

    initial_nu = np.zeros(6)
    initial_nu[0] = config.initial_state.u

    yacht.reset(VesselState(eta=initial_eta, nu=initial_nu))

    # Set up environment
    wind_model = build_wind_model(config.wind)
    current_model = build_current_model(config.current)
    wave_model = build_wave_model(config.waves)

    # Set autopilot target
    target_heading = config.target_heading
    autopilot.set_target_heading(target_heading)

    # Prepare maneuver queue (sorted by time)
    maneuver_steps = sorted(config.maneuvers.steps, key=lambda s: s.time_s)
    maneuver_idx = 0

    # Prepare waypoint route (mutually exclusive with maneuvers)
    waypoints: list[Waypoint] = [
        Waypoint(x=wp.x, y=wp.y, tolerance=wp.tolerance) for wp in config.route.waypoints
    ]
    use_waypoints = bool(waypoints) and not maneuver_steps
    active_wp_idx: int | None = 0 if use_waypoints else None
    route_complete = False

    if use_waypoints:
        target_heading = _compute_waypoint_heading(
            config.initial_state.x,
            config.initial_state.y,
            waypoints[0].x,
            waypoints[0].y,
        )
        autopilot.set_target_heading(target_heading)

    # Recorder
    recorder = Recorder()
    if waypoints:
        recorder.route = waypoints

    # Time loop
    t = 0.0
    dt = config.dt
    n_steps = int(config.duration_s / dt)

    control = ControlCommand()

    for _ in range(n_steps):
        # Dispatch any maneuvers that are due
        while maneuver_idx < len(maneuver_steps) and t >= maneuver_steps[maneuver_idx].time_s:
            target_heading = maneuver_steps[maneuver_idx].target_heading
            autopilot.set_target_heading(target_heading)
            maneuver_idx += 1

        # Waypoint guidance (line-of-sight)
        if use_waypoints and not route_complete and active_wp_idx is not None:
            wp = waypoints[active_wp_idx]
            dist = _distance_to_waypoint(
                yacht.state.x,
                yacht.state.y,
                wp.x,
                wp.y,
            )
            if dist <= wp.tolerance:
                active_wp_idx += 1
                if active_wp_idx >= len(waypoints):
                    route_complete = True
                    active_wp_idx = len(waypoints) - 1
                else:
                    wp = waypoints[active_wp_idx]
            if not route_complete:
                target_heading = _compute_waypoint_heading(
                    yacht.state.x,
                    yacht.state.y,
                    wp.x,
                    wp.y,
                )
                autopilot.set_target_heading(target_heading)

        wind = wind_model.get(t)
        current = current_model.get(t)
        wave_model.set_boat_position(yacht.state.x, yacht.state.y)
        waves = wave_model.get(t)

        # Get sensor readings from current state
        sensors = SensorData.from_state(yacht.state, wind, current)

        # Compute force breakdown at current state (before step)
        forces = yacht.compute_forces(wind, control, waves)
        rudder_torque = float(-forces.rudder[1] * config.yacht.rudder_cp_offset)

        # Record current state with forces
        recorder.record(
            t,
            yacht.state,
            sensors,
            control,
            wind,
            forces=forces,
            current=current,
            waves=waves,
            target_heading=target_heading,
            active_waypoint_index=active_wp_idx,
            rudder_torque=rudder_torque,
        )

        # Autopilot computes new control
        control = autopilot.compute(sensors, dt)

        # Step physics
        yacht.step(wind, control, dt, current=current, waves=waves)

        t += dt

    # Record final state
    wind = wind_model.get(t)
    current = current_model.get(t)
    wave_model.set_boat_position(yacht.state.x, yacht.state.y)
    waves = wave_model.get(t)
    sensors = SensorData.from_state(yacht.state, wind, current)
    forces = yacht.compute_forces(wind, control, waves)
    rudder_torque = float(-forces.rudder[1] * config.yacht.rudder_cp_offset)
    recorder.record(
        t,
        yacht.state,
        sensors,
        control,
        wind,
        forces=forces,
        current=current,
        waves=waves,
        target_heading=target_heading,
        active_waypoint_index=active_wp_idx,
        rudder_torque=rudder_torque,
    )

    return recorder
