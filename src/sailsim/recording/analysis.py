"""Quality gate analysis for scenario verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sailsim.core.config import QualityGateConfig
    from sailsim.recording.recorder import Recorder


@dataclass
class QualityResult:
    """Result of quality gate evaluation."""

    passed: bool
    details: dict[str, dict]

    def summary(self) -> str:
        """Human-readable summary."""
        lines = []
        status = "PASS" if self.passed else "FAIL"
        lines.append(f"Quality Gate Result: {status}")
        lines.append("-" * 40)
        for name, info in self.details.items():
            check = "OK" if info["passed"] else "FAIL"
            lines.append(f"  [{check}] {name}: {info['value']:.2f} (limit: {info['limit']:.2f})")
        return "\n".join(lines)


def evaluate_heading_hold(
    recorder: Recorder,
    target_heading_deg: float,
    gates: QualityGateConfig,
    settling_start_s: float = 10.0,
) -> QualityResult:
    """Evaluate a heading-hold scenario against quality gates.

    Args:
        recorder: completed simulation recording
        target_heading_deg: desired heading [degrees]
        gates: quality gate thresholds
        settling_start_s: time after which to start measuring steady-state error

    Returns:
        QualityResult with pass/fail and details.
    """
    details: dict[str, dict] = {}
    all_passed = True

    # Extract heading errors (after settling period)
    heading_errors = []
    for step in recorder.steps:
        if step.t >= settling_start_s:
            error = target_heading_deg - np.degrees(step.sensors.heading)
            # Wrap to [-180, 180]
            error = (error + 180) % 360 - 180
            heading_errors.append(abs(error))

    if heading_errors:
        mean_error = float(np.mean(heading_errors))
        max_error = float(np.max(heading_errors))
    else:
        mean_error = 999.0
        max_error = 999.0

    # Check mean heading error
    mean_ok = mean_error <= gates.max_mean_heading_error_deg
    details["mean_heading_error_deg"] = {
        "value": mean_error,
        "limit": gates.max_mean_heading_error_deg,
        "passed": mean_ok,
    }
    all_passed &= mean_ok

    # Check max heading deviation
    max_ok = max_error <= gates.max_heading_deviation_deg
    details["max_heading_deviation_deg"] = {
        "value": max_error,
        "limit": gates.max_heading_deviation_deg,
        "passed": max_ok,
    }
    all_passed &= max_ok

    # Check rudder rate (detect oscillation)
    rudder_rates = []
    for i in range(1, len(recorder.steps)):
        dt = recorder.steps[i].t - recorder.steps[i - 1].t
        if dt > 0:
            dr = abs(
                np.degrees(recorder.steps[i].control.rudder_angle)
                - np.degrees(recorder.steps[i - 1].control.rudder_angle)
            )
            rudder_rates.append(dr / dt)

    max_rudder_rate = float(np.max(rudder_rates)) if rudder_rates else 0.0
    rate_ok = max_rudder_rate <= gates.max_rudder_rate_deg_per_s
    details["max_rudder_rate_deg_per_s"] = {
        "value": max_rudder_rate,
        "limit": gates.max_rudder_rate_deg_per_s,
        "passed": rate_ok,
    }
    all_passed &= rate_ok

    # Check settling time (time to reach ±threshold of target)
    threshold_deg = gates.max_mean_heading_error_deg
    settling_time = float(recorder.steps[-1].t) if recorder.steps else 999.0
    for step in recorder.steps:
        error = abs(target_heading_deg - np.degrees(step.sensors.heading))
        error = min(error, 360 - error)
        if error <= threshold_deg:
            settling_time = step.t
            break

    settle_ok = settling_time <= gates.max_settling_time_s
    details["settling_time_s"] = {
        "value": settling_time,
        "limit": gates.max_settling_time_s,
        "passed": settle_ok,
    }
    all_passed &= settle_ok

    return QualityResult(passed=all_passed, details=details)


@dataclass
class ManeuverResult:
    """Result of a single maneuver evaluation."""

    completion_time_s: float
    overshoot_deg: float
    speed_loss_pct: float
    completed: bool


def evaluate_maneuver(
    recorder: Recorder,
    maneuver_time_s: float,
    target_heading_deg: float,
    threshold_deg: float = 5.0,
    window_s: float = 60.0,
) -> ManeuverResult:
    """Evaluate a single course change maneuver.

    Args:
        recorder: simulation recording
        maneuver_time_s: time at which the maneuver was commanded
        target_heading_deg: target heading after maneuver [degrees]
        threshold_deg: heading error threshold for "complete" [degrees]
        window_s: max time window to look for completion

    Returns:
        ManeuverResult with completion time, overshoot, and speed loss.
    """
    # Find steps within the maneuver window
    start_idx = None
    for i, step in enumerate(recorder.steps):
        if step.t >= maneuver_time_s:
            start_idx = i
            break

    if start_idx is None:
        return ManeuverResult(
            completion_time_s=window_s,
            overshoot_deg=0.0,
            speed_loss_pct=0.0,
            completed=False,
        )

    # Speed at maneuver start
    speed_at_start = recorder.steps[start_idx].sensors.speed_through_water

    completion_time = window_s
    completed = False
    max_overshoot = 0.0
    min_speed = speed_at_start

    for step in recorder.steps[start_idx:]:
        if step.t > maneuver_time_s + window_s:
            break

        heading_deg = np.degrees(step.sensors.heading)
        error = target_heading_deg - heading_deg
        error = (error + 180) % 360 - 180

        if not completed and abs(error) <= threshold_deg:
            completion_time = step.t - maneuver_time_s
            completed = True

        # Track overshoot after first reaching target
        if completed:
            max_overshoot = max(max_overshoot, abs(error))

        min_speed = min(min_speed, step.sensors.speed_through_water)

    speed_loss_pct = 0.0
    if speed_at_start > 0.01:
        speed_loss_pct = (1.0 - min_speed / speed_at_start) * 100.0

    return ManeuverResult(
        completion_time_s=completion_time,
        overshoot_deg=max_overshoot,
        speed_loss_pct=speed_loss_pct,
        completed=completed,
    )


@dataclass
class SteeringEffortResult:
    """Result of steering effort evaluation."""

    peak_torque_nm: float  # max |T|
    mean_torque_nm: float  # mean |T|
    total_energy_j: float  # cumulative energy [J]
    energy_rate_j_per_s: float  # mean power [W]


def evaluate_steering_effort(recorder: Recorder) -> SteeringEffortResult:
    """Evaluate rudder steering effort from recorded torque data.

    Args:
        recorder: completed simulation recording (with rudder_torque)

    Returns:
        SteeringEffortResult with peak/mean torque, energy, and power.
    """
    torques = [s.rudder_torque for s in recorder.steps if s.rudder_torque is not None]

    if not torques:
        return SteeringEffortResult(
            peak_torque_nm=0.0,
            mean_torque_nm=0.0,
            total_energy_j=0.0,
            energy_rate_j_per_s=0.0,
        )

    abs_torques = [abs(t) for t in torques]
    peak = float(np.max(abs_torques))
    mean = float(np.mean(abs_torques))

    # Cumulative energy: sum |T_i * delta_rudder_i|
    total_energy = 0.0
    for i in range(1, len(recorder.steps)):
        s_prev = recorder.steps[i - 1]
        s_curr = recorder.steps[i]
        if s_curr.rudder_torque is None:
            continue
        delta = abs(s_curr.control.rudder_angle - s_prev.control.rudder_angle)
        total_energy += abs(s_curr.rudder_torque) * delta

    duration = recorder.steps[-1].t - recorder.steps[0].t if len(recorder.steps) > 1 else 0.0
    rate = total_energy / duration if duration > 0 else 0.0

    return SteeringEffortResult(
        peak_torque_nm=peak,
        mean_torque_nm=mean,
        total_energy_j=total_energy,
        energy_rate_j_per_s=rate,
    )


@dataclass
class WaypointResult:
    """Result of waypoint route evaluation."""

    waypoints_reached: int
    total_waypoints: int
    all_reached: bool
    waypoint_times: list[float]

    def summary(self) -> str:
        status = "PASS" if self.all_reached else "FAIL"
        lines = [f"Waypoint Route: {status}"]
        lines.append(f"  Reached {self.waypoints_reached}/{self.total_waypoints}")
        for i, t in enumerate(self.waypoint_times):
            lines.append(f"  WP{i}: t={t:.1f}s")
        return "\n".join(lines)


def evaluate_waypoint_route(
    recorder: Recorder,
    waypoints: list,
) -> WaypointResult:
    """Evaluate whether all waypoints in a route were reached.

    Checks waypoints in order: a waypoint is reached when the boat is
    within its tolerance radius.  Only advances to the next waypoint after
    the previous one is reached.
    """
    reached = [False] * len(waypoints)
    reach_times: list[float] = []
    current_wp = 0

    for step in recorder.steps:
        if current_wp >= len(waypoints):
            break
        wp = waypoints[current_wp]
        dist = float(np.sqrt((step.state.x - wp.x) ** 2 + (step.state.y - wp.y) ** 2))
        if dist <= wp.tolerance:
            reached[current_wp] = True
            reach_times.append(step.t)
            current_wp += 1

    return WaypointResult(
        waypoints_reached=sum(reached),
        total_waypoints=len(waypoints),
        all_reached=all(reached),
        waypoint_times=reach_times,
    )
