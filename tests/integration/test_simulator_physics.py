"""Simulator physics validation tests with static rudder (no controller).

These tests verify the physical plausibility of the simulator WITHOUT
any autopilot control. The rudder is set to a fixed angle, and we verify
that the yacht behaves as expected under known conditions.

This is critical: we must trust the physics before testing any autopilot.
"""

import numpy as np

from sailsim.autopilot.fixed import FixedRudderAutopilot
from sailsim.core.config import InitialStateConfig, ScenarioConfig, WindConfig
from sailsim.core.runner import run_scenario

# ──────────────────────────────────────────────────────────────────────
# Helper: run a short simulation with fixed rudder
# ──────────────────────────────────────────────────────────────────────


def _run_fixed_rudder(
    rudder_deg: float = 0.0,
    wind_speed: float = 5.0,
    wind_dir_deg: float = 60.0,
    initial_speed: float = 2.0,
    initial_heading_deg: float = 0.0,
    duration_s: float = 30.0,
    dt: float = 0.05,
):
    """Run simulation with fixed rudder and return recorder."""
    config = ScenarioConfig(
        name="physics_test",
        duration_s=duration_s,
        dt=dt,
        wind=WindConfig(speed=wind_speed, direction=np.radians(wind_dir_deg)),
        initial_state=InitialStateConfig(
            u=initial_speed,
            psi=np.radians(initial_heading_deg),
        ),
    )
    autopilot = FixedRudderAutopilot(rudder_angle=np.radians(rudder_deg))
    return run_scenario(config, autopilot)


# ──────────────────────────────────────────────────────────────────────
# Test 1: No wind, no rudder → boat should drift to a stop
# ──────────────────────────────────────────────────────────────────────


class TestNoWindDriftToStop:
    """Without wind and with neutral rudder, the boat should slow down
    due to hydrodynamic damping and eventually stop."""

    def test_speed_decreases(self):
        rec = _run_fixed_rudder(
            rudder_deg=0.0,
            wind_speed=0.0,
            initial_speed=3.0,
            duration_s=60.0,
        )
        initial_speed = rec.steps[0].state.speed
        final_speed = rec.steps[-1].state.speed
        assert final_speed < initial_speed * 0.5, (
            f"Boat should decelerate significantly: {initial_speed:.2f} -> {final_speed:.2f}"
        )

    def test_heading_stays_constant(self):
        rec = _run_fixed_rudder(
            rudder_deg=0.0,
            wind_speed=0.0,
            initial_speed=3.0,
            duration_s=60.0,
        )
        headings = [np.degrees(s.state.psi) for s in rec.steps]
        heading_range = max(headings) - min(headings)
        assert heading_range < 1.0, (
            f"Heading should stay constant without wind/rudder, range={heading_range:.2f}°"
        )

    def test_straight_line_trajectory(self):
        rec = _run_fixed_rudder(
            rudder_deg=0.0,
            wind_speed=0.0,
            initial_speed=3.0,
            initial_heading_deg=0.0,
            duration_s=30.0,
        )
        # Heading north → x should increase, y should stay near 0
        final_x = rec.steps[-1].state.x
        final_y = rec.steps[-1].state.y
        assert final_x > 10.0, f"Boat should move forward: x={final_x:.2f}"
        assert abs(final_y) < 1.0, f"Boat should not drift sideways: y={final_y:.2f}"


# ──────────────────────────────────────────────────────────────────────
# Test 2: Rudder turns the boat
# ──────────────────────────────────────────────────────────────────────


class TestRudderTurnsBoat:
    """A fixed rudder deflection should turn the boat in the expected direction."""

    def test_starboard_rudder_turns_right(self):
        """Positive rudder angle should turn the boat to starboard (positive yaw)."""
        rec = _run_fixed_rudder(
            rudder_deg=10.0,
            wind_speed=0.0,
            initial_speed=3.0,
            duration_s=20.0,
        )
        final_heading = np.degrees(rec.steps[-1].state.psi)
        # Should have turned to starboard (positive heading change)
        assert final_heading > 5.0, (
            f"Starboard rudder should turn boat right: heading={final_heading:.2f}°"
        )

    def test_port_rudder_turns_left(self):
        """Negative rudder angle should turn the boat to port (negative yaw)."""
        rec = _run_fixed_rudder(
            rudder_deg=-10.0,
            wind_speed=0.0,
            initial_speed=3.0,
            duration_s=20.0,
        )
        final_heading = np.degrees(rec.steps[-1].state.psi)
        assert final_heading < -5.0, (
            f"Port rudder should turn boat left: heading={final_heading:.2f}°"
        )

    def test_more_rudder_turns_faster(self):
        """Larger rudder angle should produce more heading change.

        We compare total heading change over a short period rather than peak
        yaw rate, because higher rudder angles also induce more drag which
        slows the boat and limits peak yaw rate over longer simulations.
        """
        rec_5 = _run_fixed_rudder(
            rudder_deg=5.0,
            wind_speed=0.0,
            initial_speed=3.0,
            duration_s=2.0,
        )
        rec_15 = _run_fixed_rudder(
            rudder_deg=15.0,
            wind_speed=0.0,
            initial_speed=3.0,
            duration_s=2.0,
        )
        heading_5 = abs(np.degrees(rec_5.steps[-1].state.psi))
        heading_15 = abs(np.degrees(rec_15.steps[-1].state.psi))
        assert heading_15 > heading_5, (
            f"More rudder should turn more: 5°→{heading_5:.1f}°, 15°→{heading_15:.1f}°"
        )

    def test_no_turn_without_speed(self):
        """Rudder without boat speed should not produce significant turning."""
        rec = _run_fixed_rudder(
            rudder_deg=20.0,
            wind_speed=0.0,
            initial_speed=0.0,
            duration_s=10.0,
        )
        final_heading = abs(np.degrees(rec.steps[-1].state.psi))
        assert final_heading < 5.0, (
            f"Rudder should be ineffective without speed: heading={final_heading:.2f}°"
        )


# ──────────────────────────────────────────────────────────────────────
# Test 3: Wind propels the boat
# ──────────────────────────────────────────────────────────────────────


class TestWindPropulsion:
    """Wind from the side should accelerate the boat and create leeway."""

    def test_wind_accelerates_stationary_boat(self):
        """A boat at rest in wind should accelerate."""
        rec = _run_fixed_rudder(
            rudder_deg=0.0,
            wind_speed=8.0,
            wind_dir_deg=60.0,
            initial_speed=0.0,
            duration_s=30.0,
        )
        final_speed = rec.steps[-1].state.speed
        assert final_speed > 0.5, (
            f"Wind should accelerate the boat: final speed={final_speed:.2f} m/s"
        )

    def test_reach_produces_more_drive_than_close_hauled(self):
        """Close reach (60°) should produce more forward drive than very close-hauled (25°).
        Without a rudder to maintain heading, the boat will yaw freely, so we compare
        maximum speed reached rather than final speed."""
        rec_close = _run_fixed_rudder(
            rudder_deg=0.0,
            wind_speed=8.0,
            wind_dir_deg=25.0,
            initial_speed=0.0,
            duration_s=30.0,
        )
        rec_reach = _run_fixed_rudder(
            rudder_deg=0.0,
            wind_speed=8.0,
            wind_dir_deg=60.0,
            initial_speed=0.0,
            duration_s=30.0,
        )
        max_speed_close = max(s.state.speed for s in rec_close.steps)
        max_speed_reach = max(s.state.speed for s in rec_reach.steps)
        assert max_speed_reach > max_speed_close, (
            f"Close reach should accelerate faster: "
            f"reach={max_speed_reach:.2f}, close={max_speed_close:.2f}"
        )

    def test_no_force_head_to_wind(self):
        """Head-to-wind: minimal forward drive (in irons)."""
        rec = _run_fixed_rudder(
            rudder_deg=0.0,
            wind_speed=8.0,
            wind_dir_deg=0.0,
            initial_speed=0.0,
            duration_s=30.0,
        )
        final_speed = rec.steps[-1].state.speed
        # Should barely move when pointing directly into the wind
        assert final_speed < 0.5, (
            f"Head-to-wind should produce minimal speed: {final_speed:.2f} m/s"
        )


# ──────────────────────────────────────────────────────────────────────
# Test 4: Leeway (drift) behavior
# ──────────────────────────────────────────────────────────────────────


class TestLeeway:
    """Side wind should cause the boat to drift to leeward."""

    def test_crosswind_causes_drift(self):
        """Wind from the side should cause lateral displacement.
        Without rudder correction, the boat yaws freely under wind, so we
        check that the final position is NOT at the origin (drift occurred)."""
        rec = _run_fixed_rudder(
            rudder_deg=0.0,
            wind_speed=8.0,
            wind_dir_deg=60.0,
            initial_speed=2.0,
            initial_heading_deg=0.0,
            duration_s=20.0,
        )
        final_y = rec.steps[-1].state.y
        # The boat should have moved laterally (doesn't matter which way
        # because the boat yaws under sail moment)
        assert abs(final_y) > 0.5, f"Crosswind should cause lateral displacement: y={final_y:.2f}"

    def test_sway_velocity_with_crosswind(self):
        """Cross wind should produce non-zero sway velocity."""
        rec = _run_fixed_rudder(
            rudder_deg=0.0,
            wind_speed=8.0,
            wind_dir_deg=60.0,
            initial_speed=2.0,
            duration_s=20.0,
        )
        # Check that sway velocity is non-zero at some point
        max_sway = max(abs(s.state.v) for s in rec.steps)
        assert max_sway > 0.01, (
            f"Should have non-zero sway from crosswind: max |v| = {max_sway:.4f}"
        )


# ──────────────────────────────────────────────────────────────────────
# Test 5: Weather helm (wind-induced yaw moment)
# ──────────────────────────────────────────────────────────────────────


class TestWeatherHelm:
    """A sailing yacht should exhibit weather or lee helm tendency
    depending on the CE/CLR balance."""

    def test_wind_causes_yaw(self):
        """Wind should cause the boat to yaw (weather/lee helm)."""
        rec = _run_fixed_rudder(
            rudder_deg=0.0,
            wind_speed=8.0,
            wind_dir_deg=60.0,
            initial_speed=2.0,
            initial_heading_deg=0.0,
            duration_s=30.0,
        )
        initial_heading = np.degrees(rec.steps[0].state.psi)
        final_heading = np.degrees(rec.steps[-1].state.psi)
        heading_change = abs(final_heading - initial_heading)
        assert heading_change > 1.0, (
            f"Wind should cause yaw tendency: heading change = {heading_change:.2f}°"
        )


# ──────────────────────────────────────────────────────────────────────
# Test 6: Energy conservation / physical plausibility
# ──────────────────────────────────────────────────────────────────────


class TestPhysicalPlausibility:
    """Basic sanity checks for physical plausibility."""

    def test_no_infinite_speed(self):
        """Speed should not exceed reasonable limits even with strong wind."""
        rec = _run_fixed_rudder(
            rudder_deg=0.0,
            wind_speed=15.0,
            wind_dir_deg=90.0,
            initial_speed=0.0,
            duration_s=120.0,
        )
        max_speed = max(s.state.speed for s in rec.steps)
        assert max_speed < 15.0, (
            f"Speed should not exceed wind speed (hull speed limit): {max_speed:.2f} m/s"
        )

    def test_no_nan_values(self):
        """Simulation should never produce NaN values."""
        rec = _run_fixed_rudder(
            rudder_deg=15.0,
            wind_speed=10.0,
            wind_dir_deg=45.0,
            initial_speed=3.0,
            duration_s=60.0,
        )
        for step in rec.steps:
            assert not np.any(np.isnan(step.state.eta)), f"NaN in eta at t={step.t}"
            assert not np.any(np.isnan(step.state.nu)), f"NaN in nu at t={step.t}"

    def test_symmetry_port_starboard(self):
        """Port and starboard wind should produce mirrored behavior."""
        rec_stb = _run_fixed_rudder(
            rudder_deg=0.0,
            wind_speed=5.0,
            wind_dir_deg=60.0,
            initial_speed=2.0,
            duration_s=20.0,
        )
        rec_port = _run_fixed_rudder(
            rudder_deg=0.0,
            wind_speed=5.0,
            wind_dir_deg=-60.0,
            initial_speed=2.0,
            duration_s=20.0,
        )
        # Speeds should be similar
        speed_stb = rec_stb.steps[-1].state.speed
        speed_port = rec_port.steps[-1].state.speed
        assert abs(speed_stb - speed_port) < 0.5, (
            f"Port/starboard should be symmetric: stb={speed_stb:.2f}, port={speed_port:.2f}"
        )
        # Y positions should be mirrored
        y_stb = rec_stb.steps[-1].state.y
        y_port = rec_port.steps[-1].state.y
        assert abs(y_stb + y_port) < 2.0, (
            f"Drift should be mirrored: y_stb={y_stb:.2f}, y_port={y_port:.2f}"
        )
