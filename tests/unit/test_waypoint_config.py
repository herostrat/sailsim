"""Tests for waypoint configuration loading."""

from __future__ import annotations

from pathlib import Path

from sailsim.core.config import RouteConfig, ScenarioConfig, WaypointConfig, load_scenario

SCENARIOS_DIR = Path(__file__).resolve().parents[2] / "configs" / "scenarios"


def test_empty_route_by_default():
    """ScenarioConfig should have empty route by default."""
    config = ScenarioConfig()
    assert config.route.waypoints == []


def test_route_config_from_dict():
    """RouteConfig should parse waypoints from a dictionary."""
    route = RouteConfig(
        waypoints=[
            WaypointConfig(x=100.0, y=0.0),
            WaypointConfig(x=100.0, y=100.0, tolerance=20.0),
        ]
    )
    assert len(route.waypoints) == 2
    assert route.waypoints[0].tolerance == 10.0  # default
    assert route.waypoints[1].tolerance == 20.0


def test_existing_scenarios_still_load():
    """All heading-based scenarios (without route section) should still load."""
    for toml_file in sorted(SCENARIOS_DIR.glob("*.toml")):
        config = load_scenario(toml_file)
        if "waypoint" in toml_file.name:
            assert len(config.route.waypoints) > 0
        else:
            assert config.route.waypoints == [], f"{toml_file.name} has non-empty route"
