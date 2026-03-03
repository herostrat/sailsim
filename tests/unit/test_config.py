"""Tests for scenario, yacht, and autopilot configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from sailsim.core.config import (
    AutopilotConfig,
    ScenarioConfig,
    YachtConfig,
    load_autopilot,
    load_scenario,
    load_yacht,
)

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"
SCENARIOS_DIR = CONFIGS_DIR / "scenarios"
YACHTS_DIR = CONFIGS_DIR / "yachts"
AUTOPILOTS_DIR = CONFIGS_DIR / "autopilots"


# --- Scenario loading ---


def test_scenario_loads_without_yacht_or_autopilot():
    """load_scenario returns ScenarioConfig with default yacht (no [yacht] section)."""
    config = load_scenario(SCENARIOS_DIR / "calm_heading_hold.toml")
    assert config.name == "calm_heading_hold"
    assert config.target_heading == 0.0
    # yacht should be default since scenario has no [yacht] section
    assert config.yacht.mass == 4000.0


def test_scenario_strips_yacht_and_autopilot_sections(tmp_path):
    """Scenario TOML with [yacht] or [autopilot] sections: they are ignored."""
    scenario = tmp_path / "test.toml"
    scenario.write_text(
        'name = "test"\n'
        "target_heading = 1.0\n"
        "[yacht]\n"
        "mass = 9999.0\n"
        "[autopilot]\n"
        "kp = 9.9\n"
    )
    config = load_scenario(scenario)
    assert config.target_heading == 1.0
    # [yacht] and [autopilot] should be stripped → defaults
    assert config.yacht.mass == 4000.0  # default, not 9999


def test_scenario_target_heading_at_top_level():
    """target_heading is a top-level scenario field."""
    config = ScenarioConfig(target_heading=1.5)
    assert config.target_heading == 1.5


def test_no_profile_uses_defaults():
    """ScenarioConfig constructed without arguments uses hardcoded defaults."""
    config = ScenarioConfig()
    assert config.yacht.mass == 4000.0
    assert config.yacht.dof == 3
    assert config.target_heading == 0.0


def test_all_scenarios_load():
    """All scenario TOMLs should load without error."""
    for toml_file in sorted(SCENARIOS_DIR.glob("*.toml")):
        config = load_scenario(toml_file)
        assert config.name, f"{toml_file.name} has no name"


# --- Yacht loading ---


def test_load_yacht_by_name():
    """load_yacht('default') loads configs/yachts/default.toml."""
    cfg = load_yacht("default", configs_root=CONFIGS_DIR)
    assert cfg.mass == 4000.0
    assert cfg.sail_area == 50.0


def test_load_yacht_j24():
    """load_yacht('j24') loads the J/24 profile."""
    cfg = load_yacht("j24", configs_root=CONFIGS_DIR)
    assert isinstance(cfg, YachtConfig)
    assert cfg.mass != 4000.0  # J/24 has different mass


def test_load_yacht_by_toml_path():
    """load_yacht with a .toml path loads that file."""
    cfg = load_yacht(str(YACHTS_DIR / "default.toml"))
    assert cfg.mass == 4000.0


def test_load_yacht_missing_raises():
    """Nonexistent yacht profile raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_yacht("nonexistent", configs_root=CONFIGS_DIR)


def test_yacht_config_has_6dof_fields():
    """YachtConfig should expose all 6-DOF parameters."""
    cfg = YachtConfig()
    assert cfg.Ix == 8000.0
    assert cfg.GM_T == 1.2
    assert cfg.sail_ce_z == 5.0
    assert cfg.keel_z == 1.5


def test_rudder_cp_offset_default():
    """YachtConfig should have rudder_cp_offset with default 0.08."""
    cfg = YachtConfig()
    assert cfg.rudder_cp_offset == 0.08


def test_yacht_profile_values_match_toml():
    """Profile TOML values should match what load_yacht returns."""
    import tomllib

    with (YACHTS_DIR / "default.toml").open("rb") as f:
        profile = tomllib.load(f)

    cfg = load_yacht("default", configs_root=CONFIGS_DIR)
    for key, value in profile.items():
        assert getattr(cfg, key) == value, f"{key}: {getattr(cfg, key)} != {value}"


# --- Autopilot loading ---


def test_load_autopilot_heading_hold():
    """load_autopilot('heading_hold') loads the profile."""
    cfg = load_autopilot("heading_hold", configs_root=CONFIGS_DIR)
    assert cfg.type == "nomoto"
    assert cfg.omega_n == 0.5
    assert cfg.zeta == 0.8
    assert cfg.auto_sail_trim is True


def test_load_autopilot_tack():
    """load_autopilot('tack') loads responsive maneuver gains."""
    cfg = load_autopilot("tack", configs_root=CONFIGS_DIR)
    assert cfg.type == "nomoto"
    assert cfg.omega_n == 0.6
    assert cfg.rudder_rate_max_deg_s == 10.0


def test_load_autopilot_signalk():
    """load_autopilot('signalk') loads SignalK config."""
    cfg = load_autopilot("signalk", configs_root=CONFIGS_DIR)
    assert cfg.type == "signalk"
    assert cfg.signalk_url == "http://localhost:3000"


def test_load_autopilot_by_toml_path(tmp_path):
    """load_autopilot with a .toml path loads that file."""
    ap_file = tmp_path / "custom.toml"
    ap_file.write_text('type = "nomoto"\nomega_n = 0.9\n')
    cfg = load_autopilot(str(ap_file))
    assert cfg.omega_n == 0.9


def test_load_autopilot_missing_raises():
    """Nonexistent autopilot profile raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_autopilot("nonexistent", configs_root=CONFIGS_DIR)


def test_autopilot_config_defaults():
    """AutopilotConfig defaults are sensible."""
    cfg = AutopilotConfig()
    assert cfg.type == "nomoto"
    assert cfg.signalk_url == "http://localhost:3000"
    assert cfg.auto_sail_trim is False


def test_autopilot_config_no_target_heading():
    """AutopilotConfig should NOT have target_heading (it's on ScenarioConfig)."""
    assert not hasattr(AutopilotConfig.model_fields, "target_heading")
    cfg = AutopilotConfig()
    assert not hasattr(cfg, "target_heading") or "target_heading" not in cfg.model_fields


def test_yacht_config_no_profile():
    """YachtConfig should NOT have a profile field."""
    assert "profile" not in YachtConfig.model_fields
