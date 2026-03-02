"""Tests for scenario and yacht profile loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from sailsim.core.config import ScenarioConfig, YachtConfig, load_scenario

SCENARIOS_DIR = Path(__file__).resolve().parents[2] / "configs" / "scenarios"
YACHTS_DIR = Path(__file__).resolve().parents[2] / "configs" / "yachts"


def test_default_profile_loads():
    """Loading a scenario with profile='default' reads the yacht TOML."""
    config = load_scenario(SCENARIOS_DIR / "calm_heading_hold.toml")
    assert config.yacht.mass == 4000.0
    assert config.yacht.sail_area == 50.0


def test_profile_with_dof_override():
    """Scenario can set profile and override dof in the same [yacht] section."""
    config = load_scenario(SCENARIOS_DIR / "rough_6dof.toml")
    assert config.yacht.dof == 6
    assert config.yacht.mass == 4000.0  # from profile
    assert config.yacht.GM_T == 1.2  # 6-DOF param from profile


def test_no_profile_uses_defaults():
    """ScenarioConfig constructed without profile uses hardcoded defaults."""
    config = ScenarioConfig()
    assert config.yacht.profile is None
    assert config.yacht.mass == 4000.0
    assert config.yacht.dof == 3


def test_missing_profile_raises(tmp_path):
    """A nonexistent profile should raise FileNotFoundError."""
    scenario = tmp_path / "bad.toml"
    scenario.write_text('name = "bad"\n[yacht]\nprofile = "nonexistent"\n')
    # The yacht dir won't exist relative to tmp_path
    with pytest.raises(FileNotFoundError):
        load_scenario(scenario)


def test_yacht_config_has_6dof_fields():
    """YachtConfig should expose all 6-DOF parameters."""
    cfg = YachtConfig()
    assert cfg.Ix == 8000.0
    assert cfg.GM_T == 1.2
    assert cfg.sail_ce_z == 5.0
    assert cfg.keel_z == 1.5


def test_profile_values_match_toml():
    """Profile TOML values should match YachtConfig defaults."""
    import tomllib

    with (YACHTS_DIR / "default.toml").open("rb") as f:
        profile = tomllib.load(f)

    cfg = YachtConfig()
    for key, value in profile.items():
        assert getattr(cfg, key) == value, f"{key}: {getattr(cfg, key)} != {value}"
