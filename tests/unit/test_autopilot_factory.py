"""Tests for the autopilot factory."""

from __future__ import annotations

import pytest

from sailsim.autopilot.factory import create_autopilot
from sailsim.autopilot.nomoto import NomotoAutopilot
from sailsim.autopilot.signalk import SignalKAutopilot
from sailsim.core.config import AutopilotConfig, YachtConfig


def test_create_signalk_autopilot():
    """Factory creates SignalKAutopilot for type='signalk'."""
    config = AutopilotConfig(type="signalk", signalk_url="http://myserver:3000")
    ap = create_autopilot(config)
    assert isinstance(ap, SignalKAutopilot)
    assert ap.base_url == "http://myserver:3000"


def test_create_unknown_type_raises():
    """Factory raises ValueError for unknown autopilot type."""
    config = AutopilotConfig(type="neural_net")
    with pytest.raises(ValueError, match="Unknown autopilot type"):
        create_autopilot(config)


def test_create_nomoto_default_config():
    """Factory with default AutopilotConfig creates NomotoAutopilot."""
    config = AutopilotConfig()
    yacht = YachtConfig()
    ap = create_autopilot(config, yacht=yacht)
    assert isinstance(ap, NomotoAutopilot)


def test_factory_creates_nomoto():
    """Factory creates NomotoAutopilot for type='nomoto' with yacht."""
    config = AutopilotConfig(type="nomoto", omega_n=0.5, zeta=0.9)
    yacht = YachtConfig()
    ap = create_autopilot(config, yacht=yacht)
    assert isinstance(ap, NomotoAutopilot)
    assert ap.omega_n == 0.5
    assert ap.zeta == 0.9


def test_factory_nomoto_requires_yacht():
    """Factory raises ValueError for type='nomoto' without yacht."""
    config = AutopilotConfig(type="nomoto")
    with pytest.raises(ValueError, match="Nomoto autopilot requires yacht"):
        create_autopilot(config)
