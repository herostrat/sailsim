"""Autopilot factory — create autopilot instances from config."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sailsim.autopilot.base import AutopilotProtocol
    from sailsim.core.config import AutopilotConfig, YachtConfig


def create_autopilot(
    config: AutopilotConfig,
    yacht: YachtConfig | None = None,
) -> AutopilotProtocol:
    """Create an autopilot instance based on *config.type*.

    Supported types:
    - ``"nomoto"``: Nomoto model with pole placement (requires *yacht*)
    - ``"signalk"``: adapter that communicates via a SignalK server
    - ``"signalk_rs"``: adapter for signalk-rs (Rust-based SignalK with PID autopilot)
    - ``"pypilot"``: adapter that communicates via pypilot JSON-TCP
    """
    if config.type == "nomoto":
        if yacht is None:
            raise ValueError("Nomoto autopilot requires yacht configuration")
        from sailsim.autopilot.nomoto import NomotoAutopilot

        return NomotoAutopilot(
            yacht=yacht,
            omega_n=config.omega_n,
            zeta=config.zeta,
            rudder_rate_max=np.radians(config.rudder_rate_max_deg_s),
            U_min=config.U_min,
            auto_sail_trim=config.auto_sail_trim,
        )
    elif config.type == "signalk":
        from sailsim.autopilot.signalk import SignalKAutopilot

        return SignalKAutopilot(url=config.signalk_url)
    elif config.type == "pypilot":
        from sailsim.autopilot.pypilot import PypilotAutopilot

        return PypilotAutopilot(
            host=config.pypilot_host,
            json_port=config.pypilot_json_port,
            nmea_port=config.pypilot_nmea_port,
            rudder_max=np.radians(config.pypilot_rudder_max_deg),
            mode=config.pypilot_mode,
            sim_sleep_ms=config.pypilot_sim_sleep_ms,
        )
    elif config.type == "signalk_rs":
        from sailsim.autopilot.signalk_rs import SignalKRsAutopilot

        return SignalKRsAutopilot(
            host=config.signalk_rs_host,
            http_port=config.signalk_rs_http_port,
            nmea_port=config.signalk_rs_nmea_port,
            device_id=config.signalk_rs_device_id,
            rudder_max=np.radians(config.signalk_rs_rudder_max_deg),
            mode=config.signalk_rs_mode,
            sim_sleep_ms=config.signalk_rs_sim_sleep_ms,
        )
    else:
        raise ValueError(f"Unknown autopilot type: {config.type!r}")
