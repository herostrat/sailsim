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
    else:
        raise ValueError(f"Unknown autopilot type: {config.type!r}")
