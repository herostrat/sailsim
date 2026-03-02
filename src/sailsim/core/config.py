"""TOML configuration loading and validation with Pydantic."""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel


class YachtConfig(BaseModel):
    """Yacht physical parameters (maps to YachtParams / YachtParams6DOF).

    Use ``profile`` to load parameters from a TOML file in configs/yachts/.
    Individual fields set in the scenario override the profile values.
    """

    profile: str | None = None  # e.g. "default" → configs/yachts/default.toml
    dof: int = 3  # 3 or 6

    # Mass properties
    mass: float = 4000.0
    Iz: float = 25000.0
    xg: float = 0.0

    # Added mass
    X_udot: float = -200.0
    Y_vdot: float = -4000.0
    Y_rdot: float = -400.0
    N_vdot: float = -400.0
    N_rdot: float = -15000.0

    # Linear damping
    Xu: float = -100.0
    Yv: float = -2000.0
    Yr: float = -500.0
    Nv: float = -500.0
    Nr: float = -20000.0

    # Quadratic damping
    Xuu: float = -50.0
    Yvv: float = -3000.0
    Yrr: float = -500.0
    Nvv: float = -500.0
    Nrr: float = -50000.0

    # Sail
    sail_area: float = 50.0
    mast_height: float = 12.0
    sail_ce_x: float = 0.3

    # Rudder
    rudder_area: float = 0.25
    rudder_x: float = -4.5
    rudder_max: float = 0.52

    # Keel
    keel_area: float = 1.5
    keel_x: float = -0.3

    # --- 6-DOF extensions (ignored for 3-DOF) ---
    Ix: float = 8000.0
    Iy: float = 30000.0
    zg: float = 0.2

    Z_wdot: float = -4000.0
    K_pdot: float = -500.0
    M_qdot: float = -20000.0

    Zw: float = -3000.0
    Kp: float = -5000.0
    Mq: float = -30000.0

    Zww: float = -3000.0
    Kpp: float = -3000.0
    Mqq: float = -20000.0

    sail_ce_z: float = 5.0
    rudder_z: float = 0.8
    keel_z: float = 1.5

    GM_T: float = 1.2
    GM_L: float = 15.0
    Aw: float = 22.0


class WindConfig(BaseModel):
    """Wind conditions."""

    speed: float = 5.0
    direction: float = 1.047  # ~60° (close reach)
    model: str = "constant"  # "constant" | "gust" | "shifting"
    # Gust parameters (model="gust")
    gust_intensity: float = 2.0
    gust_tau: float = 10.0
    gust_seed: int | None = None
    # Shifting parameters (model="shifting")
    shift_mode: str = "sinusoidal"  # "linear" | "sinusoidal"
    shift_rate: float = 0.0  # [rad/s] for linear mode
    shift_amplitude: float = 0.0  # [rad] for sinusoidal mode
    shift_period: float = 300.0  # [s] for sinusoidal mode


class AutopilotConfig(BaseModel):
    """PID autopilot parameters."""

    kp: float = 1.0
    ki: float = 0.05
    kd: float = 0.8
    target_heading: float = 0.0  # [rad]
    auto_sail_trim: bool = False


class InitialStateConfig(BaseModel):
    """Initial vessel state."""

    x: float = 0.0
    y: float = 0.0
    psi: float = 0.0  # heading [rad]
    u: float = 0.0  # initial surge velocity [m/s]


class QualityGateConfig(BaseModel):
    """Pass/fail criteria for scenario tests."""

    max_heading_deviation_deg: float = 5.0
    max_rudder_rate_deg_per_s: float = 15.0
    max_settling_time_s: float = 30.0
    max_mean_heading_error_deg: float = 3.0


class CurrentConfig(BaseModel):
    """Ocean current conditions."""

    model: str = "none"  # "none" | "constant" | "tidal"
    speed: float = 0.0
    direction: float = 0.0  # direction current flows TOWARDS [rad]
    tidal_amplitude: float = 0.5
    tidal_period: float = 21600.0  # 6 hours
    tidal_phase: float = 0.0


class WaveConfig(BaseModel):
    """Wave/sea state conditions."""

    model: str = "none"  # "none" | "spectral"
    Hs: float = 0.0  # significant wave height [m]
    Tp: float = 6.0  # peak period [s]
    direction: float = 0.0  # propagation direction [rad]
    spectrum: str = "jonswap"  # "jonswap" | "pm"
    gamma: float = 3.3  # JONSWAP peak enhancement
    n_components: int = 50
    seed: int | None = None


class ManeuverStep(BaseModel):
    """A single scheduled heading change."""

    time_s: float
    target_heading: float  # [rad]


class ManeuverConfig(BaseModel):
    """Scheduled course changes during simulation."""

    steps: list[ManeuverStep] = []


class WaypointConfig(BaseModel):
    """A single waypoint in a route."""

    x: float
    y: float
    tolerance: float = 10.0


class RouteConfig(BaseModel):
    """Ordered list of waypoints defining a route."""

    waypoints: list[WaypointConfig] = []


class ScenarioConfig(BaseModel):
    """Complete scenario configuration."""

    name: str = "unnamed"
    duration_s: float = 120.0
    dt: float = 0.05  # 20 Hz
    yacht: YachtConfig = YachtConfig()
    wind: WindConfig = WindConfig()
    current: CurrentConfig = CurrentConfig()
    waves: WaveConfig = WaveConfig()
    autopilot: AutopilotConfig = AutopilotConfig()
    initial_state: InitialStateConfig = InitialStateConfig()
    maneuvers: ManeuverConfig = ManeuverConfig()
    route: RouteConfig = RouteConfig()
    quality_gates: QualityGateConfig = QualityGateConfig()


def load_scenario(path: str | Path) -> ScenarioConfig:
    """Load and validate a scenario from a TOML file.

    If the ``[yacht]`` section contains a ``profile`` key, the corresponding
    TOML file is loaded from ``configs/yachts/<profile>.toml`` (resolved
    relative to the scenario file) and used as the base.  Any yacht fields
    set directly in the scenario override the profile values.
    """
    path = Path(path)
    with path.open("rb") as f:
        data = tomllib.load(f)

    # Resolve yacht profile
    yacht_data = data.get("yacht", {})
    profile = yacht_data.get("profile")
    if profile:
        yacht_dir = path.parent.parent / "yachts"
        yacht_file = yacht_dir / f"{profile}.toml"
        with yacht_file.open("rb") as f:
            profile_data = tomllib.load(f)
        # Scenario-specific overrides on top of profile defaults
        profile_data.update(yacht_data)
        profile_data.pop("profile", None)
        data["yacht"] = profile_data

    return ScenarioConfig(**data)
