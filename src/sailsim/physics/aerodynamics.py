"""Sail aerodynamics model.

Computes driving force and side force from sails based on apparent wind.
Uses lift/drag coefficient curves typical for a mainsail + genoa combination.

Convention: Apparent Wind Angle (AWA) is measured as the angle the wind is
COMING FROM, relative to the bow. Positive = starboard, negative = port.
Range: [-pi, pi]. 0 = head-to-wind, +/-pi = dead run.

References:
    Larsson, L. & Eliasson, R. (2014) Principles of Yacht Design.
    ORC VPP aerodynamic model (simplified).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Air density at sea level [kg/m^3]
RHO_AIR = 1.225


def apparent_wind(
    true_wind_speed: float,
    true_wind_direction: float,
    u: float,
    v: float,
    psi: float,
) -> tuple[float, float]:
    """Compute apparent wind speed and angle (AWA).

    Args:
        true_wind_speed: true wind speed [m/s]
        true_wind_direction: direction wind comes FROM in NED [rad]
        u: surge velocity [m/s]
        v: sway velocity [m/s]
        psi: heading [rad]

    Returns:
        (apparent_wind_speed [m/s], AWA [rad])
        AWA = angle wind comes FROM relative to bow.
        Positive = starboard, negative = port.
    """
    # True wind velocity in NED ("comes from" -> velocity goes opposite)
    tw_n = -true_wind_speed * np.cos(true_wind_direction)
    tw_e = -true_wind_speed * np.sin(true_wind_direction)

    # Transform to body frame (x forward, y starboard)
    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)
    tw_x = cos_psi * tw_n + sin_psi * tw_e
    tw_y = -sin_psi * tw_n + cos_psi * tw_e

    # Apparent wind velocity = true wind velocity - boat velocity
    aw_x = tw_x - u
    aw_y = tw_y - v

    aw_speed = float(np.sqrt(aw_x**2 + aw_y**2))

    # AWA = direction wind comes FROM = opposite of velocity vector
    awa = float(np.arctan2(-aw_y, -aw_x))

    return aw_speed, awa


def optimal_sail_trim(awa: float) -> float:
    """Compute optimal sail trim for a given apparent wind angle.

    Args:
        awa: apparent wind angle [rad], where wind comes FROM relative to bow.

    Returns:
        Optimal sail_trim in [0.0, 1.0].
        1.0 = fully sheeted (close-hauled), 0.0 = fully eased (running).
    """
    awa_deg = abs(np.degrees(awa))
    return float(np.clip(1.0 - (awa_deg - 30.0) / 150.0, 0.0, 1.0))


def sail_coefficients(awa: float, sail_trim: float = 0.5) -> tuple[float, float]:
    """Lookup lift and drag coefficients for combined sail plan.

    Simplified model based on typical mainsail + genoa polar data.
    Sail trim affects efficiency: coefficients are reduced when trim
    deviates from the optimum for the current apparent wind angle.

    Args:
        awa: apparent wind angle (where wind comes from) [rad]
        sail_trim: sheet tension [0..1], 0=eased, 1=sheeted. Default 0.5.

    Returns:
        (CL, CD) lift and drag coefficients.
    """
    awa_deg = abs(np.degrees(awa))

    if awa_deg < 20.0:
        cl = 0.0
        cd = 0.05
    elif awa_deg < 30.0:
        t = (awa_deg - 20.0) / 10.0
        cl = t * 1.2
        cd = 0.05 + t * 0.05
    elif awa_deg < 50.0:
        t = (awa_deg - 30.0) / 20.0
        cl = 1.2 + t * 0.2
        cd = 0.10 + t * 0.05
    elif awa_deg < 90.0:
        t = (awa_deg - 50.0) / 40.0
        cl = 1.4 * (1.0 - 0.5 * t)
        cd = 0.15 + t * 0.20
    elif awa_deg < 150.0:
        t = (awa_deg - 90.0) / 60.0
        cl = 0.7 * (1.0 - t)
        cd = 0.35 + t * 0.35
    else:
        t = min((awa_deg - 150.0) / 30.0, 1.0)
        cl = 0.0
        cd = 0.70 + t * 0.30

    # Apply sail trim efficiency.
    # The base coefficient curve models a "generic" trim.  The sail_trim
    # control allows fine-tuning.  A deadzone of 0.5 ensures the
    # default sail_trim=0.5 never degrades performance.  Penalties only
    # kick in for deliberately wrong trim (e.g., fully sheeted while running).
    trim_opt = optimal_sail_trim(awa)
    trim_error = abs(sail_trim - trim_opt)
    penalty = max(0.0, trim_error - 0.5)
    efficiency = 1.0 - 4.0 * penalty**2
    cl *= efficiency

    # Overtrimmed penalty: sheeted too tight for the AWA increases drag
    overtrim = sail_trim - trim_opt - 0.5
    if overtrim > 0:
        cd += 0.2 * overtrim

    return cl, cd


def sail_forces_3dof(
    aw_speed: float,
    awa: float,
    sail_area: float,
    sail_ce_x: float,
    sail_trim: float = 0.5,
) -> NDArray[np.float64]:
    """Compute sail forces and yaw moment in body frame (3-DOF).

    Uses standard yacht design decomposition:
        X_sail (driving force) = L * sin(|AWA|) - D * cos(|AWA|)
        Y_sail (heeling force) = -(L * cos(|AWA|) + D * sin(|AWA|)) * sign(AWA)

    The heeling force pushes to leeward (opposite side from wind).

    Args:
        aw_speed: apparent wind speed [m/s]
        awa: apparent wind angle (from direction) [rad], positive = starboard
        sail_area: total sail area [m^2]
        sail_ce_x: center of effort x-position from CO [m]
        sail_trim: sheet tension [0..1], 0=eased, 1=sheeted. Default 0.5.

    Returns:
        3-element force vector [X_sail, Y_sail, N_sail] in body frame.
    """
    cl, cd = sail_coefficients(awa, sail_trim)

    q = 0.5 * RHO_AIR * aw_speed**2

    lift = q * sail_area * cl
    drag = q * sail_area * cd

    awa_abs = abs(awa)
    cos_a = np.cos(awa_abs)
    sin_a = np.sin(awa_abs)

    # Driving force (forward along boat axis)
    X_sail = lift * sin_a - drag * cos_a

    # Heeling force (leeward = opposite side from wind)
    # Wind from starboard (awa > 0) -> heeling to port (negative Y)
    # Wind from port (awa < 0) -> heeling to starboard (positive Y)
    heeling = lift * cos_a + drag * sin_a
    if awa >= 0:
        Y_sail = -heeling  # push to port
    else:
        Y_sail = heeling  # push to starboard

    # Yaw moment: sail side force * lever arm.
    # Weather helm: sail_ce_x < 0 (CE aft) → N_sail opposes leeward drift.
    N_sail = Y_sail * sail_ce_x

    return np.array([X_sail, Y_sail, N_sail])


def sail_forces_6dof(
    aw_speed: float,
    awa: float,
    sail_area: float,
    sail_ce_x: float,
    sail_ce_z: float,
    sail_trim: float = 0.5,
) -> NDArray[np.float64]:
    """Compute sail forces and moments in body frame (6-DOF).

    Extends the 3-DOF model with heeling moment (K) and pitch moment (M).

    Args:
        aw_speed: apparent wind speed [m/s]
        awa: apparent wind angle [rad]
        sail_area: total sail area [m^2]
        sail_ce_x: CE longitudinal offset from CO [m]
        sail_ce_z: CE height above waterline [m]
        sail_trim: sheet tension [0..1]

    Returns:
        6-element force vector [X, Y, Z, K, M, N] in body frame.
    """
    forces_3 = sail_forces_3dof(aw_speed, awa, sail_area, sail_ce_x, sail_trim)
    X_sail, Y_sail, N_sail = forces_3

    # Heeling moment: sail side force x height above WL
    K_sail = Y_sail * sail_ce_z

    # Pitching moment: driving force pushes bow down
    M_sail = -X_sail * sail_ce_z * 0.3  # reduced lever arm for pitch

    return np.array([X_sail, Y_sail, 0.0, K_sail, M_sail, N_sail])
