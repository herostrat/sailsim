"""Hydrodynamic models for rudder and keel forces.

Provides force calculations for:
- Rudder: control surface generating lateral force and yaw moment
- Keel: lifting surface resisting leeway (drift)

References:
    Fossen (2021), Larsson & Eliasson (2014)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Water density [kg/m^3]
RHO_WATER = 1025.0


def rudder_forces_3dof(
    rudder_angle: float,
    u: float,
    v: float,
    r: float,
    rudder_area: float,
    rudder_x: float,
    cl_slope: float = 3.0,
    max_cl: float = 1.0,
    stall_angle: float = 0.35,  # ~20 degrees
) -> NDArray[np.float64]:
    """Compute rudder forces and yaw moment (3-DOF).

    The rudder operates in the flow behind the keel. Its effective angle of
    attack combines the commanded rudder angle with local flow angle.

    Args:
        rudder_angle: commanded rudder deflection [rad], positive = starboard
        u: surge velocity [m/s]
        v: sway velocity at rudder location [m/s]
        r: yaw rate [rad/s]
        rudder_area: rudder planform area [m^2]
        rudder_x: longitudinal position of rudder from CO [m] (negative = aft)
        cl_slope: lift curve slope [1/rad] (typically 2*pi ≈ 6.28 for thin foil,
                  ~3.0 for realistic aspect ratios)
        max_cl: maximum lift coefficient before stall
        stall_angle: angle of attack at which stall occurs [rad]

    Returns:
        3-element force vector [X_rudder, Y_rudder, N_rudder].
    """
    # Effective inflow velocity at rudder (reduced by wake fraction)
    u_eff = max(u * 0.9, 0.1)  # wake fraction ~0.1, avoid division by zero

    # Local sway velocity at rudder position (v + r * x_rudder)
    v_local = v + r * rudder_x

    # Effective angle of attack
    inflow_angle = np.arctan2(-v_local, u_eff)
    alpha = rudder_angle + inflow_angle

    # Lift coefficient with stall model
    if abs(alpha) < stall_angle:
        cl = cl_slope * alpha
        cl = np.clip(cl, -max_cl, max_cl)
    else:
        # Post-stall: reduced lift
        cl = max_cl * np.sign(alpha) * 0.6

    # Drag coefficient (profile drag + induced drag)
    cd = 0.01 + 0.1 * cl**2

    # Dynamic pressure
    V_eff = np.sqrt(u_eff**2 + v_local**2)
    q = 0.5 * RHO_WATER * V_eff**2

    # Forces in body frame
    # Convention: positive rudder angle (starboard) creates negative Y force
    # on the vessel (pushes stern to port, bow to starboard).
    # The lift coefficient sign follows alpha, so we negate to get vessel force.
    Y_rudder = -q * rudder_area * cl
    X_rudder = -q * rudder_area * cd  # drag opposes motion

    # Yaw moment (rudder force * lever arm from CO)
    N_rudder = Y_rudder * rudder_x

    return np.array([X_rudder, Y_rudder, N_rudder])


def keel_forces_3dof(
    u: float,
    v: float,
    keel_area: float,
    keel_x: float,
    cl_slope: float = 4.0,
    max_cl: float = 1.2,
    stall_angle: float = 0.25,  # ~14 degrees
) -> NDArray[np.float64]:
    """Compute keel lateral force and yaw moment (3-DOF).

    The keel acts as a fixed lifting surface. Its angle of attack is the
    leeway angle (angle between boat heading and velocity vector).

    Args:
        u: surge velocity [m/s]
        v: sway velocity [m/s]
        keel_area: keel lateral planform area [m^2]
        keel_x: longitudinal position of keel's center of lateral resistance [m]
        cl_slope: lift curve slope [1/rad]
        max_cl: maximum lift coefficient
        stall_angle: stall angle [rad]

    Returns:
        3-element force vector [X_keel, Y_keel, N_keel].
    """
    # Leeway angle
    speed = max(np.sqrt(u**2 + v**2), 0.1)
    leeway = np.arctan2(-v, max(u, 0.1))

    # Lift coefficient
    if abs(leeway) < stall_angle:
        cl = cl_slope * leeway
        cl = np.clip(cl, -max_cl, max_cl)
    else:
        cl = max_cl * np.sign(leeway) * 0.5

    # Drag (induced + profile)
    cd = 0.005 + 0.08 * cl**2

    q = 0.5 * RHO_WATER * speed**2

    Y_keel = q * keel_area * cl
    X_keel = -q * keel_area * cd
    N_keel = Y_keel * keel_x

    return np.array([X_keel, Y_keel, N_keel])


def rudder_forces_6dof(
    rudder_angle: float,
    u: float,
    v: float,
    r: float,
    rudder_area: float,
    rudder_x: float,
    rudder_z: float = 0.8,
) -> NDArray[np.float64]:
    """Compute rudder forces and moments (6-DOF).

    Extends 3-DOF with roll moment from rudder force acting below waterline.

    Args:
        rudder_angle: commanded rudder deflection [rad]
        u, v, r: velocities [m/s, m/s, rad/s]
        rudder_area: planform area [m^2]
        rudder_x: longitudinal position [m]
        rudder_z: depth of rudder center of effort below WL [m]

    Returns:
        6-element force vector [X, Y, Z, K, M, N].
    """
    forces_3 = rudder_forces_3dof(rudder_angle, u, v, r, rudder_area, rudder_x)
    X_r, Y_r, N_r = forces_3

    # Roll moment: rudder lateral force acts below WL, creating roll
    K_r = -Y_r * rudder_z  # negative sign: force below WL creates opposite roll

    return np.array([X_r, Y_r, 0.0, K_r, 0.0, N_r])


def keel_forces_6dof(
    u: float,
    v: float,
    keel_area: float,
    keel_x: float,
    keel_z: float = 1.5,
) -> NDArray[np.float64]:
    """Compute keel forces and moments (6-DOF).

    Extends 3-DOF with righting moment from keel acting deep below WL.

    Args:
        u, v: velocities [m/s]
        keel_area: lateral area [m^2]
        keel_x: longitudinal CLR position [m]
        keel_z: depth of keel CLR below WL [m]

    Returns:
        6-element force vector [X, Y, Z, K, M, N].
    """
    forces_3 = keel_forces_3dof(u, v, keel_area, keel_x)
    X_k, Y_k, N_k = forces_3

    # Roll moment: keel lateral force acts deep below WL → righting
    K_k = -Y_k * keel_z

    return np.array([X_k, Y_k, 0.0, K_k, 0.0, N_k])
