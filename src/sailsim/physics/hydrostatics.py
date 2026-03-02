"""Hydrostatic restoring forces for 6-DOF.

Provides buoyancy and righting forces/moments for heave, roll, and pitch.
Uses linear GZ (metacentric height) assumption valid for small angles.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

G = 9.81
RHO_WATER = 1025.0


def restoring_forces_6dof(
    z: float,
    phi: float,
    theta: float,
    mass: float,
    GM_T: float,
    GM_L: float,
    Aw: float,
) -> NDArray[np.float64]:
    """Compute Fossen's hydrostatic g(eta) vector.

    Returns g(eta) as defined in the Fossen equation:
        M*nu_dot + C*nu + D*nu + g(eta) = tau

    Positive values for positive displacement (i.e., this is NOT the
    physical restoring force direction — it's the "load" term).

    Args:
        z: heave displacement [m] (positive down = submerged more)
        phi: roll angle [rad]
        theta: pitch angle [rad]
        mass: vessel displacement mass [kg]
        GM_T: transverse metacentric height [m]
        GM_L: longitudinal metacentric height [m]
        Aw: waterplane area [m^2]

    Returns:
        6-element g(eta) vector [X, Y, Z, K, M, N].
    """
    W = mass * G

    # Heave: positive z (deeper) → positive g_Z → subtracted in EOM → pushes up
    g_Z = RHO_WATER * G * Aw * z

    # Roll: positive phi (starboard) → positive g_K → subtracted in EOM → restores port
    g_K = W * GM_T * np.sin(phi)

    # Pitch: positive theta (bow up) → positive g_M → subtracted in EOM → restores bow down
    g_M = W * GM_L * np.sin(theta)

    return np.array([0.0, 0.0, g_Z, g_K, g_M, 0.0])
