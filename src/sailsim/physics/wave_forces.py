"""Wave force model for 3-DOF.

Simplified wave forces consisting of:
1. Added resistance (ITTC-type) — depends on Hs and encounter angle
2. First-order excitation (Froude-Krylov approximation) — lateral force and yaw from elevation

References:
    ITTC Recommended Procedures 7.5-02-07-02.1 (added resistance)
    Faltinsen, O.M. (1990) Sea Loads on Ships and Offshore Structures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from sailsim.core.types import WaveState

# Water density and gravity
RHO_WATER = 1025.0
G = 9.81


def wave_forces_3dof(
    wave: WaveState,
    psi: float,
    u: float,
    beam: float = 3.5,
    length: float = 10.0,
    draft: float = 1.8,
) -> NDArray[np.float64]:
    """Compute wave forces and yaw moment in body frame (3-DOF).

    Args:
        wave: current wave state at vessel
        psi: vessel heading [rad]
        u: surge velocity [m/s]
        beam: vessel beam [m]
        length: vessel length [m]
        draft: vessel draft [m]

    Returns:
        3-element force vector [X_wave, Y_wave, N_wave] in body frame.
    """
    if wave.Hs <= 0:
        return np.zeros(3)

    # Encounter angle: angle between wave direction and vessel heading
    # beta = 0: following seas, beta = pi: head seas
    beta = wave.direction - psi
    beta = np.arctan2(np.sin(beta), np.cos(beta))

    # 1. Added resistance (always opposes forward motion)
    # R_aw = 1/16 * rho * g * Hs^2 * B * sigma(beta)
    # sigma(beta) peaks for head seas (beta=pi), zero for following seas
    sigma = 0.5 * (1.0 - np.cos(beta))  # 0 for following, 1 for head seas
    R_aw = -1.0 / 16.0 * RHO_WATER * G * wave.Hs**2 * beam * sigma

    # 2. First-order excitation (from wave elevation)
    # Lateral force proportional to elevation x lateral area x sin(beta)
    lateral_area = length * draft
    F_lateral = RHO_WATER * G * wave.elevation * lateral_area * np.sin(beta) * 0.1

    # Yaw moment from lateral force acting at offset from midship
    N_wave = F_lateral * length * 0.1  # small moment arm fraction

    X_wave = float(R_aw)
    Y_wave = float(F_lateral)
    N_wave = float(N_wave)

    return np.array([X_wave, Y_wave, N_wave])


def wave_forces_6dof(
    wave: WaveState,
    psi: float,
    u: float,
    beam: float = 3.5,
    length: float = 10.0,
    draft: float = 1.8,
) -> NDArray[np.float64]:
    """Compute wave forces and moments in body frame (6-DOF).

    Extends 3-DOF with roll excitation moment from beam seas.

    Returns:
        6-element force vector [X, Y, Z, K, M, N] in body frame.
    """
    if wave.Hs <= 0:
        return np.zeros(6)

    forces_3 = wave_forces_3dof(wave, psi, u, beam, length, draft)
    X_w, Y_w, N_w = forces_3

    # Encounter angle
    beta = wave.direction - psi
    beta = np.arctan2(np.sin(beta), np.cos(beta))

    # Roll excitation from wave slope (beam seas create max roll)
    # Simplified: proportional to elevation * beam * sin(beta)
    K_w = RHO_WATER * G * wave.elevation * beam * draft * np.sin(beta) * 0.05

    # Pitch excitation from wave slope (head/following seas)
    M_w = RHO_WATER * G * wave.elevation * length * draft * np.cos(beta) * 0.02

    return np.array([X_w, Y_w, 0.0, float(K_w), float(M_w), N_w])
