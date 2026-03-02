"""Fossen rigid-body equations of motion for marine vessels.

Implements 3-DOF (surge, sway, yaw) and 6-DOF (surge, sway, heave, roll, pitch, yaw):
    M * nu_dot + C(nu) * nu + D(nu) * nu + g(eta) = tau

References:
    Fossen, T.I. (2021) Handbook of Marine Craft Hydrodynamics and Motion Control.
    DOF indices: [surge(u), sway(v), heave(w), roll(p), pitch(q), yaw(r)]
    3-DOF maps to [0,1,5] -> [0,1,2] in 3x3 matrices
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def mass_matrix_3dof(
    m: float,
    Iz: float,
    xg: float,
    X_udot: float,
    Y_vdot: float,
    Y_rdot: float,
    N_vdot: float,
    N_rdot: float,
) -> NDArray[np.float64]:
    """Compute the 3x3 system inertia matrix M = M_RB + M_A.

    Args:
        m: vessel mass [kg]
        Iz: yaw moment of inertia [kg*m^2]
        xg: longitudinal CG position from CO [m]
        X_udot: surge added mass (negative value) [kg]
        Y_vdot: sway added mass (negative value) [kg]
        Y_rdot: sway-yaw coupled added mass [kg*m]
        N_vdot: yaw-sway coupled added mass [kg*m]
        N_rdot: yaw added mass (negative value) [kg*m^2]

    Returns:
        3x3 system inertia matrix M (for surge, sway, yaw).
    """
    # Rigid body inertia (3-DOF: surge, sway, yaw)
    M_RB = np.array(
        [
            [m, 0.0, -m * xg],
            [0.0, m, m * xg],
            [-m * xg, m * xg, Iz],
        ]
    )

    # Added mass matrix (sign convention: hydrodynamic derivatives are negative)
    M_A = np.array(
        [
            [-X_udot, 0.0, 0.0],
            [0.0, -Y_vdot, -Y_rdot],
            [0.0, -N_vdot, -N_rdot],
        ]
    )

    return M_RB + M_A  # type: ignore[no-any-return]


def coriolis_matrix_3dof(
    m: float,
    Iz: float,
    xg: float,
    X_udot: float,
    Y_vdot: float,
    Y_rdot: float,
    N_vdot: float,
    N_rdot: float,
    nu3: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute the 3x3 Coriolis + centripetal matrix C(nu) = C_RB + C_A.

    Args:
        nu3: 3-element velocity vector [u, v, r]
        (other args: same as mass_matrix_3dof)

    Returns:
        3x3 Coriolis matrix.
    """
    u, v, r = nu3

    # Rigid body Coriolis (Fossen eq. 3.53, 3-DOF)
    C_RB = np.array(
        [
            [0.0, 0.0, -m * (xg * r + v)],
            [0.0, 0.0, m * u],
            [m * (xg * r + v), -m * u, 0.0],
        ]
    )

    # Added mass Coriolis (Fossen eq. 3.62, 3-DOF)
    C_A = np.array(
        [
            [0.0, 0.0, Y_vdot * v + Y_rdot * r],
            [0.0, 0.0, -X_udot * u],
            [-Y_vdot * v - Y_rdot * r, X_udot * u, 0.0],
        ]
    )

    return C_RB + C_A  # type: ignore[no-any-return]


def damping_matrix_3dof(
    Xu: float,
    Yv: float,
    Yr: float,
    Nv: float,
    Nr: float,
    Xuu: float,
    Yvv: float,
    Yrr: float,
    Nvv: float,
    Nrr: float,
    nu3: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute the 3x3 damping matrix D(nu) = D_linear + D_nonlinear(nu).

    Uses Fossen's formulation with linear + quadratic damping.

    Args:
        Xu..Nr: linear hydrodynamic damping derivatives (negative values)
        Xuu..Nrr: quadratic hydrodynamic damping derivatives (negative values)
        nu3: 3-element velocity vector [u, v, r]

    Returns:
        3x3 damping matrix.
    """
    u, v, r = nu3

    # Linear damping
    D_lin = np.array(
        [
            [-Xu, 0.0, 0.0],
            [0.0, -Yv, -Yr],
            [0.0, -Nv, -Nr],
        ]
    )

    # Nonlinear (quadratic) damping
    D_nonlin = np.array(
        [
            [-Xuu * abs(u), 0.0, 0.0],
            [0.0, -Yvv * abs(v), -Yrr * abs(r)],
            [0.0, -Nvv * abs(v), -Nrr * abs(r)],
        ]
    )

    return D_lin + D_nonlin  # type: ignore[no-any-return]


def equations_of_motion_3dof(
    M_inv: NDArray[np.float64],
    C: NDArray[np.float64],
    D: NDArray[np.float64],
    nu3: NDArray[np.float64],
    tau3: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute acceleration: nu_dot = M^{-1} * (tau - C*nu - D*nu).

    Args:
        M_inv: inverted 3x3 mass matrix
        C: 3x3 Coriolis matrix at current nu
        D: 3x3 damping matrix at current nu
        nu3: current velocity vector [u, v, r]
        tau3: external force vector [X, Y, N]

    Returns:
        3-element acceleration vector [u_dot, v_dot, r_dot].
    """
    return M_inv @ (tau3 - C @ nu3 - D @ nu3)  # type: ignore[no-any-return]


def rotation_matrix_3dof(psi: float) -> NDArray[np.float64]:
    """2D rotation matrix for NED position update from body velocities.

    eta_dot = R(psi) * nu3

    For 3-DOF: [x_dot, y_dot, psi_dot] = R(psi) * [u, v, r]
    """
    c = np.cos(psi)
    s = np.sin(psi)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


# =========================================================================
# 6-DOF functions
# =========================================================================


def mass_matrix_6dof(
    m: float,
    Ix: float,
    Iy: float,
    Iz: float,
    xg: float,
    zg: float,
    X_udot: float,
    Y_vdot: float,
    Z_wdot: float,
    K_pdot: float,
    M_qdot: float,
    N_rdot: float,
    Y_rdot: float = 0.0,
    N_vdot: float = 0.0,
) -> NDArray[np.float64]:
    """Compute the 6x6 system inertia matrix M = M_RB + M_A.

    Simplified model assuming symmetric vessel with key coupling terms.

    Args:
        m: vessel mass [kg]
        Ix, Iy, Iz: moments of inertia [kg*m^2]
        xg, zg: CG position from CO [m]
        X_udot..N_rdot: added mass coefficients (negative by convention)
        Y_rdot, N_vdot: sway-yaw coupling added mass

    Returns:
        6x6 system inertia matrix.
    """
    # Rigid body inertia (Fossen eq. 3.44)
    M_RB = np.array(
        [
            [m, 0, 0, 0, m * zg, -m * xg],
            [0, m, 0, -m * zg, 0, 0],
            [0, 0, m, 0, -m * xg, 0],
            [0, -m * zg, 0, Ix, 0, 0],
            [m * zg, 0, -m * xg, 0, Iy, 0],
            [-m * xg, 0, 0, 0, 0, Iz],
        ],
        dtype=np.float64,
    )

    # Added mass (diagonal + sway-yaw coupling)
    M_A = np.zeros((6, 6))
    M_A[0, 0] = -X_udot
    M_A[1, 1] = -Y_vdot
    M_A[2, 2] = -Z_wdot
    M_A[3, 3] = -K_pdot
    M_A[4, 4] = -M_qdot
    M_A[5, 5] = -N_rdot
    M_A[1, 5] = -Y_rdot
    M_A[5, 1] = -N_vdot

    return M_RB + M_A  # type: ignore[no-any-return]


def coriolis_matrix_6dof(
    m: float,
    Ix: float,
    Iy: float,
    Iz: float,
    xg: float,
    zg: float,
    X_udot: float,
    Y_vdot: float,
    Z_wdot: float,
    K_pdot: float,
    M_qdot: float,
    N_rdot: float,
    nu6: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute the 6x6 Coriolis + centripetal matrix C(nu).

    Simplified formulation using antisymmetric structure.

    Args:
        nu6: 6-element velocity [u, v, w, p, q, r]
        (other args: mass and inertia parameters)

    Returns:
        6x6 antisymmetric Coriolis matrix.
    """
    u, v, w, p, q, r = nu6

    # Rigid body Coriolis (Fossen, simplified for small xg, zg)
    C = np.zeros((6, 6))

    # Upper-right block: -S(M_RB_11 * nu1 + M_RB_12 * nu2)
    C[0, 3] = -m * zg * q
    C[0, 4] = m * (zg * p + w)
    C[0, 5] = -m * (xg * r + v)
    C[1, 3] = -m * (zg * p + w)
    C[1, 4] = 0
    C[1, 5] = m * u
    C[2, 3] = m * (xg * r + v)
    C[2, 4] = -m * u
    C[2, 5] = 0

    # Lower-left = transpose of upper-right (antisymmetric)
    C[3, 0] = -C[0, 3]
    C[3, 1] = -C[1, 3]
    C[3, 2] = -C[2, 3]
    C[4, 0] = -C[0, 4]
    C[4, 1] = -C[1, 4]
    C[4, 2] = -C[2, 4]
    C[5, 0] = -C[0, 5]
    C[5, 1] = -C[1, 5]
    C[5, 2] = -C[2, 5]

    # Lower-right block: -S(M_RB_21 * nu1 + M_RB_22 * nu2)
    C[3, 4] = Iz * r
    C[3, 5] = -Iy * q
    C[4, 3] = -Iz * r
    C[4, 5] = Ix * p
    C[5, 3] = Iy * q
    C[5, 4] = -Ix * p

    # Added mass Coriolis (simplified diagonal added mass)
    C_A = np.zeros((6, 6))
    C_A[0, 4] = Z_wdot * w
    C_A[0, 5] = -Y_vdot * v
    C_A[1, 3] = -Z_wdot * w
    C_A[1, 5] = X_udot * u
    C_A[2, 3] = Y_vdot * v
    C_A[2, 4] = -X_udot * u

    C_A[3, 1] = Z_wdot * w
    C_A[3, 2] = -Y_vdot * v
    C_A[3, 4] = -N_rdot * r
    C_A[3, 5] = M_qdot * q
    C_A[4, 0] = -Z_wdot * w
    C_A[4, 2] = X_udot * u
    C_A[4, 3] = N_rdot * r
    C_A[4, 5] = -K_pdot * p
    C_A[5, 0] = Y_vdot * v
    C_A[5, 1] = -X_udot * u
    C_A[5, 3] = -M_qdot * q
    C_A[5, 4] = K_pdot * p

    return C + C_A  # type: ignore[no-any-return]


def damping_matrix_6dof(
    Xu: float,
    Yv: float,
    Zw: float,
    Kp: float,
    Mq: float,
    Nr: float,
    Xuu: float,
    Yvv: float,
    Zww: float,
    Kpp: float,
    Mqq: float,
    Nrr: float,
    Yr: float,
    Nv: float,
    nu6: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute the 6x6 damping matrix D(nu) = D_linear + D_nonlinear.

    Args:
        Xu..Nr: linear damping coefficients (negative)
        Xuu..Nrr: quadratic damping coefficients (negative)
        Yr, Nv: sway-yaw coupling damping (negative)
        nu6: 6-element velocity [u, v, w, p, q, r]

    Returns:
        6x6 damping matrix.
    """
    u, v, w, p, q, r = nu6

    D = np.diag(
        [
            -Xu - Xuu * abs(u),
            -Yv - Yvv * abs(v),
            -Zw - Zww * abs(w),
            -Kp - Kpp * abs(p),
            -Mq - Mqq * abs(q),
            -Nr - Nrr * abs(r),
        ]
    )

    # Sway-yaw coupling
    D[1, 5] = -Yr
    D[5, 1] = -Nv

    return D  # type: ignore[no-any-return]


def equations_of_motion_6dof(
    M_inv: NDArray[np.float64],
    C: NDArray[np.float64],
    D: NDArray[np.float64],
    nu6: NDArray[np.float64],
    tau6: NDArray[np.float64],
    g_eta: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute 6-DOF acceleration: nu_dot = M^{-1} * (tau - C*nu - D*nu - g(eta)).

    Args:
        M_inv: inverted 6x6 mass matrix
        C: 6x6 Coriolis matrix
        D: 6x6 damping matrix
        nu6: 6-element velocity vector
        tau6: 6-element external force vector
        g_eta: 6-element hydrostatic restoring force vector

    Returns:
        6-element acceleration vector.
    """
    return M_inv @ (tau6 - C @ nu6 - D @ nu6 - g_eta)  # type: ignore[no-any-return]


def rotation_matrix_6dof(
    phi: float,
    theta: float,
    psi: float,
) -> NDArray[np.float64]:
    """6x6 kinematic transformation: eta_dot = J(eta) * nu.

    Block diagonal: J = diag(R_nb, T_Theta)
    - R_nb: 3x3 body-to-NED rotation (ZYX Euler angles)
    - T_Theta: 3x3 angular velocity transformation

    Args:
        phi: roll [rad]
        theta: pitch [rad]
        psi: yaw [rad]

    Returns:
        6x6 transformation matrix.
    """
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    cth = np.cos(theta)
    sth = np.sin(theta)
    cpsi = np.cos(psi)
    spsi = np.sin(psi)

    # Rotation matrix (body → NED) using ZYX convention
    R = np.array(
        [
            [cpsi * cth, cpsi * sth * sphi - spsi * cphi, cpsi * sth * cphi + spsi * sphi],
            [spsi * cth, spsi * sth * sphi + cpsi * cphi, spsi * sth * cphi - cpsi * sphi],
            [-sth, cth * sphi, cth * cphi],
        ]
    )

    # Angular velocity transformation (valid for theta != +-90°)
    if abs(cth) < 1e-10:
        cth = 1e-10  # avoid division by zero
    T = np.array(
        [
            [1.0, sphi * sth / cth, cphi * sth / cth],
            [0.0, cphi, -sphi],
            [0.0, sphi / cth, cphi / cth],
        ]
    )

    J = np.zeros((6, 6))
    J[:3, :3] = R
    J[3:, 3:] = T

    return J
