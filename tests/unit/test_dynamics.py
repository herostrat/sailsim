"""Test Fossen 3-DOF dynamics against known analytical behaviors."""

import numpy as np
import pytest

from sailsim.physics.dynamics import (
    coriolis_matrix_3dof,
    damping_matrix_3dof,
    equations_of_motion_3dof,
    mass_matrix_3dof,
    rotation_matrix_3dof,
)

# Simple test parameters (uniform block, no coupling)
MASS = 1000.0
IZ = 5000.0
XG = 0.0
X_UDOT = -100.0
Y_VDOT = -1000.0
Y_RDOT = 0.0
N_VDOT = 0.0
N_RDOT = -2000.0


def test_mass_matrix_symmetric():
    """Mass matrix must be symmetric and positive definite."""
    M = mass_matrix_3dof(MASS, IZ, XG, X_UDOT, Y_VDOT, Y_RDOT, N_VDOT, N_RDOT)
    np.testing.assert_array_almost_equal(M, M.T)
    eigenvalues = np.linalg.eigvalsh(M)
    assert all(eigenvalues > 0), f"Non-positive eigenvalues: {eigenvalues}"


def test_mass_matrix_values():
    """Check specific matrix entries."""
    M = mass_matrix_3dof(MASS, IZ, XG, X_UDOT, Y_VDOT, Y_RDOT, N_VDOT, N_RDOT)
    # M[0,0] = m - X_udot = 1000 - (-100) = 1100
    assert M[0, 0] == pytest.approx(1100.0)
    # M[1,1] = m - Y_vdot = 1000 - (-1000) = 2000
    assert M[1, 1] == pytest.approx(2000.0)
    # M[2,2] = Iz - N_rdot = 5000 - (-2000) = 7000
    assert M[2, 2] == pytest.approx(7000.0)


def test_coriolis_skew_symmetric():
    """C(nu) should be skew-symmetric: C + C^T = 0."""
    nu3 = np.array([2.0, 0.5, 0.1])
    C = coriolis_matrix_3dof(MASS, IZ, XG, X_UDOT, Y_VDOT, Y_RDOT, N_VDOT, N_RDOT, nu3)
    # C + C^T should be zero (skew-symmetric)
    np.testing.assert_array_almost_equal(C + C.T, np.zeros((3, 3)), decimal=10)


def test_coriolis_zero_at_rest():
    """At rest (nu=0), Coriolis forces should be zero."""
    nu3 = np.array([0.0, 0.0, 0.0])
    C = coriolis_matrix_3dof(MASS, IZ, XG, X_UDOT, Y_VDOT, Y_RDOT, N_VDOT, N_RDOT, nu3)
    np.testing.assert_array_almost_equal(C, np.zeros((3, 3)))


def test_damping_always_dissipative():
    """Damping force should always oppose motion: nu^T * D * nu > 0."""
    nu3 = np.array([2.0, 0.5, 0.1])
    D = damping_matrix_3dof(
        Xu=-50,
        Yv=-500,
        Yr=-200,
        Nv=-200,
        Nr=-2000,
        Xuu=-30,
        Yvv=-800,
        Yrr=-100,
        Nvv=-100,
        Nrr=-3000,
        nu3=nu3,
    )
    power = nu3 @ D @ nu3
    assert power > 0, f"Damping power should be positive (dissipative), got {power}"


def test_eom_force_produces_acceleration():
    """A known force on a known mass should produce a known acceleration."""
    M = mass_matrix_3dof(MASS, IZ, XG, X_UDOT, Y_VDOT, Y_RDOT, N_VDOT, N_RDOT)
    M_inv = np.linalg.inv(M)
    # Zero velocity => C=0, D=0
    C = np.zeros((3, 3))
    D = np.zeros((3, 3))
    nu3 = np.array([0.0, 0.0, 0.0])
    tau = np.array([1100.0, 0.0, 0.0])  # Force = M[0,0] in surge
    nu_dot = equations_of_motion_3dof(M_inv, C, D, nu3, tau)
    # Expected: surge acceleration = tau/M[0,0] = 1.0 m/s²
    assert nu_dot[0] == pytest.approx(1.0, abs=1e-10)
    assert nu_dot[1] == pytest.approx(0.0, abs=1e-10)
    assert nu_dot[2] == pytest.approx(0.0, abs=1e-10)


def test_rotation_matrix_identity_at_zero():
    """At psi=0, rotation is identity (body and NED aligned)."""
    R = rotation_matrix_3dof(0.0)
    np.testing.assert_array_almost_equal(R, np.eye(3))


def test_rotation_matrix_90_degrees():
    """At psi=pi/2, forward motion (u) maps to east (y)."""
    R = rotation_matrix_3dof(np.pi / 2)
    nu3 = np.array([1.0, 0.0, 0.0])  # pure surge
    eta_dot = R @ nu3
    assert eta_dot[0] == pytest.approx(0.0, abs=1e-10)  # no northward motion
    assert eta_dot[1] == pytest.approx(1.0, abs=1e-10)  # eastward motion
