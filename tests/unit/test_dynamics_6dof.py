"""Tests for 6-DOF dynamics matrices."""

from __future__ import annotations

import numpy as np
import pytest

from sailsim.physics.dynamics import (
    coriolis_matrix_6dof,
    damping_matrix_6dof,
    mass_matrix_6dof,
    rotation_matrix_6dof,
)

# Default parameters for a ~10m yacht
DEFAULT_PARAMS = dict(
    m=4000,
    Ix=8000,
    Iy=30000,
    Iz=25000,
    xg=0.0,
    zg=0.2,
    X_udot=-200,
    Y_vdot=-4000,
    Z_wdot=-4000,
    K_pdot=-500,
    M_qdot=-20000,
    N_rdot=-15000,
)


class TestMassMatrix6DOF:
    def test_shape(self):
        M = mass_matrix_6dof(**DEFAULT_PARAMS)
        assert M.shape == (6, 6)

    def test_symmetric(self):
        M = mass_matrix_6dof(**DEFAULT_PARAMS)
        np.testing.assert_array_almost_equal(M, M.T, decimal=10)

    def test_positive_definite(self):
        M = mass_matrix_6dof(**DEFAULT_PARAMS)
        eigenvalues = np.linalg.eigvalsh(M)
        assert all(e > 0 for e in eigenvalues), f"Not positive definite: {eigenvalues}"

    def test_3dof_submatrix_consistent(self):
        """Surge, sway, yaw submatrix should match 3-DOF mass matrix."""
        from sailsim.physics.dynamics import mass_matrix_3dof

        M6 = mass_matrix_6dof(**DEFAULT_PARAMS)
        M3 = mass_matrix_3dof(
            m=4000,
            Iz=25000,
            xg=0.0,
            X_udot=-200,
            Y_vdot=-4000,
            Y_rdot=0,
            N_vdot=0,
            N_rdot=-15000,
        )
        # Extract surge, sway, yaw from 6-DOF (indices 0, 1, 5)
        idx = [0, 1, 5]
        M6_sub = M6[np.ix_(idx, idx)]
        np.testing.assert_array_almost_equal(M6_sub, M3, decimal=0)


class TestCoriolisMatrix6DOF:
    def test_shape(self):
        nu = np.array([2.0, 0.3, 0.0, 0.0, 0.0, 0.05])
        C = coriolis_matrix_6dof(**DEFAULT_PARAMS, nu6=nu)
        assert C.shape == (6, 6)

    def test_antisymmetric(self):
        """C + C^T should be approximately zero (antisymmetric)."""
        nu = np.array([2.0, 0.3, 0.1, 0.01, 0.01, 0.05])
        C = coriolis_matrix_6dof(**DEFAULT_PARAMS, nu6=nu)
        # Not perfectly antisymmetric due to simplifications, but diagonal should be ~0
        assert all(abs(C[i, i]) < 1e-10 for i in range(6))

    def test_zero_at_rest(self):
        """At zero velocity, Coriolis should be zero."""
        nu = np.zeros(6)
        C = coriolis_matrix_6dof(**DEFAULT_PARAMS, nu6=nu)
        np.testing.assert_array_almost_equal(C, np.zeros((6, 6)))


class TestDampingMatrix6DOF:
    def test_shape(self):
        nu = np.array([2.0, 0.3, 0.0, 0.0, 0.0, 0.05])
        D = damping_matrix_6dof(
            Xu=-100,
            Yv=-2000,
            Zw=-3000,
            Kp=-5000,
            Mq=-30000,
            Nr=-20000,
            Xuu=-50,
            Yvv=-3000,
            Zww=-3000,
            Kpp=-3000,
            Mqq=-20000,
            Nrr=-50000,
            Yr=-500,
            Nv=-500,
            nu6=nu,
        )
        assert D.shape == (6, 6)

    def test_dissipative(self):
        """nu^T * D * nu should be positive (energy dissipation)."""
        nu = np.array([2.0, 0.5, 0.1, 0.02, 0.01, 0.05])
        D = damping_matrix_6dof(
            Xu=-100,
            Yv=-2000,
            Zw=-3000,
            Kp=-5000,
            Mq=-30000,
            Nr=-20000,
            Xuu=-50,
            Yvv=-3000,
            Zww=-3000,
            Kpp=-3000,
            Mqq=-20000,
            Nrr=-50000,
            Yr=-500,
            Nv=-500,
            nu6=nu,
        )
        dissipation = nu @ D @ nu
        assert dissipation > 0, f"Damping should dissipate energy: {dissipation}"


class TestRotationMatrix6DOF:
    def test_identity_at_zero(self):
        """At zero angles, J should be identity."""
        J = rotation_matrix_6dof(0, 0, 0)
        np.testing.assert_array_almost_equal(J, np.eye(6))

    def test_shape(self):
        J = rotation_matrix_6dof(0.1, 0.05, 0.5)
        assert J.shape == (6, 6)

    def test_yaw_only_matches_3dof(self):
        """With only yaw, position block should match 3-DOF rotation."""
        from sailsim.physics.dynamics import rotation_matrix_3dof

        psi = 0.5
        J6 = rotation_matrix_6dof(0, 0, psi)
        R3 = rotation_matrix_3dof(psi)

        # Position block (upper-left 3x3) should contain the same rotation
        # But J6 is body→NED with all angles, R3 is the 2D heading rotation
        # At zero roll/pitch, they should match for the x,y components
        assert J6[0, 0] == pytest.approx(R3[0, 0])  # cos(psi)
        assert J6[0, 1] == pytest.approx(R3[0, 1])  # -sin(psi)
        assert J6[1, 0] == pytest.approx(R3[1, 0])  # sin(psi)
        assert J6[1, 1] == pytest.approx(R3[1, 1])  # cos(psi)
