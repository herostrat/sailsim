"""Tests for hydrostatic restoring forces."""

from __future__ import annotations

import numpy as np
import pytest

from sailsim.physics.hydrostatics import restoring_forces_6dof


class TestRestoringForces:
    def test_zero_at_equilibrium(self):
        """No restoring forces when upright and at waterline."""
        forces = restoring_forces_6dof(z=0, phi=0, theta=0, mass=4000, GM_T=1.2, GM_L=15, Aw=22)
        np.testing.assert_array_almost_equal(forces, np.zeros(6))

    def test_heave_g_eta_direction(self):
        """Submerged more (positive z) → positive g_Z (subtracted in EOM → pushes up)."""
        g = restoring_forces_6dof(z=0.1, phi=0, theta=0, mass=4000, GM_T=1.2, GM_L=15, Aw=22)
        assert g[2] > 0, "g(eta) heave should be positive when submerged"

    def test_heave_g_eta_opposite(self):
        """Above waterline (negative z) → negative g_Z."""
        g = restoring_forces_6dof(z=-0.1, phi=0, theta=0, mass=4000, GM_T=1.2, GM_L=15, Aw=22)
        assert g[2] < 0

    def test_roll_g_eta_direction(self):
        """Starboard heel (positive phi) → positive g_K (subtracted in EOM → restores port)."""
        g = restoring_forces_6dof(z=0, phi=0.2, theta=0, mass=4000, GM_T=1.2, GM_L=15, Aw=22)
        assert g[3] > 0, "g(eta) roll should be positive for positive heel"

    def test_roll_restoring_symmetric(self):
        """Port and starboard heel should give opposite moments."""
        f_stbd = restoring_forces_6dof(z=0, phi=0.1, theta=0, mass=4000, GM_T=1.2, GM_L=15, Aw=22)
        f_port = restoring_forces_6dof(z=0, phi=-0.1, theta=0, mass=4000, GM_T=1.2, GM_L=15, Aw=22)
        assert f_stbd[3] == pytest.approx(-f_port[3], abs=0.01)

    def test_pitch_g_eta_direction(self):
        """Bow up (positive theta) → positive g_M (subtracted in EOM → restores bow down)."""
        g = restoring_forces_6dof(z=0, phi=0, theta=0.1, mass=4000, GM_T=1.2, GM_L=15, Aw=22)
        assert g[4] > 0

    def test_gm_sensitivity(self):
        """Larger GM_T → stronger roll restoring."""
        f_small = restoring_forces_6dof(z=0, phi=0.1, theta=0, mass=4000, GM_T=0.5, GM_L=15, Aw=22)
        f_large = restoring_forces_6dof(z=0, phi=0.1, theta=0, mass=4000, GM_T=2.0, GM_L=15, Aw=22)
        assert abs(f_large[3]) > abs(f_small[3])

    def test_no_surge_sway_yaw(self):
        """Restoring forces should only affect Z, K, M — never X, Y, N."""
        forces = restoring_forces_6dof(
            z=0.1, phi=0.2, theta=0.1, mass=4000, GM_T=1.2, GM_L=15, Aw=22
        )
        assert forces[0] == 0.0  # X
        assert forces[1] == 0.0  # Y
        assert forces[5] == 0.0  # N
