"""Tests for wave force model."""

from __future__ import annotations

import numpy as np

from sailsim.core.types import WaveState
from sailsim.physics.wave_forces import wave_forces_3dof


class TestWaveForces3DOF:
    def test_zero_for_no_waves(self):
        """Zero Hs should give zero forces."""
        wave = WaveState(Hs=0.0, Tp=8.0, direction=0.0, elevation=0.5)
        forces = wave_forces_3dof(wave, psi=0.0, u=3.0)
        np.testing.assert_array_equal(forces, np.zeros(3))

    def test_added_resistance_negative(self):
        """Added resistance should oppose forward motion (X < 0) in head seas."""
        wave = WaveState(Hs=2.0, Tp=8.0, direction=np.pi, elevation=0.0)
        # Head seas: wave direction = pi, boat heading = 0
        forces = wave_forces_3dof(wave, psi=0.0, u=3.0)
        assert forces[0] < 0, f"Expected negative X force, got {forces[0]}"

    def test_added_resistance_max_at_head_seas(self):
        """Head seas should give more resistance than following seas."""
        WaveState(Hs=2.0, Tp=8.0, direction=0.0, elevation=0.0)

        # Head seas (wave coming from ahead: direction 180° opposite heading)
        wave_head = WaveState(Hs=2.0, Tp=8.0, direction=np.pi, elevation=0.0)
        forces_head = wave_forces_3dof(wave_head, psi=0.0, u=3.0)

        # Following seas (wave going same direction as boat)
        wave_follow = WaveState(Hs=2.0, Tp=8.0, direction=0.0, elevation=0.0)
        forces_follow = wave_forces_3dof(wave_follow, psi=0.0, u=3.0)

        assert forces_head[0] < forces_follow[0]

    def test_lateral_force_from_beam_seas(self):
        """Beam seas with elevation should produce lateral force."""
        wave = WaveState(Hs=2.0, Tp=8.0, direction=np.pi / 2, elevation=0.5)
        forces = wave_forces_3dof(wave, psi=0.0, u=3.0)
        assert abs(forces[1]) > 0, "Expected non-zero Y force for beam seas"

    def test_returns_3_element_array(self):
        wave = WaveState(Hs=1.5, Tp=7.0, direction=0.5, elevation=0.3)
        forces = wave_forces_3dof(wave, psi=0.3, u=2.0)
        assert forces.shape == (3,)
