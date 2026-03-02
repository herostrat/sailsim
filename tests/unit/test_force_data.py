"""Tests for ForceData and Yacht3DOF.compute_forces()."""

import numpy as np

from sailsim.core.types import ControlCommand, ForceData, VesselState, WindState
from sailsim.vessel.yacht_3dof import Yacht3DOF
from sailsim.viewer.playback import _force_n_index


def test_force_data_defaults():
    """Default ForceData should be all zeros."""
    fd = ForceData()
    assert np.allclose(fd.sail, 0.0)
    assert np.allclose(fd.rudder, 0.0)
    assert np.allclose(fd.keel, 0.0)


def test_force_data_total():
    """Total property should sum all components."""
    fd = ForceData(
        sail=np.array([10.0, 20.0, 30.0]),
        rudder=np.array([1.0, 2.0, 3.0]),
        keel=np.array([0.5, 0.5, 0.5]),
    )
    expected = np.array([11.5, 22.5, 33.5])
    assert np.allclose(fd.total, expected)


def test_compute_forces_returns_force_data():
    """compute_forces() should return a ForceData instance."""
    yacht = Yacht3DOF()
    yacht.reset(
        VesselState(
            eta=np.zeros(6),
            nu=np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )
    )
    wind = WindState(speed=5.0, direction=1.0)
    control = ControlCommand(rudder_angle=0.1)

    forces = yacht.compute_forces(wind, control)
    assert isinstance(forces, ForceData)
    assert forces.sail.shape == (3,)
    assert forces.rudder.shape == (3,)
    assert forces.keel.shape == (3,)


def test_compute_forces_does_not_mutate_state():
    """compute_forces() should not change yacht state."""
    yacht = Yacht3DOF()
    initial_state = VesselState(
        eta=np.array([10.0, 5.0, 0.0, 0.0, 0.0, 0.5]),
        nu=np.array([3.0, 0.2, 0.0, 0.0, 0.0, 0.05]),
    )
    yacht.reset(initial_state)
    eta_before = yacht.state.eta.copy()
    nu_before = yacht.state.nu.copy()

    wind = WindState(speed=8.0, direction=1.0)
    control = ControlCommand(rudder_angle=0.2)
    yacht.compute_forces(wind, control)

    assert np.allclose(yacht.state.eta, eta_before)
    assert np.allclose(yacht.state.nu, nu_before)


def test_compute_forces_with_wind_produces_sail_force():
    """With wind, sail force should be non-zero."""
    yacht = Yacht3DOF()
    yacht.reset(
        VesselState(
            eta=np.zeros(6),
            nu=np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )
    )
    wind = WindState(speed=8.0, direction=np.radians(60.0))
    control = ControlCommand()

    forces = yacht.compute_forces(wind, control)
    assert np.linalg.norm(forces.sail) > 0


def test_compute_forces_rudder_with_speed():
    """Rudder deflection at speed should produce rudder force."""
    yacht = Yacht3DOF()
    yacht.reset(
        VesselState(
            eta=np.zeros(6),
            nu=np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )
    )
    wind = WindState()
    control = ControlCommand(rudder_angle=0.15)

    forces = yacht.compute_forces(wind, control)
    assert abs(forces.rudder[1]) > 0  # Y force
    assert abs(forces.rudder[2]) > 0  # N moment


def test_force_n_index_3dof():
    """3-element force array should have N at index 2."""
    arr = np.array([1.0, 2.0, 3.0])
    assert _force_n_index(arr) == 2


def test_force_n_index_6dof():
    """6-element force array should have N at index 5."""
    arr = np.array([1.0, 2.0, 0.0, 0.0, 0.0, 6.0])
    assert _force_n_index(arr) == 5
