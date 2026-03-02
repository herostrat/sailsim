"""Smoke tests for the PlaybackViewer (non-interactive, Agg backend)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from sailsim.core.types import (
    ControlCommand,
    CurrentState,
    ForceData,
    SensorData,
    VesselState,
    WaveState,
    WindState,
)
from sailsim.recording.recorder import Recorder
from sailsim.viewer.playback import PlaybackViewer


def _make_3dof_recording(n_steps: int = 20) -> Recorder:
    """Create a minimal 3-DOF recording."""
    rec = Recorder()
    for i in range(n_steps):
        t = i * 0.1
        state = VesselState(
            eta=np.array([t * 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]),
            nu=np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )
        sensors = SensorData(heading=0.0, speed_through_water=2.0)
        control = ControlCommand(rudder_angle=0.01 * i)
        wind = WindState(speed=5.0, direction=1.0)
        forces = ForceData(
            sail=np.array([100.0, -50.0, -20.0]),
            rudder=np.array([-5.0, 20.0, -80.0]),
            keel=np.array([-10.0, 100.0, -30.0]),
        )
        rec.record(t, state, sensors, control, wind, forces=forces, target_heading=0.0)
    return rec


def _make_6dof_recording(n_steps: int = 20) -> Recorder:
    """Create a minimal 6-DOF recording with waves and current."""
    rec = Recorder()
    for i in range(n_steps):
        t = i * 0.1
        state = VesselState(
            eta=np.array([t * 0.5, 0.1 * i, 0.0, 0.05 * np.sin(i), 0.01, 0.0]),
            nu=np.array([2.0, 0.1, 0.0, 0.05, 0.01, 0.0]),
        )
        sensors = SensorData(
            heading=0.0,
            speed_through_water=2.0,
            speed_over_ground=2.3,
            course_over_ground=0.1,
            roll=0.05 * np.sin(i),
        )
        control = ControlCommand(rudder_angle=0.01 * i, sail_trim=0.6)
        wind = WindState(speed=8.0 + 0.5 * np.sin(i), direction=1.57)
        forces = ForceData(
            sail=np.array([100.0, -50.0, 0.0, -250.0, -30.0, -20.0]),
            rudder=np.array([-5.0, 20.0, 0.0, 10.0, 0.0, -80.0]),
            keel=np.array([-10.0, 100.0, 0.0, 150.0, 0.0, -30.0]),
        )
        current = CurrentState(speed=0.5, direction=0.0)
        waves = WaveState(Hs=1.0, Tp=6.0, direction=1.57, elevation=0.3 * np.sin(i))
        rec.record(
            t,
            state,
            sensors,
            control,
            wind,
            forces=forces,
            current=current,
            waves=waves,
            target_heading=0.0,
        )
    return rec


def test_viewer_constructs_3dof():
    """PlaybackViewer should construct without errors for 3-DOF data."""
    rec = _make_3dof_recording()
    viewer = PlaybackViewer([("test_3dof", rec)])
    assert viewer._active_page == 0
    plt.close("all")


def test_viewer_constructs_6dof():
    """PlaybackViewer should construct without errors for 6-DOF data with waves/current."""
    rec = _make_6dof_recording()
    viewer = PlaybackViewer([("test_6dof", rec)])
    assert viewer._active_page == 0
    plt.close("all")


def test_viewer_page_switch_3_pages():
    """Page switching should cycle through all 3 pages."""
    rec = _make_3dof_recording()
    viewer = PlaybackViewer([("test", rec)])

    assert viewer._active_page == 0
    viewer._switch_page(1)
    assert viewer._active_page == 1
    viewer._switch_page(2)
    assert viewer._active_page == 2
    viewer._switch_page(0)
    assert viewer._active_page == 0
    plt.close("all")


def test_viewer_multi_run():
    """Viewer should handle multiple runs."""
    rec1 = _make_3dof_recording()
    rec2 = _make_6dof_recording()
    viewer = PlaybackViewer([("run1", rec1), ("run2", rec2)])
    assert viewer._max_frames == 20
    plt.close("all")


def test_viewer_target_lines():
    """Target heading lines should exist for each run."""
    rec1 = _make_3dof_recording()
    rec2 = _make_6dof_recording()
    viewer = PlaybackViewer([("run1", rec1), ("run2", rec2)])
    assert len(viewer._target_lines) == 2
    plt.close("all")


def test_viewer_attitude_axes_exist():
    """Attitude axes (bow, side, top) should be created."""
    rec = _make_3dof_recording()
    viewer = PlaybackViewer([("test", rec)])
    assert hasattr(viewer, "_ax_bow")
    assert hasattr(viewer, "_ax_side")
    assert hasattr(viewer, "_ax_top")
    plt.close("all")


def test_viewer_attitude_patches_per_run():
    """Each run should have attitude patches for all three views."""
    rec1 = _make_3dof_recording()
    rec2 = _make_6dof_recording()
    viewer = PlaybackViewer([("run1", rec1), ("run2", rec2)])
    assert len(viewer._attitude_patches) == 2
    for att in viewer._attitude_patches:
        # Bow view: hull, keel, mast
        assert att.bow_hull is not None
        assert att.bow_keel is not None
        assert att.bow_mast is not None
        # Side view: hull, keel, rudder, mast
        assert att.side_hull is not None
        assert att.side_keel is not None
        assert att.side_rudder is not None
        assert att.side_mast is not None
        # Top view: hull, keel
        assert att.top_hull is not None
        assert att.top_keel is not None
        # Labels
        assert att.bow_label is not None
        assert att.side_label is not None
        assert att.top_label is not None
    plt.close("all")


def test_viewer_attitude_rotation_updates():
    """Attitude patches should update when cursors move."""
    rec = _make_6dof_recording()
    viewer = PlaybackViewer([("test", rec)])
    viewer._current_frame = 10
    viewer._update_cursors()
    # Check that angle labels have been populated
    att = viewer._attitude_patches[0]
    assert att.bow_label.get_text() != ""
    assert att.side_label.get_text() != ""
    assert att.top_label.get_text() != ""
    plt.close("all")
