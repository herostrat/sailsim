"""Unit tests for waypoint guidance helper functions."""

from __future__ import annotations

import numpy as np

from sailsim.core.runner import _compute_waypoint_heading, _distance_to_waypoint


def test_heading_due_north():
    heading = _compute_waypoint_heading(0.0, 0.0, 100.0, 0.0)
    assert abs(heading - 0.0) < 0.01


def test_heading_due_east():
    heading = _compute_waypoint_heading(0.0, 0.0, 0.0, 100.0)
    assert abs(heading - np.pi / 2) < 0.01


def test_heading_due_south():
    heading = _compute_waypoint_heading(100.0, 0.0, 0.0, 0.0)
    assert abs(abs(heading) - np.pi) < 0.01


def test_heading_due_west():
    heading = _compute_waypoint_heading(0.0, 100.0, 0.0, 0.0)
    assert abs(heading - (-np.pi / 2)) < 0.01


def test_heading_northeast():
    heading = _compute_waypoint_heading(0.0, 0.0, 100.0, 100.0)
    assert abs(heading - np.pi / 4) < 0.01


def test_distance_3_4_5():
    dist = _distance_to_waypoint(0.0, 0.0, 3.0, 4.0)
    assert abs(dist - 5.0) < 0.01


def test_distance_zero():
    dist = _distance_to_waypoint(10.0, 20.0, 10.0, 20.0)
    assert abs(dist) < 0.001
