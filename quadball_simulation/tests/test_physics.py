"""Numerical helper tests."""

import numpy as np

from quadball.physics import segment_point_distance, unit


def test_unit_zero_is_safe() -> None:
    """A zero direction never emits NaN values."""
    np.testing.assert_array_equal(unit(np.zeros(2)), np.zeros(2))


def test_segment_distance() -> None:
    """Segment distance projects onto the finite segment."""
    assert segment_point_distance(np.array([0.0, 0.0]), np.array([2.0, 0.0]), np.array([1.0, 1.0])) == 1.0
