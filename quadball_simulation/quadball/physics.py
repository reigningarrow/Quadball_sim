"""Numerical helpers for deterministic 2D movement and contact."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def norm(vector: NDArray[np.float64]) -> float:
    """Return the Euclidean magnitude of a vector.

    Parameters
    ----------
    vector : numpy.ndarray
        Input vector.

    Returns
    -------
    float
        Euclidean magnitude.
    """

    return float(np.linalg.norm(vector))


def unit(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return a safe unit vector.

    Parameters
    ----------
    vector : numpy.ndarray
        Input vector.

    Returns
    -------
    numpy.ndarray
        Normalized vector or zeros when its magnitude is negligible.
    """

    magnitude = norm(vector)
    return vector / magnitude if magnitude > 1e-9 else np.zeros(2, dtype=float)


def clamp_position(position: NDArray[np.float64], width: float, height: float) -> NDArray[np.float64]:
    """Clamp a point to the rectangular playing field.

    Parameters
    ----------
    position : numpy.ndarray
        Candidate position.
    width : float
        Field width.
    height : float
        Field height.

    Returns
    -------
    numpy.ndarray
        Clamped copy of the point.
    """

    return np.clip(position, (0.0, 0.0), (width, height)).astype(float)


def segment_point_distance(start: NDArray[np.float64], end: NDArray[np.float64], point: NDArray[np.float64]) -> float:
    """Return the shortest distance between a segment and a point.

    Parameters
    ----------
    start, end : numpy.ndarray
        Segment endpoints.
    point : numpy.ndarray
        Point to test.

    Returns
    -------
    float
        Shortest planar distance.
    """

    delta = end - start
    denom = float(np.dot(delta, delta))
    t = 0.0 if denom < 1e-12 else float(np.clip(np.dot(point - start, delta) / denom, 0.0, 1.0))
    return norm(point - (start + t * delta))
