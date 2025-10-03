"""Utilities for seeding SAM2 tracking with user-provided points."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


@dataclass
class PlateCandidate:
    """A plate seed represented by a positive point prompt."""

    center: Tuple[float, float]
    score: float = 1.0
    radius: float | None = None

    def point_prompt(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return SAM2-compatible point/label arrays."""

        coords = np.asarray([[self.center[0], self.center[1]]], dtype=np.float32)
        labels = np.ones((1,), dtype=np.int32)
        return coords, labels


def candidate_from_coordinates(point: Sequence[float]) -> PlateCandidate:
    """Create a candidate from a raw ``(x, y)`` pair."""

    if len(point) != 2:
        raise ValueError("Expected two values for a point coordinate")
    x, y = point
    return PlateCandidate(center=(float(x), float(y)))
