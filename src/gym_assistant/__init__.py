"""Gym Assistant package.

Provides tools for extracting barbell plate trajectories from training videos using
Segment Anything 2 (SAM 2.1) as the base perception model.
"""

from .cli import main
from .plate_tracker import PlateTracker, PlateTrackerConfig
from .trajectory import Trajectory

__all__ = [
    "main",
    "PlateTracker",
    "PlateTrackerConfig",
    "Trajectory",
]
