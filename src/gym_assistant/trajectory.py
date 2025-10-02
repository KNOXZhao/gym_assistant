"""Data structures for handling plate trajectories."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List

import csv
import math

import numpy as np


@dataclass
class Trajectory:
    """Stores the tracked trajectory of the left and right plates."""

    frame_indices: List[int]
    timestamps: List[float]
    left_positions: np.ndarray
    right_positions: np.ndarray
    frame_size: tuple[int, int]
    metadata: dict[str, str] = field(default_factory=dict)

    def to_csv(self, path: Path) -> None:
        """Serialize the trajectory to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    "frame",
                    "timestamp",
                    "left_x",
                    "left_y",
                    "right_x",
                    "right_y",
                ]
            )
            for idx, ts, left, right in zip(
                self.frame_indices,
                self.timestamps,
                self.left_positions,
                self.right_positions,
            ):
                writer.writerow(
                    [
                        idx,
                        f"{ts:.6f}",
                        _format_float(left[0]),
                        _format_float(left[1]),
                        _format_float(right[0]),
                        _format_float(right[1]),
                    ]
                )

    def normalized(self) -> "Trajectory":
        """Return a copy with coordinates normalized to [0, 1]."""
        width, height = self.frame_size
        denom = np.array([width, height], dtype=np.float32)
        left = self.left_positions / denom
        right = self.right_positions / denom
        return Trajectory(
            frame_indices=list(self.frame_indices),
            timestamps=list(self.timestamps),
            left_positions=left,
            right_positions=right,
            frame_size=self.frame_size,
            metadata=dict(self.metadata),
        )

    @classmethod
    def from_measurements(
        cls,
        measurements: Iterable[tuple[int, float, np.ndarray, np.ndarray]],
        frame_size: tuple[int, int],
        metadata: dict[str, str] | None = None,
    ) -> "Trajectory":
        frames: List[int] = []
        timestamps: List[float] = []
        left_positions: List[np.ndarray] = []
        right_positions: List[np.ndarray] = []
        for frame_idx, timestamp, left, right in measurements:
            frames.append(frame_idx)
            timestamps.append(timestamp)
            left_positions.append(left)
            right_positions.append(right)
        return cls(
            frame_indices=frames,
            timestamps=timestamps,
            left_positions=np.stack(left_positions),
            right_positions=np.stack(right_positions),
            frame_size=frame_size,
            metadata=metadata or {},
        )

    def with_smoothing(self, window_size: int) -> "Trajectory":
        if window_size <= 1:
            return self
        left = _smooth_array(self.left_positions, window_size)
        right = _smooth_array(self.right_positions, window_size)
        return Trajectory(
            frame_indices=list(self.frame_indices),
            timestamps=list(self.timestamps),
            left_positions=left,
            right_positions=right,
            frame_size=self.frame_size,
            metadata=dict(self.metadata),
        )


def _format_float(value: float) -> str:
    if math.isnan(value):
        return ""
    return f"{value:.6f}"


def _smooth_array(values: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 1 or values.size == 0:
        return values
    if window_size % 2 == 0:
        raise ValueError("Smoothing window must be odd.")
    pad = window_size // 2
    padded = np.pad(values, ((pad, pad), (0, 0)), mode="edge")
    smoothed = np.empty_like(values, dtype=np.float32)
    for idx in range(values.shape[0]):
        window = padded[idx : idx + window_size]
        if np.all(np.isnan(window)):
            smoothed[idx] = np.array([np.nan, np.nan], dtype=np.float32)
        else:
            with np.errstate(invalid="ignore"):
                smoothed[idx] = np.nanmean(window, axis=0)
    return smoothed
