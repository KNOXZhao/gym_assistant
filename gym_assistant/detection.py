"""Utilities for detecting candidate weight plates in a frame."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class PlateCandidate:
    """A detected plate candidate represented by a circle."""

    center: Tuple[float, float]
    radius: float
    score: float

    def to_box(self, frame_shape: Sequence[int], padding: float = 1.1) -> Tuple[int, int, int, int]:
        """Convert the candidate circle to a bounding box."""
        height, width = frame_shape[:2]
        pad_radius = self.radius * padding
        cx, cy = self.center
        x1 = max(int(round(cx - pad_radius)), 0)
        y1 = max(int(round(cy - pad_radius)), 0)
        x2 = min(int(round(cx + pad_radius)), width - 1)
        y2 = min(int(round(cy + pad_radius)), height - 1)
        return x1, y1, x2, y2


def detect_plate_candidates(
    frame: np.ndarray,
    max_candidates: int = 5,
    dp: float = 1.2,
    min_dist_ratio: float = 0.2,
    param1: float = 120.0,
    param2: float = 30.0,
    min_radius_ratio: float = 0.05,
    max_radius_ratio: float = 0.4,
) -> List[PlateCandidate]:
    """Detect circular plate candidates using Hough transform."""
    if frame.ndim != 3:
        raise ValueError("Expected color frame with shape (H, W, C)")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 1.5)

    height, width = gray.shape
    min_dist = min(height, width) * min_dist_ratio
    min_radius = int(min(height, width) * min_radius_ratio)
    max_radius = int(min(height, width) * max_radius_ratio)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=max(int(min_dist), 1),
        param1=param1,
        param2=param2,
        minRadius=max(min_radius, 1),
        maxRadius=max(max_radius, min_radius + 1),
    )

    candidates: List[PlateCandidate] = []
    if circles is not None:
        circles = np.squeeze(circles, axis=0)
        for x, y, r in circles:
            score = float(r)
            candidates.append(PlateCandidate(center=(float(x), float(y)), radius=float(r), score=score))

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:max_candidates]


def overlay_candidates(frame: np.ndarray, candidates: Sequence[PlateCandidate]) -> np.ndarray:
    """Draw the candidate boxes onto the frame."""
    canvas = frame.copy()
    for idx, candidate in enumerate(candidates):
        x1, y1, x2, y2 = candidate.to_box(frame.shape)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cx, cy = map(int, candidate.center)
        cv2.circle(canvas, (cx, cy), int(candidate.radius), (255, 0, 0), 2)
        cv2.putText(
            canvas,
            f"{idx}",
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return canvas
