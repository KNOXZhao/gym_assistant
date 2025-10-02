"""Utilities for working with video inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class VideoFrame:
    index: int
    timestamp: float
    image: np.ndarray


class VideoReader:
    """Iterate over frames in a video file."""

    def __init__(self, path: Path, sample_rate: int = 1) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video file '{self.path}' does not exist")
        self.sample_rate = max(1, sample_rate)
        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            raise RuntimeError(f"Failed to open video file: {self.path}")
        fps = self._capture.get(cv2.CAP_PROP_FPS)
        self._fps = fps if fps > 0 else 30.0

    @property
    def fps(self) -> float:
        return self._fps

    def __iter__(self) -> Iterator[VideoFrame]:
        frame_index = 0
        output_index = 0
        while True:
            success, frame = self._capture.read()
            if not success or frame is None:
                break
            if frame_index % self.sample_rate == 0:
                timestamp = output_index / self.fps
                output_index += 1
                yield VideoFrame(index=frame_index, timestamp=timestamp, image=frame)
            frame_index += 1

    def release(self) -> None:
        if self._capture:
            self._capture.release()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.release()
