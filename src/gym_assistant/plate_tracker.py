"""Core tracking logic for the gym assistant application."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import math

import cv2
import numpy as np

from .sam_model import Sam2Config, Sam2Segmenter, SamModelNotAvailableError
from .trajectory import Trajectory
from .video import VideoReader


@dataclass
class PlateTrackerConfig:
    model_checkpoint: Path
    device: str = "cuda"
    sample_rate: int = 2
    min_plate_diameter: float = 0.15
    max_plate_diameter: float = 0.6
    min_circularity: float = 0.5
    smoothing_window: int = 5
    sam_config_file: str = "configs/sam2.1/sam2.1_hiera_l.yaml"


@dataclass
class PlateDetection:
    mask: np.ndarray
    centroid: Tuple[float, float]
    bbox: Tuple[float, float, float, float]
    area: float


class PlateTracker:
    def __init__(self, config: PlateTrackerConfig) -> None:
        if config.smoothing_window % 2 == 0:
            raise ValueError("Smoothing window must be an odd integer.")
        self.config = config
        sam_config = Sam2Config(
            checkpoint=config.model_checkpoint,
            device=config.device,
            config_file=config.sam_config_file,
        )
        try:
            self.segmenter = Sam2Segmenter(sam_config)
        except SamModelNotAvailableError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "SAM 2 could not be initialised. Ensure the 'sam2' package is installed and the checkpoint path is correct."
            ) from exc

    def track(self, video_path: Path) -> Trajectory:
        measurements: List[Tuple[int, float, np.ndarray, np.ndarray]] = []
        frame_size: Optional[Tuple[int, int]] = None
        prev_left_mask: Optional[np.ndarray] = None
        prev_right_mask: Optional[np.ndarray] = None

        with VideoReader(video_path, sample_rate=self.config.sample_rate) as reader:
            for frame in reader:
                frame_size = (frame.image.shape[1], frame.image.shape[0])
                detections = self._detect_plates(frame.image)
                left_detection, right_detection = self._assign_detections(
                    detections, prev_left_mask, prev_right_mask, frame.image.shape
                )

                left_centroid = (
                    np.array(left_detection.centroid, dtype=np.float32)
                    if left_detection
                    else np.array([np.nan, np.nan], dtype=np.float32)
                )
                right_centroid = (
                    np.array(right_detection.centroid, dtype=np.float32)
                    if right_detection
                    else np.array([np.nan, np.nan], dtype=np.float32)
                )
                measurements.append(
                    (frame.index, frame.timestamp, left_centroid, right_centroid)
                )

                prev_left_mask = left_detection.mask if left_detection else None
                prev_right_mask = right_detection.mask if right_detection else None

        if frame_size is None:
            raise RuntimeError("Video contained no frames.")

        trajectory = Trajectory.from_measurements(
            measurements=measurements,
            frame_size=frame_size,
            metadata={
                "video_path": str(video_path),
                "sample_rate": str(self.config.sample_rate),
            },
        )
        return trajectory.with_smoothing(self.config.smoothing_window)

    def _detect_plates(self, frame_bgr: np.ndarray) -> List[PlateDetection]:
        mask_records = self.segmenter.generate(frame_bgr)
        _, frame_w = frame_bgr.shape[:2]
        detections: List[PlateDetection] = []
        for record in mask_records:
            mask = record["segmentation"]
            if mask.dtype != np.uint8:
                mask = mask.astype(np.uint8)
            if mask.max() <= 1:
                mask = mask * 255
            area = float(np.count_nonzero(mask))
            if area <= 0:
                continue
            x, y, w, h = record["bbox"]
            diameter = max(w, h)
            rel_diameter = diameter / frame_w
            if rel_diameter < self.config.min_plate_diameter or rel_diameter > self.config.max_plate_diameter:
                continue
            circularity = _estimate_circularity(mask)
            if circularity < self.config.min_circularity:
                continue
            centroid = _centroid_from_mask(mask)
            if centroid is None:
                continue
            detections.append(
                PlateDetection(
                    mask=(mask > 0).astype(np.uint8),
                    centroid=centroid,
                    bbox=(x, y, w, h),
                    area=area,
                )
            )
        return detections

    def _assign_detections(
        self,
        detections: Sequence[PlateDetection],
        prev_left_mask: Optional[np.ndarray],
        prev_right_mask: Optional[np.ndarray],
        frame_shape: Tuple[int, int, int],
    ) -> Tuple[Optional[PlateDetection], Optional[PlateDetection]]:
        if not detections:
            return None, None

        _, frame_w = frame_shape[:2]
        sorted_detections = sorted(detections, key=lambda det: det.area, reverse=True)

        if prev_left_mask is None or prev_right_mask is None:
            top_two = sorted_detections[:2]
            if not top_two:
                return None, None
            if len(top_two) == 1:
                det = top_two[0]
                return det if det.centroid[0] < frame_w / 2 else None, det if det.centroid[0] >= frame_w / 2 else None
            left_det, right_det = sorted(top_two, key=lambda det: det.centroid[0])
            return left_det, right_det

        left_candidate = _best_match(prev_left_mask, detections)
        right_candidate = _best_match(prev_right_mask, detections)

        if left_candidate is None or right_candidate is None or left_candidate == right_candidate:
            # fallback to positional assignment
            left_det, right_det = _fallback_by_position(sorted_detections, frame_w)
            return left_det, right_det

        return left_candidate, right_candidate


def _best_match(reference_mask: np.ndarray, detections: Sequence[PlateDetection]) -> Optional[PlateDetection]:
    best_det: Optional[PlateDetection] = None
    best_iou = 0.0
    for det in detections:
        iou = _mask_iou(reference_mask, det.mask)
        if iou > best_iou:
            best_iou = iou
            best_det = det
    if best_iou < 0.2:
        return None
    return best_det


def _fallback_by_position(detections: Sequence[PlateDetection], frame_width: int) -> Tuple[Optional[PlateDetection], Optional[PlateDetection]]:
    if not detections:
        return None, None
    left_det: Optional[PlateDetection] = None
    right_det: Optional[PlateDetection] = None
    for det in detections:
        if det.centroid[0] < frame_width / 2:
            if left_det is None:
                left_det = det
        else:
            if right_det is None:
                right_det = det
    return left_det, right_det


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    mask_a_bool = mask_a.astype(bool)
    mask_b_bool = mask_b.astype(bool)
    intersection = np.logical_and(mask_a_bool, mask_b_bool).sum()
    union = np.logical_or(mask_a_bool, mask_b_bool).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def _centroid_from_mask(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    moments = cv2.moments(mask, binaryImage=True)
    if moments["m00"] == 0:
        return None
    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]
    return float(cx), float(cy)


def _estimate_circularity(mask: np.ndarray) -> float:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area <= 0:
        return 0.0
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0.0
    circularity = 4 * math.pi * area / (perimeter ** 2)
    return float(circularity)
