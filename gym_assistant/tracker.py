"""Video tracking pipeline using SAM 2.1 for weight plates."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from sam2.sam2_video_predictor import SAM2VideoPredictor

from .detection import PlateCandidate


@dataclass
class TrajectoryPoint:
    """Represents a plate center in a specific frame."""

    frame_idx: int
    time_s: float
    x: float
    y: float


class PlateTracker:
    """High-level interface for tracking a barbell plate across a video."""

    def __init__(
        self,
        model_id: str = "facebook/sam2.1-hiera-large",
        device: Optional[str] = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_id = model_id
        self.predictor: Optional[SAM2VideoPredictor] = None

    def _ensure_predictor(self) -> SAM2VideoPredictor:
        if self.predictor is None:
            self.predictor = SAM2VideoPredictor.from_pretrained(self.model_id, device=self.device)
        return self.predictor

    @staticmethod
    def _video_metadata(video_path: str) -> Tuple[int, float]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Unable to open video: {video_path}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        cap.release()
        return frame_count, fps

    @staticmethod
    def load_first_frame(video_path: str) -> np.ndarray:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Unable to open video: {video_path}")
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RuntimeError("Failed to read the first frame from video")
        return frame

    @staticmethod
    def save_coordinate_grid(
        frame: np.ndarray,
        output_path: Path,
        grid_divisions: int = 10,
        color: Tuple[int, int, int] = (0, 255, 0),
    ) -> None:
        """Save an image with a coordinate grid overlaid on the frame."""

        output_path.parent.mkdir(parents=True, exist_ok=True)
        preview = frame.copy()
        height, width = preview.shape[:2]
        step_x = width / grid_divisions
        step_y = height / grid_divisions

        for i in range(grid_divisions + 1):
            x = int(round(i * step_x))
            y = int(round(i * step_y))

            # Draw vertical grid line and annotate the x-coordinate.
            cv2.line(preview, (x, 0), (x, height - 1), color, 1)
            if 0 <= x < width:
                label_pos = (min(x + 5, width - 80), 20)
                cv2.putText(
                    preview,
                    str(x),
                    label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )

            # Draw horizontal grid line and annotate the y-coordinate.
            cv2.line(preview, (0, y), (width - 1, y), color, 1)
            if 0 <= y < height:
                label_pos = (5, min(y + 15, height - 10))
                cv2.putText(
                    preview,
                    str(y),
                    label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        cv2.imwrite(str(output_path), preview)

    @staticmethod
    def candidate_from_box(box: Tuple[int, int, int, int]) -> PlateCandidate:
        """Create a plate candidate from a bounding box."""

        x1, y1, x2, y2 = box
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        radius = max(x2 - x1, y2 - y1) / 2.0
        if radius <= 0:
            raise ValueError("Bounding box must have positive width and height")
        return PlateCandidate(center=(cx, cy), radius=radius, score=radius)

    def run(
        self,
        video_path: str,
        candidate: PlateCandidate,
        output_dir: str,
        frame_idx: int = 0,
        obj_id: str = "plate",
        offload_video_to_cpu: bool = True,
    ) -> List[TrajectoryPoint]:
        predictor = self._ensure_predictor()
        state = predictor.init_state(
            video_path,
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=False,
        )

        box = candidate.to_box(self.load_first_frame(video_path).shape)
        predictor.add_new_points_or_box(state, frame_idx=frame_idx, obj_id=obj_id, box=box)

        frame_count, fps = self._video_metadata(video_path)
        if frame_count <= 0:
            raise RuntimeError("Unable to determine frame count for video")
        if fps <= 0:
            fps = 30.0

        obj_idx = None
        trajectory: List[TrajectoryPoint] = []
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir_path / "trajectory.csv"

        with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["frame_idx", "time_s", "x", "y"])
            for frame_idx_out, obj_ids, masks in predictor.propagate_in_video(state):
                if obj_idx is None:
                    obj_idx = obj_ids.index(obj_id)
                frame_mask = masks[obj_idx]
                if isinstance(frame_mask, torch.Tensor):
                    mask_np = frame_mask.detach().cpu().numpy()
                else:
                    mask_np = np.asarray(frame_mask)
                if mask_np.dtype != np.bool_:
                    mask_np = mask_np > 0.5
                ys, xs = np.nonzero(mask_np)
                if len(xs) == 0:
                    continue
                cx = float(xs.mean())
                cy = float(ys.mean())
                time_s = frame_idx_out / fps
                point = TrajectoryPoint(frame_idx=frame_idx_out, time_s=time_s, x=cx, y=cy)
                trajectory.append(point)
                writer.writerow([frame_idx_out, f"{time_s:.6f}", f"{cx:.2f}", f"{cy:.2f}"])

        self._export_visualizations(video_path, trajectory, output_dir_path)
        return trajectory

    def _export_visualizations(
        self,
        video_path: str,
        trajectory: Sequence[TrajectoryPoint],
        output_dir: Path,
    ) -> None:
        if not trajectory:
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        suffix = Path(video_path).suffix.lower() or ".mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_out_path = output_dir / f"trajectory_overlay{suffix}"
        writer = cv2.VideoWriter(str(video_out_path), fourcc, fps, (width, height))

        points_by_frame = {p.frame_idx: p for p in trajectory}
        path_points: List[Tuple[int, int]] = []

        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx in points_by_frame:
                pt = points_by_frame[frame_idx]
                path_points.append((int(pt.x), int(pt.y)))
            if len(path_points) > 1:
                cv2.polylines(frame, [np.array(path_points, dtype=np.int32)], False, (0, 0, 255), 2)
            if path_points:
                cx, cy = path_points[-1]
                cv2.circle(frame, (cx, cy), 6, (0, 255, 255), -1)
            writer.write(frame)
            frame_idx += 1

        cap.release()
        writer.release()

        # Save a static plot of the trajectory on top of the first frame
        first_frame = self.load_first_frame(video_path)
        plot_path = output_dir / "trajectory_plot.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        annotated = first_frame.copy()
        if path_points:
            cv2.polylines(annotated, [np.array(path_points, dtype=np.int32)], False, (0, 0, 255), 2)
            cv2.circle(annotated, path_points[0], 8, (0, 255, 0), -1)
            cv2.circle(annotated, path_points[-1], 8, (0, 0, 255), -1)
        cv2.imwrite(str(plot_path), annotated)


def trajectory_to_array(trajectory: Iterable[TrajectoryPoint]) -> np.ndarray:
    data = [(p.frame_idx, p.time_s, p.x, p.y) for p in trajectory]
    return np.array(data, dtype=np.float32)
