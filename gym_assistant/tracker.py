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


@dataclass
class RepSegment:
    """Container for a single repetition extracted from the trajectory."""

    index: int
    start_frame: int
    end_frame: int
    points: List[TrajectoryPoint]


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

    def _init_state(
        self,
        video_path: str,
        candidate: PlateCandidate,
        frame_idx: int = 0,
        obj_id: str = "plate",
        offload_video_to_cpu: bool = True,
    ) -> tuple[SAM2VideoPredictor, object]:
        predictor = self._ensure_predictor()
        state = predictor.init_state(
            video_path,
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=False,
        )

        point_coords, point_labels = candidate.point_prompt()
        predictor.add_new_points_or_box(
            state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=point_coords,
            labels=point_labels,
        )
        return predictor, state

    def _mask_to_center(self, mask_np: np.ndarray) -> tuple[float, float] | None:
        if mask_np.dtype != np.bool_:
            mask_np = mask_np > 0.5
        if mask_np.ndim != 2:
            return None
        ys, xs = np.nonzero(mask_np)
        if len(xs) == 0:
            return None
        cx = float(xs.mean())
        cy = float(ys.mean())
        return cx, cy

    def save_mask_preview(
        self,
        video_path: str,
        candidate: PlateCandidate,
        output_path: Path,
        frame_idx: int = 0,
        obj_id: str = "plate",
        offload_video_to_cpu: bool = True,
    ) -> tuple[float, float] | None:
        predictor, state = self._init_state(
            video_path,
            candidate,
            frame_idx=frame_idx,
            obj_id=obj_id,
            offload_video_to_cpu=offload_video_to_cpu,
        )

        mask_np: np.ndarray | None = None
        for frame_idx_out, obj_ids, masks in predictor.propagate_in_video(state):
            if obj_id not in obj_ids:
                continue
            if frame_idx_out != frame_idx:
                continue
            obj_idx = obj_ids.index(obj_id)
            frame_mask = masks[obj_idx]
            if isinstance(frame_mask, torch.Tensor):
                mask_np = frame_mask.detach().cpu().numpy()
            else:
                mask_np = np.asarray(frame_mask)
            break

        if mask_np is None:
            return None

        mask_np = np.squeeze(mask_np)
        center = self._mask_to_center(mask_np)
        if center is None:
            return None

        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame = self.load_first_frame(video_path)
        mask_bool = (mask_np > 0.5).astype(np.uint8)
        mask_color = np.zeros_like(frame)
        mask_color[:, :, 2] = mask_bool * 255
        preview = cv2.addWeighted(frame, 1.0, mask_color, 0.4, 0)
        cv2.circle(preview, (int(round(center[0])), int(round(center[1]))), 8, (0, 255, 255), -1)
        cv2.imwrite(str(output_path), preview)
        return center

    def run(
        self,
        video_path: str,
        candidate: PlateCandidate,
        output_dir: str,
        frame_idx: int = 0,
        obj_id: str = "plate",
        offload_video_to_cpu: bool = True,
    ) -> tuple[List[TrajectoryPoint], List[RepSegment]]:
        predictor, state = self._init_state(
            video_path,
            candidate,
            frame_idx=frame_idx,
            obj_id=obj_id,
            offload_video_to_cpu=offload_video_to_cpu,
        )

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
                mask_np = np.squeeze(mask_np)
                center = self._mask_to_center(mask_np)
                if center is None:
                    continue
                cx, cy = center
                time_s = frame_idx_out / fps
                point = TrajectoryPoint(frame_idx=frame_idx_out, time_s=time_s, x=cx, y=cy)
                trajectory.append(point)
                writer.writerow([frame_idx_out, f"{time_s:.6f}", f"{cx:.2f}", f"{cy:.2f}"])

        rep_segments = self._segment_repetitions(trajectory)
        self._export_visualizations(video_path, trajectory, rep_segments, output_dir_path)
        return trajectory, rep_segments

    def _export_visualizations(
        self,
        video_path: str,
        trajectory: Sequence[TrajectoryPoint],
        rep_segments: Sequence[RepSegment],
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

        rep_points: List[List[Tuple[int, int]]] = []
        rep_frame_bounds: List[Tuple[int, int]] = []

        if not rep_segments:
            rep_segments = [
                RepSegment(
                    index=0,
                    start_frame=trajectory[0].frame_idx,
                    end_frame=trajectory[-1].frame_idx,
                    points=list(trajectory),
                )
            ]

        for segment in rep_segments:
            pts = [(int(round(p.x)), int(round(p.y))) for p in segment.points]
            rep_points.append(pts)
            rep_frame_bounds.append((segment.start_frame, segment.end_frame))

        frame_idx = 0
        fade_frames = max(15, int(round(fps * 0.5)))
        trajectory_idx = 0
        latest_point: TrajectoryPoint | None = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            overlay = np.zeros_like(frame)
            current_rep: Optional[int] = None
            for idx, (start_f, end_f) in enumerate(rep_frame_bounds):
                if start_f <= frame_idx <= end_f:
                    current_rep = idx
                    break

            for idx, segment in enumerate(rep_segments):
                pts = rep_points[idx]
                if not pts:
                    continue
                start_f, end_f = rep_frame_bounds[idx]

                if frame_idx < start_f:
                    continue

                if current_rep == idx:
                    active_pts = [
                        (int(round(p.x)), int(round(p.y)))
                        for p in segment.points
                        if p.frame_idx <= frame_idx
                    ]
                    if len(active_pts) > 1:
                        cv2.polylines(
                            overlay,
                            [np.array(active_pts, dtype=np.int32)],
                            False,
                            (0, 0, 255),
                            2,
                        )
                    continue

                if frame_idx <= end_f:
                    partial_pts = [
                        (int(round(p.x)), int(round(p.y)))
                        for p in segment.points
                        if p.frame_idx <= frame_idx
                    ]
                    if len(partial_pts) > 1:
                        cv2.polylines(
                            overlay,
                            [np.array(partial_pts, dtype=np.int32)],
                            False,
                            (0, 0, 200),
                            2,
                        )
                    continue

                frames_since_end = frame_idx - end_f
                alpha = max(0.0, 1.0 - frames_since_end / fade_frames)
                if alpha <= 0.0:
                    continue
                color = (0, 0, int(round(200 * alpha)))
                cv2.polylines(
                    overlay,
                    [np.array(pts, dtype=np.int32)],
                    False,
                    color,
                    2,
                )

            while trajectory_idx < len(trajectory) and trajectory[trajectory_idx].frame_idx <= frame_idx:
                latest_point = trajectory[trajectory_idx]
                trajectory_idx += 1

            if latest_point is not None:
                cx = int(round(latest_point.x))
                cy = int(round(latest_point.y))
                cv2.circle(overlay, (cx, cy), 6, (0, 255, 255), -1)

            frame = cv2.addWeighted(frame, 1.0, overlay, 1.0, 0)
            writer.write(frame)
            frame_idx += 1

        cap.release()
        writer.release()

        self._export_rep_plots(rep_segments, output_dir)

    def _export_rep_plots(self, rep_segments: Sequence[RepSegment], output_dir: Path) -> None:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return

        if not rep_segments:
            return

        for segment in rep_segments:
            if not segment.points:
                continue
            xs = [p.x for p in segment.points]
            ys = [p.y for p in segment.points]
            times = [p.time_s for p in segment.points]
            plot_path = output_dir / f"trajectory_plot_rep_{segment.index + 1}.png"
            fig, (ax_path, ax_height) = plt.subplots(1, 2, figsize=(10, 4))

            ax_path.plot(xs, ys, color="tab:red", linewidth=2)
            ax_path.scatter(xs[0], ys[0], color="tab:green", label="start")
            ax_path.scatter(xs[-1], ys[-1], color="tab:red", label="end")
            ax_path.set_title(f"Rep {segment.index + 1} path")
            ax_path.set_xlabel("X (px)")
            ax_path.set_ylabel("Y (px)")
            ax_path.invert_yaxis()
            ax_path.legend(loc="best")

            ax_height.plot(times, ys, color="tab:blue", linewidth=2)
            ax_height.set_title("Vertical position over time")
            ax_height.set_xlabel("Time (s)")
            ax_height.set_ylabel("Y (px)")
            ax_height.invert_yaxis()
            ax_height.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

            fig.tight_layout()
            fig.savefig(str(plot_path))
            plt.close(fig)

    def _segment_repetitions(
        self,
        trajectory: Sequence[TrajectoryPoint],
        min_points: int = 10,
        min_delta: float = 5.0,
    ) -> List[RepSegment]:
        if not trajectory:
            return []

        if len(trajectory) <= 1:
            point = trajectory[0]
            return [RepSegment(index=0, start_frame=point.frame_idx, end_frame=point.frame_idx, points=list(trajectory))]

        data = trajectory_to_array(trajectory)
        x_vals = data[:, 2]
        y_vals = data[:, 3]
        x_range = float(x_vals.max() - x_vals.min())
        y_range = float(y_vals.max() - y_vals.min())
        primary_vals = y_vals if y_range >= x_range else x_vals

        smoothed = self._moving_average(primary_vals, window=5)
        delta = max(min_delta, 0.05 * float(smoothed.max() - smoothed.min()))
        if delta <= 0:
            return [
                RepSegment(
                    index=0,
                    start_frame=trajectory[0].frame_idx,
                    end_frame=trajectory[-1].frame_idx,
                    points=list(trajectory),
                )
            ]

        prominence = max(0.25 * min_delta, 0.01 * float(smoothed.max() - smoothed.min()), 1.0)

        turning_points = self._find_turning_points(smoothed, prominence)
        if not turning_points:
            return [
                RepSegment(
                    index=0,
                    start_frame=trajectory[0].frame_idx,
                    end_frame=trajectory[-1].frame_idx,
                    points=list(trajectory),
                )
            ]

        diffs = np.diff(smoothed)
        initial_sign = 0
        for diff in diffs:
            if abs(diff) >= 1e-3:
                initial_sign = 1 if diff > 0 else -1
                break

        turning_points = self._augment_turning_points(turning_points, smoothed, initial_sign)

        if len(turning_points) < 2:
            return [
                RepSegment(
                    index=0,
                    start_frame=trajectory[0].frame_idx,
                    end_frame=trajectory[-1].frame_idx,
                    points=list(trajectory),
                )
            ]

        rep_segments: List[RepSegment] = []
        idx = 0
        while idx < len(turning_points) - 1:
            start_sample, start_type = turning_points[idx]
            opposite_found = False
            end_idx = idx + 1

            while end_idx < len(turning_points):
                sample_idx, sample_type = turning_points[end_idx]
                if sample_type != start_type:
                    opposite_found = True
                elif opposite_found:
                    break
                end_idx += 1

            if end_idx >= len(turning_points):
                break

            end_sample, _ = turning_points[end_idx]
            if end_sample <= start_sample:
                idx += 1
                continue

            segment_points = list(trajectory[start_sample : end_sample + 1])
            if not segment_points:
                idx += 1
                continue

            window_values = smoothed[start_sample : end_sample + 1]
            amplitude = float(window_values.max() - window_values.min()) if window_values.size else 0.0

            if len(segment_points) >= min_points and amplitude >= delta:
                rep_segments.append(
                    RepSegment(
                        index=len(rep_segments),
                        start_frame=segment_points[0].frame_idx,
                        end_frame=segment_points[-1].frame_idx,
                        points=segment_points,
                    )
                )
                idx = end_idx
                continue

            idx += 1

        if not rep_segments:
            return [
                RepSegment(
                    index=0,
                    start_frame=trajectory[0].frame_idx,
                    end_frame=trajectory[-1].frame_idx,
                    points=list(trajectory),
                )
            ]

        return rep_segments

    def _augment_turning_points(
        self,
        turning_points: List[Tuple[int, str]],
        smoothed: np.ndarray,
        initial_sign: int,
    ) -> List[Tuple[int, str]]:
        if not turning_points:
            return []

        augmented = list(turning_points)
        first_idx, first_type = augmented[0]
        if first_idx > 0:
            if initial_sign > 0:
                start_type = "min"
            elif initial_sign < 0:
                start_type = "max"
            else:
                start_type = "min" if first_type == "max" else "max"
            augmented.insert(0, (0, start_type))

        last_idx, last_type = augmented[-1]
        if last_idx < len(smoothed) - 1:
            final_sign = 0
            for diff in reversed(np.diff(smoothed)):
                if abs(diff) >= 1e-3:
                    final_sign = 1 if diff > 0 else -1
                    break
            if final_sign < 0:
                end_type = "min"
            elif final_sign > 0:
                end_type = "max"
            else:
                end_type = "max" if last_type == "max" else "min"
            augmented.append((len(smoothed) - 1, end_type))

        cleaned: List[Tuple[int, str]] = []
        for idx, kind in augmented:
            if cleaned and cleaned[-1][1] == kind:
                prev_idx, _ = cleaned[-1]
                if kind == "max":
                    better = idx if smoothed[idx] >= smoothed[prev_idx] else prev_idx
                else:
                    better = idx if smoothed[idx] <= smoothed[prev_idx] else prev_idx
                cleaned[-1] = (better, kind)
            else:
                cleaned.append((idx, kind))

        return cleaned

    def _find_turning_points(self, values: np.ndarray, prominence: float) -> List[Tuple[int, str]]:
        if values.size < 3:
            return []

        turning_points: List[Tuple[int, str]] = []
        for idx in range(1, values.size - 1):
            prev_val = float(values[idx - 1])
            curr_val = float(values[idx])
            next_val = float(values[idx + 1])

            if curr_val >= prev_val and curr_val >= next_val:
                local_prom = curr_val - min(prev_val, next_val)
                if local_prom >= prominence:
                    turning_points.append((idx, "max"))
            if curr_val <= prev_val and curr_val <= next_val:
                local_prom = max(prev_val, next_val) - curr_val
                if local_prom >= prominence:
                    turning_points.append((idx, "min"))

        filtered: List[Tuple[int, str]] = []
        for idx, kind in turning_points:
            if not filtered:
                filtered.append((idx, kind))
                continue
            prev_idx, prev_kind = filtered[-1]
            if prev_kind == kind:
                if kind == "max":
                    better_idx = idx if values[idx] >= values[prev_idx] else prev_idx
                else:
                    better_idx = idx if values[idx] <= values[prev_idx] else prev_idx
                filtered[-1] = (better_idx, kind)
            else:
                filtered.append((idx, kind))

        return filtered

    @staticmethod
    def _moving_average(values: np.ndarray, window: int = 5) -> np.ndarray:
        if values.size == 0:
            return values
        if window <= 1 or values.size <= window:
            return values.astype(np.float32)
        kernel = np.ones(window, dtype=np.float32) / window
        padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
        smoothed = np.convolve(padded, kernel, mode="valid")
        return smoothed.astype(np.float32)

def trajectory_to_array(trajectory: Iterable[TrajectoryPoint]) -> np.ndarray:
    data = [(p.frame_idx, p.time_s, p.x, p.y) for p in trajectory]
    return np.array(data, dtype=np.float32)
