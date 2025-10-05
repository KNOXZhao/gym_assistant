"""Video tracking pipeline using SAM 2.1 for weight plates."""
from __future__ import annotations

import csv
import math
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
    x_side: Optional[float] = None


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

    def _extract_plate_region(
        self,
        mask_np: np.ndarray,
        reference_center: tuple[float, float] | None = None,
    ) -> tuple[np.ndarray, tuple[float, float]] | None:
        """Select the plate region from a SAM mask.

        The raw SAM mask occasionally includes parts of the barbell. We refine the
        mask by (a) trimming pixels that extend far from the expected plate center
        and (b) selecting the most plate-like connected component. This keeps the
        circular plate while discarding elongated regions along the bar.
        """

        if mask_np.ndim != 2:
            return None

        mask_bool = mask_np.astype(np.float32) > 0.5
        if not np.any(mask_bool):
            return None

        if reference_center is not None:
            ys, xs = np.nonzero(mask_bool)
            distances = np.hypot(xs - reference_center[0], ys - reference_center[1])
            if len(distances) == 0:
                return None
            median_dist = float(np.median(distances))
            # Estimate radius assuming a roughly circular plate.
            radius_est = max(median_dist * 1.4142, 1.0)
            radius_limit = radius_est * 1.2
            keep = distances <= radius_limit
            trimmed_mask = np.zeros_like(mask_bool)
            trimmed_mask[ys[keep], xs[keep]] = True
            # Only accept the trimmed mask if it preserves a substantial area.
            if np.count_nonzero(trimmed_mask) >= max(50, int(np.count_nonzero(mask_bool) * 0.25)):
                mask_bool = trimmed_mask

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(mask_bool.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        if np.count_nonzero(closed) > 0:
            mask_uint8 = closed
        else:
            mask_uint8 = mask_bool.astype(np.uint8)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask_uint8, connectivity=8
        )

        if num_labels <= 1:
            final_mask = mask_uint8.astype(bool)
        else:
            best_label = None
            best_score = -np.inf
            for label in range(1, num_labels):
                area = stats[label, cv2.CC_STAT_AREA]
                if area < 30:
                    continue
                width = stats[label, cv2.CC_STAT_WIDTH]
                height = stats[label, cv2.CC_STAT_HEIGHT]
                if width <= 0 or height <= 0:
                    continue
                aspect_ratio = max(width, height) / max(1, min(width, height))
                extent = area / float(width * height)
                circularity_penalty = abs(np.log(aspect_ratio + 1e-6))
                distance_penalty = 0.0
                if reference_center is not None:
                    centroid_x, centroid_y = centroids[label]
                    distance = float(
                        np.hypot(centroid_x - reference_center[0], centroid_y - reference_center[1])
                    )
                    distance_penalty = distance / max(max(width, height), 1.0)
                score = area * (extent ** 0.5)
                score /= 1.0 + 0.5 * circularity_penalty + 0.3 * distance_penalty
                if score > best_score:
                    best_score = score
                    best_label = label

            if best_label is None:
                final_mask = mask_uint8.astype(bool)
            else:
                final_mask = labels == best_label

        ys_final, xs_final = np.nonzero(final_mask)
        if len(xs_final) == 0:
            return None

        cx = float(xs_final.mean())
        cy = float(ys_final.mean())
        return final_mask.astype(bool), (cx, cy)

    def _estimate_camera_angle(self, mask_bool: np.ndarray) -> Optional[float]:
        """Estimate camera yaw angle from the apparent ellipse of the plate mask.

        The camera is assumed to rotate around the vertical axis relative to a
        true side-on view. A circular plate therefore appears as an ellipse.
        The ratio between the minor and major axes of this ellipse is
        cos(theta), where theta is the yaw angle. Returns the angle in radians
        when it can be estimated reliably.
        """

        if mask_bool.dtype != bool:
            mask_bool = mask_bool.astype(bool)

        ys, xs = np.nonzero(mask_bool)
        if xs.size < 5:
            return None

        points = np.column_stack((xs, ys)).astype(np.float32)
        try:
            ellipse = cv2.fitEllipse(points)
        except cv2.error:
            return None

        (_, _), axes, _ = ellipse
        if axes[0] <= 0 or axes[1] <= 0:
            return None

        major = max(axes)
        minor = min(axes)
        if major <= 0:
            return None

        ratio = float(minor / major)
        ratio = float(np.clip(ratio, 0.01, 1.0))
        return math.acos(ratio)

    def _mask_to_center(
        self,
        mask_np: np.ndarray,
        reference_center: tuple[float, float] | None = None,
    ) -> tuple[float, float] | None:
        result = self._extract_plate_region(mask_np, reference_center=reference_center)
        if result is None:
            return None
        _, center = result
        return center

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
        result = self._extract_plate_region(
            mask_np, reference_center=candidate.center
        )
        if result is None:
            return None
        mask_bool, center = result
        if center is None:
            return None

        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame = self.load_first_frame(video_path)
        mask_uint8 = mask_bool.astype(np.uint8)
        mask_color = np.zeros_like(frame)
        mask_color[:, :, 2] = mask_uint8 * 255
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

        last_center: tuple[float, float] | None = candidate.center
        first_center: tuple[float, float] | None = None
        angle_samples: List[float] = []
        cos_angle = 1.0

        with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["frame_idx", "time_s", "x", "y", "x_side"])
            for frame_idx_out, obj_ids, masks in predictor.propagate_in_video(state):
                if obj_idx is None:
                    obj_idx = obj_ids.index(obj_id)
                frame_mask = masks[obj_idx]
                if isinstance(frame_mask, torch.Tensor):
                    mask_np = frame_mask.detach().cpu().numpy()
                else:
                    mask_np = np.asarray(frame_mask)
                mask_np = np.squeeze(mask_np)
                result = self._extract_plate_region(mask_np, reference_center=last_center)
                if result is None:
                    continue
                mask_bool, center = result
                if center is None:
                    continue
                last_center = center
                if first_center is None:
                    first_center = center

                angle_est = self._estimate_camera_angle(mask_bool)
                if angle_est is not None:
                    angle_samples.append(angle_est)
                    median_angle = float(np.median(angle_samples))
                    cos_angle = math.cos(median_angle)
                    cos_angle = float(np.clip(cos_angle, 0.1, 1.0))

                if first_center is not None:
                    delta_x = center[0] - first_center[0]
                    side_x = first_center[0] + delta_x / cos_angle
                else:
                    side_x = center[0]
                cx, cy = center
                time_s = frame_idx_out / fps
                point = TrajectoryPoint(
                    frame_idx=frame_idx_out,
                    time_s=time_s,
                    x=cx,
                    y=cy,
                    x_side=side_x,
                )
                trajectory.append(point)
                writer.writerow(
                    [
                        frame_idx_out,
                        f"{time_s:.6f}",
                        f"{cx:.2f}",
                        f"{cy:.2f}",
                        f"{side_x:.2f}",
                    ]
                )

        rep_segments = self._segment_repetitions(trajectory)
        side_angle_deg: Optional[float] = None
        if angle_samples:
            side_angle_deg = math.degrees(float(np.median(angle_samples)))
            angle_path = output_dir_path / "camera_angle_degrees.txt"
            angle_path.write_text(f"{side_angle_deg:.2f}\n", encoding="utf-8")

        self._export_visualizations(
            video_path,
            trajectory,
            rep_segments,
            output_dir_path,
            side_angle_deg=side_angle_deg,
        )
        return trajectory, rep_segments

    def _export_visualizations(
        self,
        video_path: str,
        trajectory: Sequence[TrajectoryPoint],
        rep_segments: Sequence[RepSegment],
        output_dir: Path,
        *,
        side_angle_deg: Optional[float] = None,
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

        self._export_rep_plots(rep_segments, output_dir, side_angle_deg=side_angle_deg)

    def _export_rep_plots(
        self,
        rep_segments: Sequence[RepSegment],
        output_dir: Path,
        *,
        side_angle_deg: Optional[float] = None,
    ) -> None:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return

        if not rep_segments:
            return

        for segment in rep_segments:
            if not segment.points:
                continue
            xs = [p.x_side if p.x_side is not None else p.x for p in segment.points]
            ys = [p.y for p in segment.points]
            times = [p.time_s for p in segment.points]
            plot_path = output_dir / f"trajectory_plot_rep_{segment.index + 1}.png"
            fig, (ax_path, ax_height) = plt.subplots(1, 2, figsize=(10, 4))

            ax_path.plot(xs, ys, color="tab:red", linewidth=2)
            ax_path.scatter(xs[0], ys[0], color="tab:green", label="start")
            ax_path.scatter(xs[-1], ys[-1], color="tab:red", label="end")
            if side_angle_deg is not None:
                ax_path.set_title(
                    f"Rep {segment.index + 1} side path (angle {side_angle_deg:.1f}°)"
                )
            else:
                ax_path.set_title(f"Rep {segment.index + 1} path")
            ax_path.set_xlabel("Side-view X (px)")
            ax_path.set_ylabel("Y (px)")
            ax_path.invert_yaxis()
            ax_path.set_aspect("equal", adjustable="box")
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
        """
        Segment trajectory into repetitions using a robust peak-detection method.
        
        New approach:
        1. Heavy smoothing to remove jitter/noise
        2. Calculate amplitude threshold as percentage of total range (much more robust)
        3. Use scipy-like peak finding with distance constraint to avoid multiple peaks
        4. Define rep as complete cycle from one rest position through movement and back
        """
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
        
        # Use dimension with most movement
        primary_vals = y_vals if y_range >= x_range else x_vals
        total_range = float(primary_vals.max() - primary_vals.min())
        
        # Heavy smoothing to eliminate noise - use larger window
        smoothed = self._moving_average(primary_vals, window=15)
        
        # Set amplitude threshold as 15% of total range - this filters out noise
        # For a deadlift with ~300px range, this means we need 45px amplitude minimum
        amplitude_threshold = max(min_delta, 0.15 * total_range)
        
        if total_range < min_delta:
            # No significant movement - treat as one segment
            return [
                RepSegment(
                    index=0,
                    start_frame=trajectory[0].frame_idx,
                    end_frame=trajectory[-1].frame_idx,
                    points=list(trajectory),
                )
            ]
        
        # Find all local maxima and minima with significant prominence
        # Prominence should be substantial to avoid noise
        prominence_threshold = max(amplitude_threshold * 0.5, 10.0)
        
        maxima = self._find_peaks(smoothed, prominence_threshold, is_max=True)
        minima = self._find_peaks(smoothed, prominence_threshold, is_max=False)
        
        if not maxima or not minima:
            # No clear peaks found
            return [
                RepSegment(
                    index=0,
                    start_frame=trajectory[0].frame_idx,
                    end_frame=trajectory[-1].frame_idx,
                    points=list(trajectory),
                )
            ]
        
        # Merge and sort turning points
        turning_points = sorted(
            [(idx, 'max') for idx in maxima] + [(idx, 'min') for idx in minima],
            key=lambda x: x[0]
        )
        
        # Build rep segments: each rep goes from one extreme through opposite extreme and back
        # E.g., for deadlift (vertical movement):
        # - Rest at top (max Y ~1091) -> lift down (min Y ~793) -> back to top (max Y ~1091)
        # This is one complete repetition
        
        rep_segments: List[RepSegment] = []
        i = 0
        
        while i < len(turning_points) - 2:
            start_idx, start_type = turning_points[i]
            middle_idx, middle_type = turning_points[i + 1]
            end_idx, end_type = turning_points[i + 2]
            
            # Valid rep: start and end should be same type, middle should be opposite
            if start_type == end_type and middle_type != start_type:
                # Calculate amplitude
                segment_vals = smoothed[start_idx:end_idx + 1]
                amplitude = float(segment_vals.max() - segment_vals.min())
                
                # Only accept if amplitude is significant
                if amplitude >= amplitude_threshold:
                    segment_points = list(trajectory[start_idx:end_idx + 1])
                    
                    if len(segment_points) >= min_points:
                        rep_segments.append(
                            RepSegment(
                                index=len(rep_segments),
                                start_frame=segment_points[0].frame_idx,
                                end_frame=segment_points[-1].frame_idx,
                                points=segment_points,
                            )
                        )
                        # Move past this rep
                        i += 2
                        continue
            
            # Move to next potential rep
            i += 1
        
        if not rep_segments:
            # Fallback: treat entire trajectory as one segment
            return [
                RepSegment(
                    index=0,
                    start_frame=trajectory[0].frame_idx,
                    end_frame=trajectory[-1].frame_idx,
                    points=list(trajectory),
                )
            ]
        
        return rep_segments

    def _find_peaks(
        self, 
        values: np.ndarray, 
        prominence: float, 
        is_max: bool = True,
        min_distance: int = 30
    ) -> List[int]:
        """
        Find peaks (maxima or minima) with given prominence and minimum distance.
        
        Args:
            values: 1D array of values
            prominence: Minimum prominence for a peak
            is_max: If True, find maxima; if False, find minima
            min_distance: Minimum distance between peaks (prevents multiple detections)
        
        Returns:
            List of peak indices
        """
        if values.size < 3:
            return []
        
        # For minima, invert the signal
        signal = values if is_max else -values
        
        peaks: List[int] = []
        
        # Simple peak detection: point higher than neighbors
        for idx in range(1, len(signal) - 1):
            if signal[idx] > signal[idx - 1] and signal[idx] > signal[idx + 1]:
                # Check prominence
                # Find the lowest point in the local region
                left_min = signal[max(0, idx - 50):idx].min() if idx > 0 else signal[idx]
                right_min = signal[idx + 1:min(len(signal), idx + 51)].min() if idx < len(signal) - 1 else signal[idx]
                local_min = min(left_min, right_min)
                prom = signal[idx] - local_min
                
                if prom >= prominence:
                    peaks.append(idx)
        
        # Filter peaks by minimum distance
        if len(peaks) <= 1:
            return peaks
        
        filtered_peaks = [peaks[0]]
        for peak_idx in peaks[1:]:
            if peak_idx - filtered_peaks[-1] >= min_distance:
                filtered_peaks.append(peak_idx)
            else:
                # Keep the higher peak
                if signal[peak_idx] > signal[filtered_peaks[-1]]:
                    filtered_peaks[-1] = peak_idx
        
        return filtered_peaks

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
    data = [
        (
            p.frame_idx,
            p.time_s,
            p.x_side if p.x_side is not None else p.x,
            p.y,
        )
        for p in trajectory
    ]
    return np.array(data, dtype=np.float32)
