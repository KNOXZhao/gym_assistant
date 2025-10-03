#!/usr/bin/env python
"""CLI for generating a barbell plate trajectory using SAM 2.1."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


from gym_assistant.detection import PlateCandidate
from gym_assistant.tracker import PlateTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=str, help="Path to the input video file")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs",
        help="Directory where the trajectory artifacts will be stored",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="facebook/sam2.1-hiera-large",
        help="Hugging Face model id for the SAM 2 video predictor",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device to run inference on (default: auto)",
    )
    parser.add_argument(
        "--box",
        type=int,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Bounding box to initialize tracking (pixels)",
    )
    parser.add_argument(
        "--center",
        type=float,
        nargs=3,
        metavar=("CX", "CY", "RADIUS"),
        help="Center and radius to initialize tracking (pixels)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Save a preview image with a coordinate grid",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Video not found: {video_path}", file=sys.stderr)
        return 1

    tracker = PlateTracker(model_id=args.model_id, device=args.device)
    frame = tracker.load_first_frame(str(video_path))

    preview_path = Path(args.output) / "coordinate_grid.jpg"
    if args.preview or (args.box is None and args.center is None):
        tracker.save_coordinate_grid(frame, preview_path)
        print(f"Saved coordinate grid preview to {preview_path}")

    candidate: PlateCandidate
    if args.box is not None:
        try:
            candidate = tracker.candidate_from_box(tuple(args.box))
        except ValueError as exc:
            print(f"Invalid bounding box: {exc}", file=sys.stderr)
            return 2
    elif args.center is not None:
        cx, cy, radius = args.center
        if radius <= 0:
            print("Radius must be positive.", file=sys.stderr)
            return 2
        candidate = PlateCandidate(center=(float(cx), float(cy)), radius=float(radius), score=float(radius))
    else:
        print(
            "Enter coordinates for the plate you want to track. "
            "Provide either 'x1,y1,x2,y2' for a bounding box or 'cx,cy,radius' for a circle.",
        )
        while True:
            try:
                response = input("Coordinates: ").strip()
            except EOFError:
                print("No coordinates provided", file=sys.stderr)
                return 2

            if not response:
                print("Please enter coordinates in one of the supported formats.")
                continue

            parts = [p.strip() for p in response.split(",")]
            if len(parts) == 4:
                try:
                    box = tuple(int(float(v)) for v in parts)
                except ValueError:
                    print("Invalid bounding box; please enter four numeric values.")
                    continue
                try:
                    candidate = tracker.candidate_from_box(box)  # type: ignore[arg-type]
                except ValueError as exc:
                    print(f"Invalid bounding box: {exc}")
                    continue
                break
            if len(parts) == 3:
                try:
                    cx, cy, radius = (float(v) for v in parts)
                except ValueError:
                    print("Invalid circle specification; please enter three numeric values.")
                    continue
                if radius <= 0:
                    print("Radius must be positive.")
                    continue
                candidate = PlateCandidate(center=(cx, cy), radius=radius, score=radius)
                break

            print("Unrecognized format. Use 'x1,y1,x2,y2' or 'cx,cy,radius'.")

    print(
        "Tracking with center=(%.1f, %.1f) and radius=%.1f"
        % (candidate.center[0], candidate.center[1], candidate.radius)
    )
    trajectory = tracker.run(str(video_path), candidate, args.output)
    if not trajectory:
        print("No trajectory points generated", file=sys.stderr)
        return 4

    print(f"Trajectory saved to {Path(args.output) / 'trajectory.csv'}")
    overlay_suffix = Path(video_path).suffix.lower() or ".mp4"
    overlay_name = f"trajectory_overlay{overlay_suffix}"
    print(f"Overlay video saved to {Path(args.output) / overlay_name}")
    print(f"Plot saved to {Path(args.output) / 'trajectory_plot.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
