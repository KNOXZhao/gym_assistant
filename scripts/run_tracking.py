#!/usr/bin/env python
"""CLI for generating a barbell plate trajectory using SAM 2.1."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


from gym_assistant.detection import PlateCandidate, candidate_from_coordinates
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
        "--point",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        help="Pixel coordinate to seed SAM2 tracking",
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
    if args.preview or args.point is None:
        tracker.save_coordinate_grid(frame, preview_path)
        print(f"Saved coordinate grid preview to {preview_path}")

    candidate: PlateCandidate
    if args.point is not None:
        try:
            candidate = candidate_from_coordinates(args.point)
        except ValueError as exc:
            print(f"Invalid point: {exc}", file=sys.stderr)
            return 2
    else:
        print("Enter the (x,y) coordinate for the plate you want to track.")
        while True:
            try:
                response = input("Coordinates: ").strip()
            except EOFError:
                print("No coordinates provided", file=sys.stderr)
                return 2

            if not response:
                print("Please enter the coordinate as 'x,y'.")
                continue

            parts = [p.strip() for p in response.split(",")]
            if len(parts) == 2:
                try:
                    point = tuple(float(v) for v in parts)
                except ValueError:
                    print("Invalid coordinate; please enter two numeric values.")
                    continue
                try:
                    candidate = candidate_from_coordinates(point)  # type: ignore[arg-type]
                except ValueError as exc:
                    print(f"Invalid point: {exc}")
                    continue
                break

            print("Unrecognized format. Use 'x,y'.")

    print("Tracking from point=(%.1f, %.1f)" % (candidate.center[0], candidate.center[1]))
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
