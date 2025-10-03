#!/usr/bin/env python
"""CLI for generating a barbell plate trajectory using SAM 2.1."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


from gym_assistant.detection import detect_plate_candidates
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
        "--candidate-index",
        type=int,
        default=None,
        help="Index of the detected plate candidate to track (default: prompt)",
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
        "--max-candidates",
        type=int,
        default=5,
        help="Maximum number of candidates to display",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Save a preview image with candidate overlays",
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
    candidates = detect_plate_candidates(frame, max_candidates=args.max_candidates)

    if not candidates:
        print("No plate candidates detected automatically. Please provide --candidate-index with manual coordinates.", file=sys.stderr)
        return 2

    if args.preview:
        preview_path = Path(args.output) / "candidate_preview.jpg"
        tracker.save_preview_image(frame, candidates, preview_path)
        print(f"Saved candidate preview to {preview_path}")

    print("Detected candidates:")
    for idx, candidate in enumerate(candidates):
        box = candidate.to_box(frame.shape)
        print(
            f"  [{idx}] center=({candidate.center[0]:.1f}, {candidate.center[1]:.1f}), "
            f"radius={candidate.radius:.1f}, box={box}"
        )

    candidate_index = args.candidate_index
    if candidate_index is None:
        try:
            response = input("Select candidate index to track [0]: ").strip()
            candidate_index = int(response) if response else 0
        except (EOFError, ValueError):
            candidate_index = 0
    if not 0 <= candidate_index < len(candidates):
        print(f"Invalid candidate index {candidate_index}; expected 0..{len(candidates) - 1}", file=sys.stderr)
        return 3

    candidate = candidates[candidate_index]
    print(f"Tracking candidate {candidate_index}...")
    trajectory = tracker.run(str(video_path), candidate, args.output)
    if not trajectory:
        print("No trajectory points generated", file=sys.stderr)
        return 4

    print(f"Trajectory saved to {Path(args.output) / 'trajectory.csv'}")
    print(f"Overlay video saved to {Path(args.output) / 'trajectory_overlay.mp4'}")
    print(f"Plot saved to {Path(args.output) / 'trajectory_plot.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
