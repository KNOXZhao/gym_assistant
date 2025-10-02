"""Command line interface for the gym assistant application."""

from __future__ import annotations

import argparse
from pathlib import Path

from .plate_tracker import PlateTracker, PlateTrackerConfig
from .visualization import plot_trajectory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate barbell plate trajectories from a training video using SAM 2.1",
    )
    parser.add_argument("video", type=Path, help="Path to the input video file.")
    parser.add_argument(
        "--model-checkpoint",
        type=Path,
        required=True,
        help="Path to the SAM 2.1 Large checkpoint file (e.g. sam2.1_large.pt).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where artefacts (CSV + plots) will be written.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device to run inference on (e.g. 'cuda', 'cuda:0', 'mps', 'cpu').",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=2,
        help="Process every Nth frame from the video to speed up inference.",
    )
    parser.add_argument(
        "--min-plate-diameter",
        type=float,
        default=0.15,
        help=(
            "Minimum relative diameter of plate masks expressed as a fraction of the frame "
            "width. Helps reject spurious detections."
        ),
    )
    parser.add_argument(
        "--max-plate-diameter",
        type=float,
        default=0.6,
        help="Maximum relative diameter of plate masks (fraction of frame width).",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=5,
        help="Odd-sized moving average window for smoothing the trajectory coordinates.",
    )
    parser.add_argument(
        "--skip-visualization",
        action="store_true",
        help="Disable the generation of the trajectory plot image.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = PlateTrackerConfig(
        model_checkpoint=args.model_checkpoint,
        device=args.device,
        sample_rate=args.sample_rate,
        min_plate_diameter=args.min_plate_diameter,
        max_plate_diameter=args.max_plate_diameter,
        smoothing_window=args.smoothing_window,
    )

    tracker = PlateTracker(config)
    trajectory = tracker.track(args.video)

    trajectory_path = output_dir / f"{args.video.stem}_trajectory.csv"
    trajectory.to_csv(trajectory_path)

    if not args.skip_visualization:
        plot_path = output_dir / f"{args.video.stem}_trajectory.png"
        plot_trajectory(trajectory, plot_path)


if __name__ == "__main__":
    main()
