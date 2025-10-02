"""Plotting utilities for trajectories."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from .trajectory import Trajectory


COLORS = {
    "left": "#1f77b4",
    "right": "#d62728",
}


def plot_trajectory(trajectory: Trajectory, output_path: Path) -> None:
    """Generate a 2D plot of the plate trajectories."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    width, height = trajectory.frame_size

    left = trajectory.left_positions
    right = trajectory.right_positions

    valid_left = ~np.isnan(left[:, 0])
    valid_right = ~np.isnan(right[:, 0])

    if valid_left.any():
        ax.plot(left[valid_left, 0], left[valid_left, 1], label="Left plate", color=COLORS["left"], linewidth=2)
        ax.scatter(left[valid_left, 0], left[valid_left, 1], color=COLORS["left"], s=10)
    if valid_right.any():
        ax.plot(right[valid_right, 0], right[valid_right, 1], label="Right plate", color=COLORS["right"], linewidth=2)
        ax.scatter(right[valid_right, 0], right[valid_right, 1], color=COLORS["right"], s=10)

    ax.set_title("Barbell plate trajectories")
    ax.set_xlabel("X position (pixels)")
    ax.set_ylabel("Y position (pixels)")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
