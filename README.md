# Gym Assistant

A minimal example for tracking the trajectory of a barbell plate using [SAM 2.1 Large](https://huggingface.co/facebook/sam2.1-hiera-large).

## Features
- Generate a coordinate grid preview so you can read off approximate plate locations in pixel space.
- Initialize SAM 2.1 tracking from a user-provided point that seeds the segmentation mask.
- Export the plate trajectory as a CSV table along with annotated overlays, mask previews, and per-rep trajectory plots that span full down-up or up-down cycles.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Make sure the SAM 2.1 checkpoint is available in the local Hugging Face cache (the repo id `facebook/sam2.1-hiera-large` is used by default).

## Usage
Run the CLI on a squat or deadlift side-view video:
```bash
python scripts/run_tracking.py videos/squat.mov --output outputs/squat --preview
```

The script saves a `coordinate_grid.jpg` next to the outputs to help you determine the plate coordinates. When prompted, enter the plate center as `x,y`, or pass it non-interactively via `--point X Y`. After you confirm the point, the tracker renders `mask_preview.png` so you can verify the segmented plate before processing the rest of the video. Results include:
- `trajectory.csv`: frame index, time (s), and pixel coordinates of the plate center.
- `mask_preview.png`: the SAM 2 mask overlay and computed centroid for the selected plate.
- `trajectory_overlay.mov` (or `.mp4` if the input is MP4): the original video with each repetition highlighted while previous reps fade out but the live tracking marker remains on the current frame.
- `trajectory_plot_rep_*.png`: per-repetition trajectory plots that include the 2D plate path and vertical position over time for each move.

Set `--device cuda` to run inference on GPU when available.
