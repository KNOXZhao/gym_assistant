# Gym Assistant

A minimal example for tracking the trajectory of a barbell plate using [SAM 2.1 Large](https://huggingface.co/facebook/sam2.1-hiera-large).

## Features
- Generate a coordinate grid preview so you can read off approximate plate locations in pixel space.
- Initialize SAM 2.1 tracking from manual coordinates (center + radius or bounding box) selected by the user.
- Export the plate trajectory as a CSV table along with an annotated video and static plot.

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

The script saves a `coordinate_grid.jpg` next to the outputs to help you determine the plate coordinates. When prompted, enter either `x1,y1,x2,y2` (bounding box) or `cx,cy,radius` (circle). You can also pass coordinates via the `--box` or `--center` arguments for non-interactive runs. Results include:
- `trajectory.csv`: frame index, time (s), and pixel coordinates of the plate center.
- `trajectory_overlay.mov` (or `.mp4` if the input is MP4): the original video with the tracked path drawn on top.
- `trajectory_plot.png`: a static visualization of the path on the first frame.

Set `--device cuda` to run inference on GPU when available.
