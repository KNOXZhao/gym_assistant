# Gym Assistant

A minimal example for tracking the trajectory of a barbell plate using [SAM 2.1 Large](https://huggingface.co/facebook/sam2.1-hiera-large).

## Features
- Automatically propose circular plate candidates on the first frame using OpenCV Hough circles.
- Run the SAM 2 video predictor to propagate the selected plate mask through the entire clip.
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

The script prints the detected plate candidates. Provide `--candidate-index` to skip the prompt in non-interactive runs. Results include:
- `trajectory.csv`: frame index, time (s), and pixel coordinates of the plate center.
- `trajectory_overlay.mp4`: the original video with the tracked path drawn on top.
- `trajectory_plot.png`: a static visualization of the path on the first frame.

Set `--device cuda` to run inference on GPU when available.
