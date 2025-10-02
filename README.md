# Gym Assistant

A research-oriented toolkit for extracting barbell plate trajectories from training videos using the Segment Anything 2.1 Large model (SAM 2.1). The program processes a side-view video of barbell exercises (e.g. squats or deadlifts) and outputs the left/right plate trajectories so athletes can verify the bar path.

## Features

- Video ingestion with configurable frame sub-sampling for faster processing.
- SAM 2.1-based segmentation of barbell plates with geometric filtering to reject false positives.
- Temporal association of the detected plates to build smooth left/right trajectories.
- CSV export and static plot visualisation of the resulting trajectories.

## Installation

1. Create and activate a Python 3.10+ environment.
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Install SAM 2 (needed for inference):

```bash
pip install "git+https://github.com/facebookresearch/sam2.git"
```

4. Download the `sam2.1_hiera_large.pt` checkpoint from the [official release page](https://github.com/facebookresearch/sam2#model-checkpoints) and place it somewhere accessible.

## Usage

```bash
python -m gym_assistant.cli path/to/video.mp4 \
    --model-checkpoint /path/to/sam2.1_hiera_large.pt \
    --output-dir outputs \
    --device cuda:0
```

Key options:

- `--sample-rate`: process every Nth frame (default: `2`).
- `--min-plate-diameter` / `--max-plate-diameter`: reject masks outside the expected physical size range.
- `--smoothing-window`: length of the temporal moving average applied to trajectories (must be odd).

The command produces a CSV file with `(frame, timestamp, left_x, left_y, right_x, right_y)` columns and a PNG plot visualising the trajectories.

## Development

- Source code lives under `src/gym_assistant`.
- Run `python -m gym_assistant.cli --help` to inspect all CLI options.

## License

MIT License. See [LICENSE](LICENSE).
