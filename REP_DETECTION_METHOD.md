# Rep Detection Method

## Problem with Previous Method

The original rep detection method had several critical issues:

1. **Too sensitive to noise**: Used a small smoothing window (5 frames) and low prominence thresholds
2. **Detected jitter as reps**: Minor oscillations (1-2 pixels) at rest positions were mistaken for complete repetitions
3. **No amplitude filtering**: Accepted any movement as a potential rep, regardless of magnitude

### Example from Deadlift Data
- At rest position: Y oscillates between 1089-1091 pixels (2-pixel jitter)
- Actual rep movement: Y moves from 1091 to 793 pixels (~300-pixel range)
- Old method: Detected dozens of "reps" from the 2-pixel jitter
- New method: Only detects the actual 300-pixel movement as reps

## New Detection Method

### Key Improvements

1. **Heavy Smoothing**
   - Window size: 15 frames (increased from 5)
   - Eliminates high-frequency noise and tracking jitter
   - Preserves actual movement patterns

2. **Amplitude-Based Filtering**
   - Minimum amplitude: 15% of total movement range
   - For a 300-pixel deadlift: minimum 45 pixels to count as a rep
   - Filters out all noise and minor adjustments

3. **Prominence-Based Peak Detection**
   - Minimum prominence: 50% of amplitude threshold (minimum 10 pixels)
   - Ensures only significant peaks are detected
   - Uses local region analysis (50-frame window) to calculate prominence

4. **Distance Constraint**
   - Minimum distance between peaks: 30 frames (~1 second at 30fps)
   - Prevents multiple detections of the same peak
   - Ensures temporal separation between reps

### Algorithm Steps

1. **Preprocessing**
   - Extract position data (X or Y, whichever has greater range)
   - Apply 15-frame moving average smoothing
   - Calculate total range of motion

2. **Peak Finding**
   - Find local maxima (rest/top positions)
   - Find local minima (bottom positions)
   - Both with prominence >= 50% of amplitude threshold
   - Apply minimum distance constraint

3. **Rep Segmentation**
   - A complete rep = start_extreme → opposite_extreme → start_extreme
   - Example: max → min → max (deadlift: top → bottom → top)
   - Calculate amplitude for each potential rep
   - Accept only if amplitude >= 15% of total range

4. **Validation**
   - Ensure minimum number of points per rep (10 frames)
   - Ensure segments don't overlap
   - Return chronologically ordered rep segments

## Results

### Deadlift Video
- **Total frames**: 567
- **Y range**: 793-1091 pixels (298-pixel span)
- **Amplitude threshold**: 44.7 pixels (15% of 298)
- **Detected**: 2 complete repetitions
  - Rep 1: Frames 0-302 (max → min @ frame 231 → max)
  - Rep 2: Frames 302-519 (max → min @ frame 411 → max)

### Comparison
| Metric | Old Method | New Method |
|--------|-----------|------------|
| Smoothing window | 5 frames | 15 frames |
| Amplitude threshold | 5 pixels (fixed) | 44.7 pixels (15% adaptive) |
| Prominence threshold | 1-2 pixels | 22.4 pixels |
| Reps detected (deadlift) | Many false positives | 2 (correct) |

## Usage

The new method is automatic and requires no parameter tuning. It adapts to:
- Different exercise types (vertical vs horizontal movement)
- Different video resolutions and distances
- Different movement amplitudes

### Parameters (with defaults)

```python
def _segment_repetitions(
    trajectory: Sequence[TrajectoryPoint],
    min_points: int = 10,           # Minimum frames per rep
    min_delta: float = 5.0,          # Absolute minimum amplitude (fallback)
)
```

The method automatically calculates:
- `amplitude_threshold = max(min_delta, 0.15 * total_range)`
- `prominence_threshold = max(amplitude_threshold * 0.5, 10.0)`
- `min_distance = 30 frames`

## Technical Details

### Peak Detection Algorithm

```python
def _find_peaks(values, prominence, is_max=True, min_distance=30):
    """
    Find peaks with prominence and distance constraints.
    
    For each point:
    1. Check if it's a local extremum (higher/lower than neighbors)
    2. Calculate prominence using 50-frame local window
    3. Accept if prominence >= threshold
    4. Filter by minimum distance, keeping higher peaks
    """
```

### Moving Average Implementation

```python
def _moving_average(values, window=15):
    """
    Apply moving average with edge padding.
    
    - Uses edge mode padding to handle boundaries
    - Returns same length as input
    - Preserves peak positions while smoothing noise
    """
```

## Future Improvements

Potential enhancements:
1. Adaptive window sizing based on video frame rate
2. Exercise-specific templates (deadlift vs squat patterns)
3. Velocity-based rep detection (detect acceleration/deceleration)
4. Machine learning-based pattern recognition
