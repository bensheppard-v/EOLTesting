# LiDAR Target Sampling with Gimbal Control

## Overview

This project models FMCW LiDAR sampling on a flat rectangular target using gimbal micro/macro-stepping to achieve uniform coverage across all elevation beams.

### Key Concepts

**Sensor Configuration:**
- 128 vertical elevation beams organized into 8 channels (16 elevations per channel)
- 181 horizontal azimuth beams
- Fixed FOV: 45° horizontal × 30° vertical

**Sampling Goal:**
- Collect 300 samples per channel
- Equal representation from all 16 elevations → ~19 samples per elevation
- No overlapping samples (configurable minimum separation)

## How It Works

### 1. Calibration Phase

Before sampling, manually position the gimbal so that the **TOP** of the sensor's vertical FOV aligns with the **TOP** of the target. This position becomes the reference point (0, 0).

```python
# Compute calibration offset
dphi_calib, dtheta_calib = compute_calibration_offset(counter)
counter.set_calibration(dphi_calib, dtheta_calib)
```

### 2. Autofill Strategy

The algorithm adapts to target distance:

#### Close Targets (many elevations visible)
- **Microstep**: Make small gimbal adjustments within the visible elevation range
- Densify samples for all visible elevations simultaneously

#### Far Targets (few elevations visible)
- **Microstep**: Densify the few visible elevations first
- **Macrostep**: Make large vertical gimbal movements to bring new elevation groups onto target
- Repeat until all 16 elevations per channel have adequate samples

### 3. Three-Stage Scanning

The autofill process uses three stages:

1. **BASE FRAME**: Sample at calibration position (0, 0)
2. **MICROSTEPPING**: Fine azimuth/elevation offsets to densify visible elevations
3. **MACROSTEPPING**: Large elevation shifts to scan through all elevations

Each stage continues until all (channel, elevation) bins reach the target sample count.

## Usage Example

```python
from point_counter import FlatTargetHitCounter
import numpy as np

# Sensor resolution
AZIMUTH_RES_RAD = np.radians(45) / 180
ELEVATION_RES_RAD = np.radians(30) / 127

# Create counter
counter = FlatTargetHitCounter(
    target_width_m=1.8,
    target_height_m=1.3,
    distance_m=50.0,
    sensor_height_offset_m=0.0,
    sensor_width_offset_m=0.0,
    azimuth_res_rad=AZIMUTH_RES_RAD,
    elevation_res_rad=ELEVATION_RES_RAD,
)

# Calibrate
dphi_calib, dtheta_calib = compute_calibration_offset(counter)
counter.set_calibration(dphi_calib, dtheta_calib)

# Run autofill
offsets_rel, offsets_abs, samples, counts, channels, summary = \
    counter.autofill_per_channel_elevation(
        samples_per_bin=19,
        max_fine_subdiv=16,
        max_coarse_steps=20,
        spot_radius_m=0.01,
    )

# Export gimbal commands
counter.save_offsets_with_calibration(
    'gimbal_offsets.csv',
    offsets_rel,
    offsets_abs
)
```

## Output Files

### gimbal_offsets.csv
Contains both relative and absolute gimbal offsets:
- **Relative offsets**: From calibration position (used internally)
- **Absolute offsets**: Actual gimbal commands to execute (includes calibration)

Format:
```csv
Calibration Info
gimbal_dphi_calib_rad,gimbal_dtheta_calib_rad
0.05,0.12

Index,dphi_rel_rad,dtheta_rel_rad,dphi_abs_rad,dtheta_abs_rad
0,0.0,0.0,0.05,0.12
1,0.001,0.0,0.051,0.12
...
```

## Testing

Run the comprehensive test suite:

```bash
python test_point_counter.py
```

This validates:
- Channel assignment (8 channels)
- Elevation indexing (16 per channel)
- Autofill performance at various distances
- Calibration offset conversion

## Key Parameters

### `autofill_per_channel_elevation()`

- **samples_per_bin** (default 19): Target samples per (channel, elevation) bin
- **max_fine_subdiv** (default 16): Maximum microstepping subdivisions
- **max_coarse_steps** (default 20): Maximum macrostepping iterations
- **spot_radius_m** (default 0.01): Laser spot radius in metres. Spots closer than 2× radius are rejected
- **tolerance** (default 2): Allow bins to exceed target by this amount

## Distance-Dependent Behavior

| Distance | Elevations Visible | Strategy |
|----------|-------------------|----------|
| 5m       | Many (>10)        | Mostly microstepping |
| 25m      | Several (5-10)    | Mix of micro/macro |
| 50m      | Few (3-5)         | Heavy macrostepping |
| 100m     | Very few (1-3)    | Almost all macrostepping |

## Files

- **point_counter.py**: Main FlatTargetHitCounter class
- **example_usage.py**: Simple usage example
- **test_point_counter.py**: Comprehensive test suite
- **get_offsets.py**: Generate offsets for specific configuration
- **check_params.py**: Parameter validation utilities

## Notes

- Channels are organized **vertically** (not horizontally)
- Channel 0 = lowest elevations, Channel 7 = highest elevations
- Within each channel, elevation index 0 = lowest, 15 = highest
- Non-overlapping is enforced globally across all gimbal positions
- Horizontal (azimuth) representation is not prioritized - we only care about vertical coverage
