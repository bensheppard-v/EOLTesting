"""Show which specific elevations are missing samples."""

import numpy as np
from point_counter import FlatTargetHitCounter

AZIMUTH_RES_RAD = np.radians(45) / 180
ELEVATION_RES_RAD = np.radians(30) / 127

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
_, theta_half = counter._angular_limits()
angle_to_top = np.arctan((counter.H / 2) / counter.D)
dtheta_calib = angle_to_top - (counter.theta0 + theta_half)
counter.set_calibration(0.0, dtheta_calib)

# Run autofill
print("Running autofill...")
offsets_rel, offsets_abs, samples, counts, channels, summary = \
    counter.autofill_per_channel_elevation(
        samples_per_bin=19,
        max_fine_subdiv=32,
        max_coarse_steps=100,
        spot_radius_m=0.01,
        tolerance=2,
    )

print(f"\n{'='*80}")
print("ELEVATIONS WITH ZERO SAMPLES (the missing 25 bins)")
print(f"{'='*80}\n")

total_zero = 0
for ch in range(8):
    zero_elevs = [e for e in range(16) if counts[ch][e] == 0]
    if zero_elevs:
        print(f"Channel {ch}: Elevations {zero_elevs} have 0 samples ({len(zero_elevs)} missing)")
        total_zero += len(zero_elevs)

print(f"\nTotal elevations with 0 samples: {total_zero}")
print(f"Total satisfied bins: {summary['bins_satisfied']}")
print(f"Total bins: {summary['total_bins']}")
print(f"Expected missing: {128 - summary['bins_satisfied']}")
