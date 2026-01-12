"""Test to understand why macrostepping doesn't reach all channels."""

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

print("Calibration:", np.degrees(dtheta_calib), "degrees")

# Calculate channel height
theta_range = counter.elevation_angles().max() - counter.elevation_angles().min()
channel_height = theta_range / counter.num_channels

print(f"Channel height: {np.degrees(channel_height):.3f}°")
print(f"Testing downward macrosteps:\n")

# Test different macrostep levels
for step in range(0, 10):
    dtheta_macro = -step * channel_height
    
    # Project at this position
    X, Y, PHI, THETA = counter.project_with_angles(0.0, dtheta_macro)
    mask = counter.inside_mask(X, Y)
    
    if np.any(mask):
        hitting_thetas = THETA[mask].ravel()
        total_gimbal = dtheta_calib + dtheta_macro
        base_thetas = hitting_thetas - total_gimbal
        channels = counter.assign_to_channel(base_thetas)
        unique_ch = sorted(np.unique(channels).tolist())
        
        print(f"Step {step}: dtheta_macro={np.degrees(dtheta_macro):6.2f}° → "
              f"{hitting_thetas.size:3d} beams")
        print(f"  hitting_thetas range: {np.degrees(hitting_thetas.min()):.2f}° to {np.degrees(hitting_thetas.max()):.2f}°")
        print(f"  total_gimbal: {np.degrees(total_gimbal):.2f}°")
        print(f"  base_thetas range: {np.degrees(base_thetas.min()):.2f}° to {np.degrees(base_thetas.max()):.2f}°")
        print(f"  → Channels {unique_ch}\n")
    else:
        print(f"Step {step}: dtheta_macro={np.degrees(dtheta_macro):6.2f}° → NO HITS")
