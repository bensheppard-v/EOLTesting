"""Debug script to see what's happening with channel assignment."""

import numpy as np
from point_counter import FlatTargetHitCounter

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

print("="*80)
print("DEBUGGING CHANNEL ASSIGNMENT")
print("="*80)

# Get elevation angles
thetas = counter.elevation_angles()
print(f"\nTotal elevation beams: {len(thetas)}")
print(f"Theta range: {np.degrees(thetas.min()):.2f}° to {np.degrees(thetas.max()):.2f}°")
print(f"theta0 (sensor center): {np.degrees(counter.theta0):.2f}°")

# Test a few angles
print("\nChannel assignment for key angles:")
print(f"  Max angle (+15°): {np.degrees(thetas.max()):.2f}° → Channel {counter.assign_to_channel(thetas.max())}")
print(f"  Center (0°):       {np.degrees(counter.theta0):.2f}° → Channel {counter.assign_to_channel(counter.theta0)}")
print(f"  Min angle (-15°): {np.degrees(thetas.min()):.2f}° → Channel {counter.assign_to_channel(thetas.min())}")

# Compute calibration
_, theta_half = counter._angular_limits()
angle_to_top = np.arctan((counter.H / 2 - counter.sensor_height_offset_m) / counter.D)
dtheta_calib = angle_to_top - (counter.theta0 + theta_half)

print(f"\nCalibration offset: {np.degrees(dtheta_calib):.3f}°")
counter.set_calibration(0.0, dtheta_calib)

# Now project at calibration position and see what hits
print("\n" + "="*80)
print("AT CALIBRATION POSITION (0, 0)")
print("="*80)

X, Y, PHI, THETA = counter.project_with_angles(0.0, 0.0)
mask = counter.inside_mask(X, Y)

if np.any(mask):
    hitting_thetas = THETA[mask].ravel()
    hitting_channels = counter.assign_to_channel(hitting_thetas)
    
    print(f"\nBeams hitting target: {hitting_thetas.size}")
    print(f"Angle range of hits: {np.degrees(hitting_thetas.min()):.2f}° to {np.degrees(hitting_thetas.max()):.2f}°")
    print(f"Channels hit: {sorted(np.unique(hitting_channels).tolist())}")
    
    # Show breakdown
    for ch in sorted(np.unique(hitting_channels)):
        count = np.sum(hitting_channels == ch)
        print(f"  Channel {ch}: {count} beams")
else:
    print("NO BEAMS HIT TARGET!")

print("\n" + "="*80)
