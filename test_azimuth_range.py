"""Check if azimuth stepping covers both directions."""
import numpy as np
import sys
sys.path.insert(0, './offset_gimbal_program')
from point_counter import FlatTargetHitCounter

AZIMUTH_RES_RAD = np.radians(45) / 180
ELEVATION_RES_RAD = np.radians(30) / 127

counter = FlatTargetHitCounter(
    target_width_m=1.8,
    target_height_m=1.3,
    distance_m=5.0,
    sensor_height_offset_m=0.0,
    sensor_width_offset_m=0.0,
    azimuth_res_rad=AZIMUTH_RES_RAD,
    elevation_res_rad=ELEVATION_RES_RAD,
)

# Set calibration
_, theta_half = counter._angular_limits()
angle_to_top = np.arctan((counter.H / 2 - counter.sensor_height_offset_m) / counter.D)
dtheta_calib = angle_to_top - (counter.theta0 + theta_half)
counter.set_calibration(0.0, dtheta_calib)

print("Running autofill at 5m...")
offsets_rel, offsets_abs, samples, counts, channels, summary = \
    counter.autofill_per_channel_elevation(
        samples_per_bin=19,
        max_fine_subdiv=16,
        max_coarse_steps=20,
        spot_radius_m=0.0135 / 2,
    )

print(f"\nTotal offsets used: {len(offsets_rel)}")
print(f"Total samples: {summary['total_samples']}")

# Analyze azimuth offset range
dphi_values = [dphi for dphi, _ in offsets_rel]
print(f"\nAzimuth offset range:")
print(f"  Min dphi: {min(dphi_values):.6f} rad = {np.degrees(min(dphi_values)):.3f}°")
print(f"  Max dphi: {max(dphi_values):.6f} rad = {np.degrees(max(dphi_values)):.3f}°")
print(f"  Unique dphi values: {len(set(dphi_values))}")

# Check if we have both negative and positive
negative_count = sum(1 for dphi in dphi_values if dphi < -1e-9)
zero_count = sum(1 for dphi in dphi_values if abs(dphi) < 1e-9)
positive_count = sum(1 for dphi in dphi_values if dphi > 1e-9)

print(f"\nDirection distribution:")
print(f"  Negative (left): {negative_count} offsets")
print(f"  Zero (center):   {zero_count} offsets")
print(f"  Positive (right): {positive_count} offsets")

if negative_count == 0:
    print("\n❌ ERROR: No negative azimuth offsets! Not scanning left.")
elif positive_count == 0:
    print("\n❌ ERROR: No positive azimuth offsets! Not scanning right.")
else:
    print("\n✓ Scanning in both directions")

# Check sample X positions
all_x = []
for ch in channels:
    for elev in range(16):
        for x, y in samples[ch][elev]:
            all_x.append(x)

if all_x:
    print(f"\nSample X positions:")
    print(f"  Min X: {min(all_x):.3f}m")
    print(f"  Max X: {max(all_x):.3f}m")
    print(f"  Span:  {max(all_x) - min(all_x):.3f}m")
    print(f"  Target width: 1.8m (±0.9m)")
    
    if max(all_x) < 0:
        print("\n❌ All samples on LEFT side of target!")
    elif min(all_x) > 0:
        print("\n❌ All samples on RIGHT side of target!")
    elif max(all_x) - min(all_x) < 1.5:
        print(f"\n⚠️  Coverage only {(max(all_x) - min(all_x))/1.8*100:.0f}% of target width")
    else:
        print("\n✓ Good coverage across target width")
