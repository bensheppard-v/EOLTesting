"""Quick test to verify beam angle generation."""

import numpy as np
from new_point_counter import Test_Setup

# Create setup
setup = Test_Setup(
    target_width_m=1.8,
    target_height_m=1.3,
    distance_m=10.0,
    sensor_height_offset_m=0.0,
    sensor_width_offset_m=0.0,
    num_azimuth_beams=181,
    num_elevation_beams=128,
    samp_per_channel=300
)

print("Resolution Check:")
print(f"  Azimuth resolution: {np.degrees(setup.azimuth_res_rad):.4f}°")
print(f"  Expected: {45/(181-1):.4f}°")
print(f"  Elevation resolution: {np.degrees(setup.elevation_res_rad):.4f}°")
print(f"  Expected: {30/(128-1):.4f}°")
print()

# Test azimuth angles
phis = setup.azimuth_angles(0)
print("Azimuth Angles:")
print(f"  Count: {len(phis)}")
print(f"  Min: {np.degrees(phis[0]):.4f}° (expect -22.5°)")
print(f"  Max: {np.degrees(phis[-1]):.4f}° (expect +22.5°)")
print(f"  Spacing (first): {np.degrees(phis[1] - phis[0]):.4f}°")
print(f"  Spacing (last): {np.degrees(phis[-1] - phis[-2]):.4f}°")
print()

# Test elevation angles
thetas = setup.elevation_angles(0)
print("Elevation Angles:")
print(f"  Count: {len(thetas)}")
print(f"  Min: {np.degrees(thetas[0]):.4f}° (expect -15°)")
print(f"  Max: {np.degrees(thetas[-1]):.4f}° (expect +15°)")
print(f"  Spacing (first): {np.degrees(thetas[1] - thetas[0]):.4f}°")
print(f"  Spacing (last): {np.degrees(thetas[-1] - thetas[-2]):.4f}°")
print()

# Test projection at 10m with no offset
print("Projection Test (10m, no offset):")
X_vals = setup.distance_m * np.tan(phis)
Y_vals = setup.distance_m * np.tan(thetas)
print(f"  X range: {X_vals[0]:.3f}m to {X_vals[-1]:.3f}m")
print(f"  Y range: {Y_vals[0]:.3f}m to {Y_vals[-1]:.3f}m")
print(f"  X spacing at center: {X_vals[91] - X_vals[90]:.4f}m")
print(f"  Y spacing at center: {Y_vals[64] - Y_vals[63]:.4f}m")
