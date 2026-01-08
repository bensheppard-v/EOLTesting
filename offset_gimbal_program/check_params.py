"""Simple check of what parameters are being used."""
import numpy as np
import sys
sys.path.insert(0, './offset_gimbal_program')

# What the constants SHOULD be
AZIMUTH_RES_RAD_CORRECT = np.radians(45) / 180
ELEVATION_RES_RAD_CORRECT = np.radians(30) / 127

print("Expected resolution constants:")
print(f"  AZIMUTH_RES_RAD = {AZIMUTH_RES_RAD_CORRECT:.8f} rad = {np.degrees(AZIMUTH_RES_RAD_CORRECT):.6f}°")
print(f"  ELEVATION_RES_RAD = {ELEVATION_RES_RAD_CORRECT:.8f} rad = {np.degrees(ELEVATION_RES_RAD_CORRECT):.6f}°")

# Import and check what's actually in the test file
from test_point_counter import AZIMUTH_RES_RAD, ELEVATION_RES_RAD

print("\nActual values in test_point_counter.py:")
print(f"  AZIMUTH_RES_RAD = {AZIMUTH_RES_RAD:.8f} rad = {np.degrees(AZIMUTH_RES_RAD):.6f}°")
print(f"  ELEVATION_RES_RAD = {ELEVATION_RES_RAD:.8f} rad = {np.degrees(ELEVATION_RES_RAD):.6f}°")

if abs(AZIMUTH_RES_RAD - AZIMUTH_RES_RAD_CORRECT) < 1e-8:
    print("\n✓ Azimuth resolution is CORRECT")
else:
    print(f"\n❌ Azimuth resolution MISMATCH!")
    print(f"   Difference: {abs(AZIMUTH_RES_RAD - AZIMUTH_RES_RAD_CORRECT):.10f} rad")

if abs(ELEVATION_RES_RAD - ELEVATION_RES_RAD_CORRECT) < 1e-8:
    print("✓ Elevation resolution is CORRECT")
else:
    print(f"❌ Elevation resolution MISMATCH!")
    print(f"   Difference: {abs(ELEVATION_RES_RAD - ELEVATION_RES_RAD_CORRECT):.10f} rad")

# Now check what this gives us
from point_counter import FlatTargetHitCounter

counter = FlatTargetHitCounter(
    target_width_m=1.8,
    target_height_m=1.3,
    distance_m=5.0,
    sensor_height_offset_m=0.0,
    sensor_width_offset_m=0.0,
    azimuth_res_rad=AZIMUTH_RES_RAD,
    elevation_res_rad=ELEVATION_RES_RAD,
)

phis = counter.azimuth_angles()
thetas = counter.elevation_angles()

print(f"\nBeam generation results:")
print(f"  Azimuth beams: {len(phis)} (expected: 181)")
print(f"  Elevation beams: {len(thetas)} (expected: 128)")

if len(phis) == 181:
    print("  ✓ Azimuth beam count CORRECT")
else:
    print(f"  ❌ Azimuth beam count WRONG (off by {len(phis) - 181})")

if len(thetas) == 128:
    print("  ✓ Elevation beam count CORRECT")
else:
    print(f"  ❌ Elevation beam count WRONG (off by {len(thetas) - 128})")

# Check coverage at 5m
X, Y = counter.project_to_target()
mask = counter.inside_mask(X, Y)
x_span = X[mask].max() - X[mask].min() if mask.any() else 0

print(f"\nCoverage at 5m (no offset):")
print(f"  Beam hits: {mask.sum()}")
print(f"  X span: {x_span:.3f}m (target width: 1.8m)")

if x_span >= 1.7:  # Allow some margin
    print(f"  ✓ Coverage spans most of target width")
else:
    print(f"  ❌ Coverage only spans {x_span/1.8*100:.1f}% of target width!")
