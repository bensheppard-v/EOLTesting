"""Check if it's physically possible to fit all required samples on target."""

target_width = 1.8  # meters
target_height = 1.3  # meters
target_area = target_width * target_height

spot_radius = 0.0135 / 2  # meters (diameter 13.5mm)
min_separation = 2 * spot_radius  # samples must be at least this far apart

samples_needed = 8 * 300  # 8 channels × 300 samples each

# Calculate max samples in a grid pattern
samples_horizontal = int(target_width / min_separation)
samples_vertical = int(target_height / min_separation)
max_samples_grid = samples_horizontal * samples_vertical

print("="*60)
print("PHYSICAL SPACE CONSTRAINT CHECK")
print("="*60)
print(f"Target size: {target_width}m × {target_height}m = {target_area:.2f} m²")
print(f"Spot diameter: {spot_radius*2*1000:.1f} mm")
print(f"Min separation: {min_separation*1000:.1f} mm")
print(f"Samples needed: {samples_needed}")
print()
print(f"Grid capacity: {samples_horizontal} × {samples_vertical} = {max_samples_grid} samples")
print()

if max_samples_grid >= samples_needed:
    print(f"✓ PHYSICALLY POSSIBLE")
    print(f"  Maximum capacity: {max_samples_grid:,} samples")
    print(f"  Required: {samples_needed:,} samples")
    print(f"  Headroom: {max_samples_grid - samples_needed:,} extra samples")
    print(f"  Space utilization: {samples_needed / max_samples_grid * 100:.1f}%")
else:
    print(f"✗ PHYSICALLY IMPOSSIBLE")
    print(f"  Can only fit: {max_samples_grid:,} samples")
    print(f"  Need: {samples_needed:,} samples")
    print(f"  Shortage: {samples_needed - max_samples_grid:,} samples")

print("="*60)
