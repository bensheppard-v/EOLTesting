"""Test script for the new Test_Setup class."""

import numpy as np
from new_point_counter import Test_Setup

# Create test setup - 1.8m x 1.3m target at 10m
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

print("="*70)
print("Test Setup - 10m Distance")
print("="*70)
print(f"Target: {setup.target_width_m}m × {setup.target_height_m}m")
print(f"Distance: {setup.distance_m}m")
print(f"Azimuth beams: 181")
print(f"Elevation beams: 128")
print()

# Compute and set calibration
print("Calibration:")
print("-"*70)
dphi_calib, dtheta_calib = setup.compute_calibration_offset()
print(f"Computed offset: dphi={np.degrees(dphi_calib):.3f}°, dtheta={np.degrees(dtheta_calib):.3f}°")
setup.set_calibration((dphi_calib, dtheta_calib))
print(f"✓ Calibration set")
print()

# Test at calibration position
print("Beam Count at Calibration Position (0, 0):")
print("-"*70)
n_h, n_v = setup.count_beams_on_target(0, 0)
print(f"Horizontal beams hitting: {n_h}")
print(f"Vertical beams hitting: {n_v}")
print(f"Total intersection points: {n_h * n_v}")
print()

# Test with some offsets
print("Beam Count with Offsets:")
print("-"*70)

test_offsets = [
    (0, np.radians(5)),   # +5° up
    (0, np.radians(-5)),  # -5° down
    (0, np.radians(10)),  # +10° up
]

for dphi, dtheta in test_offsets:
    n_h, n_v = setup.count_beams_on_target(dphi, dtheta)
    print(f"Offset (0°, {np.degrees(dtheta):+.1f}°): {n_h} horizontal × {n_v} vertical = {n_h * n_v} total")

print()
print("="*70)
print("✓ Tests complete!")
print("="*70)
print()

# Visualization
print("Generating visualization...")
print("-"*70)

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Beam Projection Verification at 10m Distance', fontsize=16, fontweight='bold')

# Test different gimbal positions
test_cases = [
    (0, 0, "Calibration Position (0°, 0°)"),
    (0, np.radians(5), "After +5° Tilt (UP)"),
    (0, np.radians(-5), "After -5° Tilt (DOWN)"),
    (0, np.radians(10), "After +10° Tilt (UP)"),
]

for idx, (dphi_rel, dtheta_rel, title) in enumerate(test_cases):
    ax = axes[idx // 2, idx % 2]
    
    # Convert to absolute angles
    dphi_abs = setup.gimbal_h_offset_rad + dphi_rel
    dtheta_abs = setup.gimbal_v_offset_rad + dtheta_rel
    
    # Generate beam angles
    phis = setup.azimuth_angles(dphi_abs)
    thetas = setup.elevation_angles(dtheta_abs)
    
    # Create meshgrid
    PHI, THETA = np.meshgrid(phis, thetas)
    
    # Project to target
    X = setup.distance_m * np.tan(PHI)
    Y = setup.distance_m * np.tan(THETA)
    
    # Check which hit target
    half_w = setup.target_width_m / 2.0
    half_h = setup.target_height_m / 2.0
    mask = (np.abs(X) <= half_w) & (np.abs(Y) <= half_h)
    
    # Count beams
    n_h = int(np.sum(np.any(mask, axis=0)))
    n_v = int(np.sum(np.any(mask, axis=1)))
    
    # Plot target boundary
    target_rect = Rectangle(
        (-half_w, -half_h), 
        setup.target_width_m, 
        setup.target_height_m,
        fill=False, 
        edgecolor='darkred', 
        linewidth=3,
        label='Target boundary'
    )
    ax.add_patch(target_rect)
    
    # Plot beams - only show hits for clarity
    ax.scatter(X[mask], Y[mask], c='darkblue', s=15, alpha=0.8, label='Beams hitting target', edgecolors='none')
    
    # Labels and formatting
    ax.set_xlabel('Horizontal Position (m)', fontsize=10)
    ax.set_ylabel('Vertical Position (m)', fontsize=10)
    ax.set_title(f'{title}\n{n_h}H × {n_v}V = {n_h*n_v} beams', fontsize=11)
    ax.set_aspect('equal')
    ax.legend(fontsize=8, loc='upper right')
    
    # Zoom in to 1.4x target size for better visibility
    zoom_factor = 1.4
    ax.set_xlim(-half_w * zoom_factor, half_w * zoom_factor)
    ax.set_ylim(-half_h * zoom_factor, half_h * zoom_factor)
    
    # Add text showing absolute gimbal angle
    textstr = f'Absolute angle: {np.degrees(dtheta_abs):.2f}°'
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, 
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('beam_projection_verification.png', dpi=150, bbox_inches='tight')
print("✓ Saved: beam_projection_verification.png")
plt.show()
