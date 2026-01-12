"""Visualize beam projection on target to verify math."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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

# Set calibration
calib = setup.compute_calibration_offset()
setup.set_calibration(calib)

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
        edgecolor='red', 
        linewidth=2,
        label='Target boundary'
    )
    ax.add_patch(target_rect)
    
    # Plot all beam intersections (gray for outside, blue for inside)
    ax.scatter(X[~mask], Y[~mask], c='lightgray', s=1, alpha=0.3, label='Beams missing target')
    ax.scatter(X[mask], Y[mask], c='blue', s=3, alpha=0.6, label='Beams hitting target')
    
    # Add grid lines to show beam spacing
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.2, linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.2, linewidth=0.5)
    
    # Labels and formatting
    ax.set_xlabel('Horizontal Position (m)', fontsize=10)
    ax.set_ylabel('Vertical Position (m)', fontsize=10)
    ax.set_title(f'{title}\n{n_h}H × {n_v}V = {n_h*n_v} beams', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.legend(fontsize=8, loc='upper right')
    
    # Set axis limits to show full FOV projection
    fov_width = 2 * setup.distance_m * np.tan(setup.hfov_rad / 2)
    fov_height = 2 * setup.distance_m * np.tan(setup.vfov_rad / 2)
    ax.set_xlim(-fov_width/2, fov_width/2)
    ax.set_ylim(-fov_height/2, fov_height/2)
    
    # Add text showing absolute gimbal angle
    textstr = f'Absolute angle: {np.degrees(dtheta_abs):.2f}°'
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, 
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('beam_projection_verification.png', dpi=150, bbox_inches='tight')
print("✓ Saved: beam_projection_verification.png")
plt.show()
