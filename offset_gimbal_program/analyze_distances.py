"""
Visualize how sampling works at different distances.
Shows which channels are reachable and how micro/macro stepping fills them.
"""

import numpy as np
from point_counter import FlatTargetHitCounter

AZIMUTH_RES_RAD = np.radians(45) / 180
ELEVATION_RES_RAD = np.radians(30) / 127


def compute_calibration_offset(counter):
    """Compute gimbal offset to align TOP of FOV with TOP of target."""
    _, theta_half = counter._angular_limits()
    angle_to_top = np.arctan((counter.H / 2 - counter.sensor_height_offset_m) / counter.D)
    dtheta_calib = angle_to_top - (counter.theta0 + theta_half)
    return 0.0, dtheta_calib


def analyze_distance(distance_m):
    """Analyze sampling behavior at a specific distance."""
    print("\n" + "="*80)
    print(f"ANALYSIS AT {distance_m}m DISTANCE")
    print("="*80)
    
    counter = FlatTargetHitCounter(
        target_width_m=1.8,
        target_height_m=1.3,
        distance_m=distance_m,
        sensor_height_offset_m=0.0,
        sensor_width_offset_m=0.0,
        azimuth_res_rad=AZIMUTH_RES_RAD,
        elevation_res_rad=ELEVATION_RES_RAD,
    )
    
    # Calibrate
    dphi_calib, dtheta_calib = compute_calibration_offset(counter)
    counter.set_calibration(dphi_calib, dtheta_calib)
    print(f"Calibration: dtheta = {np.degrees(dtheta_calib):.2f}°")
    
    # Check what's visible at base position
    X, Y, PHI, THETA = counter.project_with_angles(0.0, 0.0)
    mask = counter.inside_mask(X, Y)
    visible_elevations = THETA[mask].size
    
    print(f"\nAt calibration position:")
    print(f"  - Beams hitting target: {visible_elevations}")
    
    # Run autofill
    offsets_rel, offsets_abs, samples, counts, channels, summary = \
        counter.autofill_per_channel_elevation(
            samples_per_bin=19,
            max_fine_subdiv=16,
            max_coarse_steps=20,
            spot_radius_m=0.0135/2,
        )
    
    print(f"\nAutofill results:")
    print(f"  - Reachable channels: {channels} ({len(channels)} of 8)")
    print(f"  - Gimbal positions used: {len(offsets_rel)}")
    print(f"  - Total samples: {summary['total_samples']}")
    print(f"  - Bins with target samples: {summary['bins_satisfied']}/{summary['total_bins']}")
    print(f"  - Completion rate: {summary['completion_rate']:.1%}")
    
    # Show per-channel breakdown
    if channels:
        print(f"\n  Per-channel breakdown:")
        for ch in channels:
            total = sum(counts[ch].values())
            non_zero = sum(1 for c in counts[ch].values() if c > 0)
            min_c = min(counts[ch].values())
            max_c = max(counts[ch].values())
            print(f"    Channel {ch}: {total:3d} samples across {non_zero:2d}/16 elevations "
                  f"(min={min_c:2d}, max={max_c:2d})")
    
    # Estimate micro vs macro stepping
    # Offsets with small dtheta are microsteps, large dtheta are macrosteps
    if len(offsets_rel) > 1:
        dthetas = [abs(dt) for _, dt in offsets_rel[1:]]  # Skip first (0,0)
        avg_dtheta = np.mean(dthetas) if dthetas else 0
        max_dtheta = max(dthetas) if dthetas else 0
        
        print(f"\n  Gimbal motion:")
        print(f"    Avg elevation offset: {np.degrees(avg_dtheta):.3f}°")
        print(f"    Max elevation offset: {np.degrees(max_dtheta):.3f}°")
        
        # Classify as micro vs macro
        threshold = np.radians(1.0)  # 1 degree threshold
        microsteps = sum(1 for dt in dthetas if dt < threshold)
        macrosteps = len(dthetas) - microsteps
        print(f"    Microsteps (<1°): {microsteps}")
        print(f"    Macrosteps (≥1°): {macrosteps}")


def main():
    """Analyze sampling at various distances."""
    print("="*80)
    print("DISTANCE-DEPENDENT SAMPLING ANALYSIS")
    print("="*80)
    print("\nTarget: 1.8m × 1.3m")
    print("Sensor: 45° H-FOV × 30° V-FOV, 8 channels × 16 elevations")
    print("Goal: 19 samples per elevation (300 per channel)")
    
    # Test at various distances
    distances = [5, 15, 25, 50, 75, 100]
    
    for dist in distances:
        analyze_distance(dist)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("Close range (5-15m):  Many elevations visible → mostly microstepping")
    print("Medium range (25-50m): Few elevations visible → mix of micro/macro")
    print("Far range (75-100m):   Very few elevations → heavy macrostepping")
    print("="*80)


if __name__ == "__main__":
    main()
