"""
Simple example demonstrating how to use FlatTargetHitCounter for LiDAR sampling.

This shows the basic workflow:
1. Create counter with target and sensor parameters
2. Compute calibration offset (align top of FOV with top of target)
3. Run autofill to collect 300 samples per channel
4. Export results
"""

import numpy as np
from point_counter import FlatTargetHitCounter

# Sensor angular resolution (based on 181 azimuth × 128 elevation beams)
AZIMUTH_RES_RAD = np.radians(45) / 180      # ~0.0044 rad
ELEVATION_RES_RAD = np.radians(30) / 127    # ~0.0041 rad


def compute_calibration_offset(counter):
    """
    Compute gimbal offset to align HIGHEST beam (channel 0) with TOP of target.
    
    This enables unidirectional UPWARD stepping:
    - At calibration: Channel 0 hits top of target
    - Step UP: Channel 0 moves above target, channels 1, 2, 3... come into view
    
    Beam 127 (channel 0) has highest angle (+15° native).
    
    Returns (dphi_calib, dtheta_calib) in radians.
    """
    # Angle from sensor to TOP edge of target
    angle_to_top_of_target = np.arctan((counter.H / 2 - counter.sensor_height_offset_m) / counter.D)
    
    # The highest beam (beam 127, channel 0) is at: theta0 + theta_half
    _, theta_half = counter._angular_limits()
    highest_beam_native_angle = counter.theta0 + theta_half  # e.g., 0 + 15° = 15°
    
    # Tilt so highest beam points at TOP of target
    dtheta_calib = angle_to_top_of_target - highest_beam_native_angle
    dphi_calib = 0.0
    
    return dphi_calib, dtheta_calib


def main():
    # ==========================================================================
    # SETUP: Define target and sensor parameters
    # ==========================================================================
    
    TARGET_WIDTH_M = 1.8        # Target width (metres)
    TARGET_HEIGHT_M = 1.3       # Target height (metres)
    DISTANCE_M = 10.0           # Distance to target (metres)
    
    print("="*80)
    print("LiDAR Sampling Configuration")
    print("="*80)
    print(f"Target: {TARGET_WIDTH_M}m × {TARGET_HEIGHT_M}m at {DISTANCE_M}m distance")
    print(f"Sensor: 8 channels × 16 elevations = 128 total beams")
    print(f"Goal: 300 samples per channel (~19 samples per elevation)")
    print("="*80)
    
    # Create counter
    counter = FlatTargetHitCounter(
        target_width_m=TARGET_WIDTH_M,
        target_height_m=TARGET_HEIGHT_M,
        distance_m=DISTANCE_M,
        sensor_height_offset_m=0.0,     # Sensor at target center height
        sensor_width_offset_m=0.0,      # Sensor at target center width
        azimuth_res_rad=AZIMUTH_RES_RAD,
        elevation_res_rad=ELEVATION_RES_RAD,
        sensor_hfov_deg=45.0,
        sensor_vfov_deg=30.0,
    )
    
    # ==========================================================================
    # STEP 1: Calibration
    # ==========================================================================
    
    print("\nSTEP 1: Calibration")
    print("-" * 80)
    
    dphi_calib, dtheta_calib = compute_calibration_offset(counter)
    counter.set_calibration(dphi_calib, dtheta_calib)
    
    print(f"Calibration offset: dphi={np.degrees(dphi_calib):.3f}°, "
          f"dtheta={np.degrees(dtheta_calib):.3f}°")
    print("✓ Top of FOV now aligned with top of target")
    
    # ==========================================================================
    # STEP 2: Autofill
    # ==========================================================================
    
    print("\nSTEP 2: Autofill Sampling")
    print("-" * 80)
    
    offsets_rel, offsets_abs, samples, counts, channels, summary = \
        counter.autofill_per_channel_elevation(
            samples_per_bin=19,         # Target 19 samples per elevation
            max_fine_subdiv=32,         # Up to 32 microsteps (increased from 16)
            max_coarse_steps=100,       # Up to 100 macrosteps (increased from 50)
            spot_radius_m=0.01,         # 1cm spot radius (2cm min separation)
            tolerance=2,                # Allow up to 21 samples per bin
        )
    
    print(f"Reachable channels: {channels}")
    print(f"Total gimbal positions used: {len(offsets_rel)}")
    print(f"Total samples collected: {summary['total_samples']}")
    print(f"Bins satisfied: {summary['bins_satisfied']} / {summary['total_bins']}")
    print(f"Completion rate: {summary['completion_rate']:.1%}")
    
    # ==========================================================================
    # STEP 3: Inspect Results
    # ==========================================================================
    
    print("\nSTEP 3: Per-Channel Summary")
    print("-" * 80)
    
    for ch in channels:
        total_ch = sum(counts[ch].values())
        min_ch = min(counts[ch].values())
        max_ch = max(counts[ch].values())
        avg_ch = total_ch / 16
        print(f"Channel {ch}: Total={total_ch:3d}, "
              f"Min={min_ch:2d}, Max={max_ch:2d}, Avg={avg_ch:.1f}")
    
    # ==========================================================================
    # STEP 4: Export Results
    # ==========================================================================
    
    print("\nSTEP 4: Export Results")
    print("-" * 80)
    
    # Export gimbal offsets with calibration info
    counter.save_offsets_with_calibration(
        filename='gimbal_offsets.csv',
        offsets_relative=offsets_rel,
        offsets_absolute=offsets_abs
    )
    print("✓ Saved gimbal offsets to: gimbal_offsets.csv")
    
    # Also save in degrees for convenience
    counter.save_offsets_with_calibration(
        filename='gimbal_offsets_deg.csv',
        offsets_relative=[(np.degrees(dp), np.degrees(dt)) for dp, dt in offsets_rel],
        offsets_absolute=[(np.degrees(dp), np.degrees(dt)) for dp, dt in offsets_abs]
    )
    print("✓ Saved gimbal offsets (degrees) to: gimbal_offsets_deg.csv")
    
    # Export actual sample data
    import csv
    with open('sample_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x_m', 'y_m', 'channel', 'elevation'])
        # samples is a nested dict: {ch: {elev: [(x, y), ...]}}
        for ch in range(8):
            for elev in range(16):
                for (x, y) in samples[ch][elev]:
                    writer.writerow([x, y, ch, elev])
    print("✓ Saved sample data to: sample_data.csv")
    
    print("\n" + "="*80)
    print("✓ Complete! Use the gimbal offsets to scan your target.")
    print("="*80)


if __name__ == "__main__":
    main()
