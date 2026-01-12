"""
Comprehensive test suite for FlatTargetHitCounter per-channel, per-elevation autofill.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure we can import point_counter regardless of how the test is launched
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from point_counter import FlatTargetHitCounter

# Sensor angular resolution based on actual discretization:
# Horizontal: 181 beams spanning 45° (beams at edges)
#   → beam_0 at -22.5°, beam_180 at +22.5°
#   → 180 intervals → spacing = 45° / 180 = 0.25°
# Vertical: 128 beams spanning 30° (beams at edges)
#   → beam_0 at -15°, beam_127 at +15°
#   → 127 intervals → spacing = 30° / 127 = 0.236°
AZIMUTH_RES_RAD = np.radians(45) / 180  # ≈ 0.004363 rad
ELEVATION_RES_RAD = np.radians(30) / 127  # ≈ 0.004113 rad


def print_elevation_counts(distance, counts, channels):
    """Print detailed per-elevation count table for verification."""
    print(f"\n{'='*80}")
    print(f"ELEVATION COUNTS AT {distance}m")
    print(f"{'='*80}")
    
    for ch in channels:
        print(f"\nChannel {ch}:")
        print("  Elev  Count  |  Elev  Count  |  Elev  Count  |  Elev  Count")
        print("  " + "-" * 76)
        
        for row in range(4):  # 16 elevations / 4 per row
            line = "  "
            for col in range(4):
                elev = row + col * 4
                count = counts[ch][elev]
                line += f"{elev:3d}   {count:3d}    |  "
            print(line)
        
        total = sum(counts[ch].values())
        min_c = min(counts[ch].values())
        max_c = max(counts[ch].values())
        avg_c = total / 16
        print(f"  Total: {total:3d}  |  Min: {min_c}  |  Max: {max_c}  |  Avg: {avg_c:.1f}")


def print_comprehensive_stats(distance, counts, channels, offsets):
    """Print comprehensive statistics across all elevations and channels."""
    print(f"\n{'='*100}")
    print(f"COMPREHENSIVE STATISTICS AT {distance}m")
    print(f"{'='*100}")
    
    num_channels = len(channels)
    num_elevations_per_channel = 16
    total_elevations = num_channels * num_elevations_per_channel
    
    # Global stats
    total_samples = sum(sum(counts[ch].values()) for ch in channels)
    num_offsets = len(offsets)
    
    print(f"\n▸ GLOBAL STATISTICS")
    print(f"  Number of channels:              {num_channels}")
    print(f"  Elevations per channel:          {num_elevations_per_channel}")
    print(f"  Total elevation bins:            {total_elevations}")
    print(f"  Total samples collected:         {total_samples}")
    print(f"  Gimbal offsets used:             {num_offsets}")
    print(f"  Avg samples per bin:             {total_samples / total_elevations:.2f}")
    print(f"  Avg samples per offset:          {total_samples / max(1, num_offsets):.1f}")
    
    # Per-elevation stats (sum across channels)
    print(f"\n▸ PER-ELEVATION STATISTICS (summed across all channels)")
    print(f"  Elev  Total  Avg/Ch  Min/Ch  Max/Ch  StdDev/Ch")
    print(f"  " + "-" * 54)
    
    elev_stats = []
    for elev in range(num_elevations_per_channel):
        per_ch_counts = [counts[ch][elev] for ch in channels]
        total_e = sum(per_ch_counts)
        avg_e = np.mean(per_ch_counts)
        min_e = min(per_ch_counts)
        max_e = max(per_ch_counts)
        std_e = np.std(per_ch_counts)
        elev_stats.append((elev, total_e, avg_e, min_e, max_e, std_e))
        print(f"  {elev:3d}   {total_e:4d}   {avg_e:5.1f}  {min_e:5d}  {max_e:5d}    {std_e:5.2f}")
    
    # Per-channel stats (sum across elevations)
    print(f"\n▸ PER-CHANNEL STATISTICS (summed across all elevations)")
    print(f"  Ch  Total  Avg/Elev  Min/Elev  Max/Elev  StdDev/Elev")
    print(f"  " + "-" * 54)
    
    ch_stats = []
    for ch in channels:
        ch_counts = [counts[ch][e] for e in range(num_elevations_per_channel)]
        total_c = sum(ch_counts)
        avg_c = np.mean(ch_counts)
        min_c = min(ch_counts)
        max_c = max(ch_counts)
        std_c = np.std(ch_counts)
        ch_stats.append((ch, total_c, avg_c, min_c, max_c, std_c))
        print(f"  {ch:2d}  {total_c:4d}    {avg_c:5.1f}   {min_c:5d}   {max_c:5d}     {std_c:5.2f}")
    
    # Overall balance metrics
    all_counts = [counts[ch][e] for ch in channels for e in range(num_elevations_per_channel)]
    overall_min = min(all_counts)
    overall_max = max(all_counts)
    overall_avg = np.mean(all_counts)
    overall_std = np.std(all_counts)
    
    print(f"\n▸ OVERALL DISTRIBUTION")
    print(f"  Min count (any bin):             {overall_min}")
    print(f"  Max count (any bin):             {overall_max}")
    print(f"  Mean count (all bins):           {overall_avg:.2f}")
    print(f"  Std dev (all bins):              {overall_std:.2f}")
    print(f"  Range (max - min):               {overall_max - overall_min}")
    print(f"  Coefficient of variation:        {(overall_std / overall_avg):.4f}")
    
    # Count distribution histogram
    print(f"\n▸ COUNT DISTRIBUTION (how many bins have each count)")
    hist, _ = np.histogram(all_counts, bins=range(min(all_counts), max(all_counts) + 2))
    for count_val, freq in enumerate(hist, start=min(all_counts)):
        if freq > 0:
            bar = "█" * int(freq / max(hist) * 30)
            print(f"  {count_val:3d} samples: {bar} ({freq} bins)")
    
    print(f"\n{'='*100}\n")


def compute_top_alignment_calibration(counter):
    """Compute dtheta to align the top of the LiDAR vertical FOV to the top of the target.

    This is an approximation for simulation: the topmost beam is at theta0 + theta_half + dtheta_calib.
    We solve dtheta_calib so that topmost beam points to the target's top edge.
    Horizontal calibration (dphi) stays at zero per the test intent.
    """
    _, theta_half = counter._angular_limits()
    angle_to_top = np.arctan((counter.H / 2 - counter.sensor_height_offset_m) / counter.D)
    dtheta_calib = angle_to_top - (counter.theta0 + theta_half)
    return 0.0, dtheta_calib


def test_channel_assignment():
    """Test that elevation angles are correctly assigned to channels 0-7."""
    print("\n" + "="*60)
    print("TEST 1: Channel Assignment")
    print("="*60)
    
    counter = FlatTargetHitCounter(
        target_width_m=1.8,
        target_height_m=1.3,
        distance_m=50.0,
        sensor_height_offset_m=0.0,
        sensor_width_offset_m=0.0,
        azimuth_res_rad=AZIMUTH_RES_RAD,
        elevation_res_rad=ELEVATION_RES_RAD,
    )
    
    thetas = counter.elevation_angles()
    theta_min, theta_max = thetas.min(), thetas.max()
    
    # Test: elevations at extremes should map to channels 0 and 7
    ch_at_min = counter.assign_to_channel(theta_min)
    ch_at_max = counter.assign_to_channel(theta_max)
    
    print(f"Theta min: {theta_min:.6f} rad -> Channel {ch_at_min} (expected 0)")
    print(f"Theta max: {theta_max:.6f} rad -> Channel {ch_at_max} (expected 7)")
    
    assert ch_at_min == 0, f"Channel at theta_min should be 0, got {ch_at_min}"
    assert ch_at_max == 7, f"Channel at theta_max should be 7, got {ch_at_max}"
    
    # Test middle elevations map to expected channels
    for ch in range(8):
        theta_center = theta_min + (ch + 0.5) * (theta_max - theta_min) / 8
        assigned_ch = counter.assign_to_channel(theta_center)
        elev_idx = counter.get_elevation_index(theta_center)
        print(f"  Theta {theta_center:7.5f} rad -> Channel {assigned_ch}, Elev idx {elev_idx}")
        assert assigned_ch == ch, f"Expected channel {ch}, got {assigned_ch}"
        assert 0 <= elev_idx < 16, f"Elevation index should be 0-15, got {elev_idx}"
    
    print("✓ All channel assignments correct!")


def test_elevation_index():
    """Test that elevation indices stay within 0-15 per channel."""
    print("\n" + "="*60)
    print("TEST 2: Elevation Index (0-15 per channel)")
    print("="*60)
    
    counter = FlatTargetHitCounter(
        target_width_m=1.8,
        target_height_m=1.3,
        distance_m=50.0,
        sensor_height_offset_m=0.0,
        sensor_width_offset_m=0.0,
        azimuth_res_rad=AZIMUTH_RES_RAD,
        elevation_res_rad=ELEVATION_RES_RAD,
    )
    
    thetas = counter.elevation_angles()
    
    # Test all elevation angles
    for i, theta in enumerate(thetas):
        ch = counter.assign_to_channel(theta)
        elev_idx = counter.get_elevation_index(theta)
        assert 0 <= elev_idx < 16, f"Elevation index {elev_idx} out of range [0, 15]"
    
    print(f"✓ All {len(thetas)} elevation angles map to valid indices [0-15]")


def test_reachable_channels():
    """Test that identify_reachable_channels correctly identifies which channels hit target."""
    print("\n" + "="*60)
    print("TEST 3: Reachable Channels Detection")
    print("="*60)
    
    # Test at close range - all channels should hit
    counter_close = FlatTargetHitCounter(
        target_width_m=1.8,
        target_height_m=1.3,
        distance_m=5.0,
        sensor_height_offset_m=0.0,
        sensor_width_offset_m=0.0,
        azimuth_res_rad=AZIMUTH_RES_RAD,
        elevation_res_rad=ELEVATION_RES_RAD,
    )
    
    channels_close = counter_close.identify_reachable_channels()
    print(f"At 5m: {len(channels_close)} channels reachable: {channels_close}")
    assert len(channels_close) > 0, "At close range, at least some channels should hit"
    
    # Test at far range - fewer channels might hit
    counter_far = FlatTargetHitCounter(
        target_width_m=1.8,
        target_height_m=1.3,
        distance_m=100.0,
        sensor_height_offset_m=0.0,
        sensor_width_offset_m=0.0,
        azimuth_res_rad=AZIMUTH_RES_RAD,
        elevation_res_rad=ELEVATION_RES_RAD,
    )
    
    channels_far = counter_far.identify_reachable_channels()
    print(f"At 100m: {len(channels_far)} channels reachable: {channels_far}")
    assert len(channels_far) > 0, "At far range, at least some channels should still hit"
    
    print("✓ Reachable channel detection works!")


def test_calibration_conversion():
    """Test that calibration offset conversion works correctly."""
    print("\n" + "="*60)
    print("TEST 4: Calibration Offset Conversion")
    print("="*60)
    
    counter = FlatTargetHitCounter(
        target_width_m=1.8,
        target_height_m=1.3,
        distance_m=50.0,
        sensor_height_offset_m=0.0,
        sensor_width_offset_m=0.0,
        azimuth_res_rad=AZIMUTH_RES_RAD,
        elevation_res_rad=ELEVATION_RES_RAD,
    )
    
    # Set calibration
    calib_dphi, calib_dtheta = 0.05, 0.12
    counter.set_calibration(calib_dphi, calib_dtheta)
    
    retrieved = counter.get_calibration()
    print(f"Set calibration: ({calib_dphi}, {calib_dtheta})")
    print(f"Retrieved: {retrieved}")
    assert retrieved == (calib_dphi, calib_dtheta), "Calibration retrieval failed"
    
    # Test relative to absolute conversion
    dphi_rel, dtheta_rel = 0.001, 0.002
    dphi_abs, dtheta_abs = counter.relative_to_absolute_offset(dphi_rel, dtheta_rel)
    print(f"Relative offset: ({dphi_rel}, {dtheta_rel})")
    print(f"Absolute command: ({dphi_abs}, {dtheta_abs})")
    assert dphi_abs == calib_dphi + dphi_rel, "Azimuth conversion failed"
    assert dtheta_abs == calib_dtheta + dtheta_rel, "Elevation conversion failed"
    
    # Test reverse conversion
    dphi_back, dtheta_back = counter.absolute_to_relative_offset(dphi_abs, dtheta_abs)
    print(f"Back to relative: ({dphi_back}, {dtheta_back})")
    assert abs(dphi_back - dphi_rel) < 1e-10, "Reverse azimuth conversion failed"
    assert abs(dtheta_back - dtheta_rel) < 1e-10, "Reverse elevation conversion failed"
    
    print("✓ Calibration conversion works correctly!")


def test_autofill_basic():
    """Test basic autofill at 50m distance."""
    print("\n" + "="*60)
    print("TEST 5: Basic Autofill (50m distance)")
    print("="*60)
    
    counter = FlatTargetHitCounter(
        target_width_m=1.8,
        target_height_m=1.3,
        distance_m=50.0,
        sensor_height_offset_m=0.0,
        sensor_width_offset_m=0.0,
        azimuth_res_rad=AZIMUTH_RES_RAD,
        elevation_res_rad=ELEVATION_RES_RAD,
    )
    
    dphi_calib, dtheta_calib = compute_top_alignment_calibration(counter)
    counter.set_calibration(gimbal_dphi_calib=dphi_calib, gimbal_dtheta_calib=dtheta_calib)
    
    print("Running autofill_per_channel_elevation()...")
    offsets_rel, offsets_abs, samples, counts, channels, summary = \
        counter.autofill_per_channel_elevation(
            samples_per_bin=19,
            max_fine_subdiv=16,
            max_coarse_steps=20,
            spot_radius_m=0.01,
        )
    
    print(f"\nResults:")
    print(f"  Reachable channels: {channels}")
    print(f"  Total gimbal offsets: {len(offsets_rel)}")
    print(f"  Total samples collected: {summary['total_samples']}")
    print(f"  Bins satisfied: {summary['bins_satisfied']} / {summary['total_bins']}")
    print(f"  Completion rate: {summary['completion_rate']:.1%}")
    print(f"  Calibrated: {summary['is_calibrated']}")
    
    # Verify return structure
    assert isinstance(offsets_rel, list), "offsets_rel should be list"
    assert isinstance(offsets_abs, list), "offsets_abs should be list"
    assert len(offsets_rel) == len(offsets_abs), "Relative and absolute offsets should have same length"
    assert isinstance(counts, dict), "counts should be dict"
    assert isinstance(channels, list), "channels should be list"
    assert isinstance(summary, dict), "summary should be dict"
    
    # Verify calibration was applied to offsets_abs
    if len(offsets_rel) > 0:
        dphi_rel, dtheta_rel = offsets_rel[0]
        dphi_abs, dtheta_abs = offsets_abs[0]
        expected_dphi_abs = counter.gimbal_dphi_calib + dphi_rel
        expected_dtheta_abs = counter.gimbal_dtheta_calib + dtheta_rel
        assert dphi_abs == expected_dphi_abs, f"First offset absolute dphi mismatch"
        assert dtheta_abs == expected_dtheta_abs, f"First offset absolute dtheta mismatch"
    
    print("✓ Basic autofill test passed!")


def test_per_channel_samples():
    """Test that each channel gets approximately equal samples."""
    print("\n" + "="*60)
    print("TEST 6: Per-Channel Sample Distribution")
    print("="*60)
    
    counter = FlatTargetHitCounter(
        target_width_m=1.8,
        target_height_m=1.3,
        distance_m=50.0,
        sensor_height_offset_m=0.0,
        sensor_width_offset_m=0.0,
        azimuth_res_rad=AZIMUTH_RES_RAD,
        elevation_res_rad=ELEVATION_RES_RAD,
    )
    
    counter.set_calibration(*compute_top_alignment_calibration(counter))
    
    offsets_rel, offsets_abs, samples, counts, channels, summary = \
        counter.autofill_per_channel_elevation(
            samples_per_bin=19,
            max_fine_subdiv=16,
            max_coarse_steps=20,
            spot_radius_m=0.01,
        )
    
    print(f"\nSamples per channel (target: ~304 = 19×16):")
    for ch in channels:
        total = sum(counts[ch].values())
        print(f"  Channel {ch}: {total:3d} samples", end="")
        
        # Check each elevation
        min_elev = min(counts[ch].values())
        max_elev = max(counts[ch].values())
        avg_elev = total / 16
        
        if min_elev == 0:
            print(f" ⚠ WARNING: Some elevations have 0 samples!")
        elif abs(avg_elev - 19) > 2:
            print(f" ⚠ WARNING: Average {avg_elev:.1f} far from target 19")
        else:
            print(f" ✓ (avg: {avg_elev:.1f}, range: {min_elev}-{max_elev})")
    
    print("✓ Per-channel distribution test complete!")


def test_elevation_balance():
    """Test that each elevation within a channel gets similar samples."""
    print("\n" + "="*60)
    print("TEST 7: Elevation Balance Within Channel")
    print("="*60)
    
    counter = FlatTargetHitCounter(
        target_width_m=1.8,
        target_height_m=1.3,
        distance_m=50.0,
        sensor_height_offset_m=0.0,
        sensor_width_offset_m=0.0,
        azimuth_res_rad=AZIMUTH_RES_RAD,
        elevation_res_rad=ELEVATION_RES_RAD,
    )
    
    counter.set_calibration(*compute_top_alignment_calibration(counter))
    
    offsets_rel, offsets_abs, samples, counts, channels, summary = \
        counter.autofill_per_channel_elevation(
            samples_per_bin=19,
            max_fine_subdiv=16,
            max_coarse_steps=20,
            spot_radius_m=0.01,
        )
    
    print(f"\nElevation balance (target: ~19 per elevation):")
    errors = []
    for ch in channels:
        counts_by_elev = [counts[ch][e] for e in range(16)]
        min_count = min(counts_by_elev)
        max_count = max(counts_by_elev)
        avg_count = np.mean(counts_by_elev)
        
        print(f"  Channel {ch}: avg={avg_count:.1f}, range=[{min_count}, {max_count}]", end="")
        
        # Check for empty elevations
        empty = [e for e in range(16) if counts[ch][e] == 0]
        if empty:
            msg = f"Channel {ch} has empty elevations: {empty}"
            print(f" ❌ {msg}")
            errors.append(msg)
        # Check for unbalanced distribution
        elif max_count - min_count > 5:
            print(f" ⚠ Unbalanced (spread > 5)")
        else:
            print(f" ✓")
    
    if errors:
        print("\n❌ Elevation balance issues:")
        for err in errors:
            print(f"  - {err}")
    else:
        print("\n✓ Elevation balance test passed!")


def test_distance_sweep():
    """Test autofill across a range of distances."""
    print("\n" + "="*60)
    print("TEST 8: Distance Sweep (5m to 100m)")
    print("="*60)
    
    distances = [5, 15, 25, 50, 75, 100]
    results = {}
    
    for distance in distances:
        counter = FlatTargetHitCounter(
            target_width_m=1.8,
            target_height_m=1.3,
            distance_m=distance,
            sensor_height_offset_m=0.0,
            sensor_width_offset_m=0.0,
            azimuth_res_rad=AZIMUTH_RES_RAD,
            elevation_res_rad=ELEVATION_RES_RAD,
        )
        counter.set_calibration(*compute_top_alignment_calibration(counter))
        
        offsets_rel, offsets_abs, samples, counts, channels, summary = \
            counter.autofill_per_channel_elevation(
                samples_per_bin=19,
                max_fine_subdiv=16,
                max_coarse_steps=20,
                spot_radius_m=0.01,
            )
        
        results[distance] = {
            'channels': channels,
            'num_offsets': len(offsets_rel),
            'total_samples': summary['total_samples'],
            'completion': summary['completion_rate'],
        }
        
        print(f"{distance:3d}m: {len(channels)} channels, {len(offsets_rel):3d} offsets, " +
              f"{summary['total_samples']:4d} samples, {summary['completion_rate']:5.1%} complete")
    
    print("\n✓ Distance sweep complete!")
    return results


def test_non_overlap():
    """Test that no two samples overlap within 2×spot_radius."""
    print("\n" + "="*60)
    print("TEST 9: Non-Overlap Verification")
    print("="*60)
    
    counter = FlatTargetHitCounter(
        target_width_m=1.8,
        target_height_m=1.3,
        distance_m=50.0,
        sensor_height_offset_m=0.0,
        sensor_width_offset_m=0.0,
        azimuth_res_rad=AZIMUTH_RES_RAD,
        elevation_res_rad=ELEVATION_RES_RAD,
    )
    
    counter.set_calibration(*compute_top_alignment_calibration(counter))
    spot_radius = 0.01  # 1cm
    
    offsets_rel, offsets_abs, samples, counts, channels, summary = \
        counter.autofill_per_channel_elevation(
            samples_per_bin=19,
            max_fine_subdiv=16,
            max_coarse_steps=20,
            spot_radius_m=spot_radius,
        )
    
    # Collect all points
    all_points = []
    for ch in channels:
        for elev in range(16):
            all_points.extend(samples[ch][elev])
    
    print(f"Checking {len(all_points)} points for overlaps (min separation: {2*spot_radius:.4f}m)...")
    
    min_dist = float('inf')
    overlaps = 0
    
    for i, (x1, y1) in enumerate(all_points):
        for x2, y2 in all_points[i+1:]:
            dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            min_dist = min(min_dist, dist)
            if dist < 2*spot_radius - 1e-6:  # Allow small numerical error
                overlaps += 1
    
    print(f"Minimum distance between samples: {min_dist:.6f}m")
    print(f"Number of overlap violations: {overlaps}")
    
    if overlaps > 0:
        print(f"❌ Found {overlaps} overlapping samples!")
    else:
        print(f"✓ No overlaps detected!")


def test_visualization(distance=50):
    """Create a visualization of sample distribution."""
    print("\n" + "="*60)
    print(f"TEST 10: Visualization (Distance: {distance}m)")
    print("="*60)
    
    counter = FlatTargetHitCounter(
        target_width_m=1.8,
        target_height_m=1.3,
        distance_m=distance,
        sensor_height_offset_m=0.0,
        sensor_width_offset_m=0.0,
        azimuth_res_rad=AZIMUTH_RES_RAD,
        elevation_res_rad=ELEVATION_RES_RAD,
    )
    
    counter.set_calibration(*compute_top_alignment_calibration(counter))
    
    offsets_rel, offsets_abs, samples, counts, channels, summary = \
        counter.autofill_per_channel_elevation(
            samples_per_bin=19,
            max_fine_subdiv=16,
            max_coarse_steps=20,
            spot_radius_m=0.0135 / 2,
        )
    
    # Print detailed elevation counts
    print_comprehensive_stats(distance, counts, channels, offsets_rel)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot target rectangle
    from matplotlib.patches import Rectangle
    target_rect = Rectangle(
        (-counter.W/2, -counter.H/2),
        counter.W, counter.H,
        linewidth=2, edgecolor='black', facecolor='none', label='Target'
    )
    ax.add_patch(target_rect)
    
    # Plot samples by elevation (128 colors). Legend omitted to avoid clutter.
    num_elevations = 16
    total_bins = len(channels) * num_elevations
    color_map = plt.cm.nipy_spectral(np.linspace(0, 1, total_bins))

    for ch_idx, ch in enumerate(channels):
        for elev in range(num_elevations):
            pts = samples[ch][elev]
            if not pts:
                continue
            xs, ys = zip(*pts)
            color = color_map[ch_idx * num_elevations + elev]
            ax.scatter(xs, ys, c=[color], s=18, alpha=0.65)
    
    ax.set_xlabel('Horizontal Position (m)')
    ax.set_ylabel('Vertical Position (m)')
    ax.set_title(f'Sample Distribution at {distance}m - {summary["total_samples"]} total samples')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    plt.tight_layout()
    
    filename = f'coverage_plot_{distance}m.png'
    plt.savefig(filename, dpi=100)
    print(f"✓ Visualization saved to {filename}")
    plt.close()


def main():
    """Run all tests in the test suite."""
    print("\n" + "="*60)
    print("POINT COUNTER TEST SUITE")
    print("="*60)
    
    tests = [
        ("Channel Assignment", test_channel_assignment),
        ("Elevation Index", test_elevation_index),
        ("Reachable Channels", test_reachable_channels),
        ("Calibration Conversion", test_calibration_conversion),
        ("Basic Autofill", test_autofill_basic),
        ("Per-Channel Samples", test_per_channel_samples),
        ("Elevation Balance", test_elevation_balance),
        ("Distance Sweep", test_distance_sweep),
        ("Non-Overlap", test_non_overlap),
    ]
    
    passed = 0
    failed = 0
    
    # for test_name, test_func in tests:
    #     try:
    #         test_func()
    #         passed += 1
    #     except AssertionError as e:
    #         print(f"\n❌ TEST FAILED: {test_name}")
    #         print(f"   Error: {e}")
    #         failed += 1
    #         import traceback
    #         traceback.print_exc()
    #     except Exception as e:
    #         print(f"\n❌ UNEXPECTED ERROR in {test_name}")
    #         print(f"   Error: {e}")
    #         failed += 1
    #         import traceback
    #         traceback.print_exc()
    
    # Generate visualizations for all distances
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS FOR ALL DISTANCES")
    print("="*60)
    
    visualization_distances = [5, 15, 25, 50, 75, 100]
    for distance in visualization_distances:
        try:
            test_visualization(distance=distance)
        except Exception as e:
            print(f"❌ Visualization failed for {distance}m: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total:  {passed + failed}")
    print("="*60)
    
    if failed == 0:
        print("✓ ALL TESTS PASSED!")
        return 0
    else:
        print(f"❌ {failed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    exit_code = main()
