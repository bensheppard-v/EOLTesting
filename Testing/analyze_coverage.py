"""Analyze coverage statistics for each channel and elevation."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from new_point_counter import Test_Setup

def analyze_coverage(distance_m, samp_per_channel=400):
    """
    Analyze coverage for a given distance.
    Returns sample counts per elevation beam.
    """
    print(f"\n{'='*60}")
    print(f"Analyzing coverage at {distance_m}m")
    print(f"{'='*60}")
    
    # Create setup
    setup = Test_Setup(
        target_width_m=1.8,
        target_height_m=1.3,
        distance_m=distance_m,
        sensor_height_offset_m=0.0,
        sensor_width_offset_m=0.0,
        num_azimuth_beams=181,
        num_elevation_beams=128,
        samp_per_channel=samp_per_channel,
        buffer_m=0.01,
        spot_diameter_m=0.0135
    )
    
    # Set calibration
    calib = setup.compute_calibration_offset()
    setup.set_calibration(calib)
    
    # Run test to get positions
    positions = setup.run_test()
    print(f"Total positions: {len(positions)}")
    
    # Track sample count for each elevation beam (128 total)
    # Index 0 = bottom-most beam, 127 = top-most beam
    sample_counts = np.zeros(128, dtype=int)
    
    # Target dimensions
    half_w = setup.target_width_m / 2.0
    half_h = setup.target_height_m / 2.0
    usable_half_w = half_w - setup.buffer_m
    usable_half_h = half_h - setup.buffer_m
    
    # For each position, count which beams hit the target
    for h_offset, v_offset in positions:
        phis = setup.azimuth_angles(h_offset)
        thetas = setup.elevation_angles(setup.gimbal_v_offset_rad + v_offset)
        
        PHI, THETA = np.meshgrid(phis, thetas)
        X = setup.distance_m * np.tan(PHI)
        Y = setup.distance_m * np.tan(THETA)
        
        # Check which beams hit the usable target area
        mask = (np.abs(X) <= usable_half_w) & (np.abs(Y) <= usable_half_h)
        
        # Count hits per elevation beam
        for elev_idx in range(128):
            if np.any(mask[elev_idx, :]):
                # Count how many azimuth beams hit for this elevation
                n_azimuth_hits = np.sum(mask[elev_idx, :])
                sample_counts[elev_idx] += n_azimuth_hits
    
    return sample_counts, setup, positions


def plot_coverage_analysis(distance_m, sample_counts):
    """
    Create comprehensive coverage plots.
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # Overall title
    fig.suptitle(f'Coverage Analysis at {distance_m}m', fontsize=16, fontweight='bold')
    
    # Calculate target samples per elevation
    target_samples = 19
    
    # Organize by channel (8 channels, 16 elevations each)
    channels = []
    for ch in range(8):
        ch_start = 128 - (ch + 1) * 16  # Top to bottom
        ch_end = 128 - ch * 16
        ch_samples = sample_counts[ch_start:ch_end]
        channels.append(ch_samples)
    
    # Channel colors
    channel_colors = ['#e41a1c', '#377eb8', '#4daf4a', "#91499c", 
                     '#ff7f00', '#ffff33', '#a65628', "#af5083"]
    
    # Plot 1: Bar chart by channel
    ax1 = plt.subplot(2, 2, 1)
    x_pos = np.arange(128)
    bars = ax1.bar(x_pos, sample_counts, width=1.0, edgecolor='black', linewidth=0.5)
    
    # Color by channel
    for ch in range(8):
        ch_start = 128 - (ch + 1) * 16
        ch_end = 128 - ch * 16
        for i in range(ch_start, ch_end):
            bars[i].set_facecolor(channel_colors[ch])
            bars[i].set_alpha(0.7)
    
    ax1.axhline(y=target_samples, color='red', linestyle='--', linewidth=2, label=f'Target: {target_samples} samples')
    ax1.set_xlabel('Elevation Beam Index', fontsize=11)
    ax1.set_ylabel('Sample Count', fontsize=11)
    ax1.set_title('Samples per Elevation Beam (All Channels)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add channel labels
    for ch in range(8):
        ch_start = 128 - (ch + 1) * 16
        ch_center = ch_start + 8
        ax1.text(ch_center, ax1.get_ylim()[1] * 0.95, f'Ch{ch}', 
                ha='center', va='top', fontsize=9, fontweight='bold',
                color=channel_colors[ch])
    
    # Plot 2: Individual channel breakdowns
    ax2 = plt.subplot(2, 2, 2)
    x_elev = np.arange(16)
    width = 0.1
    
    for ch in range(8):
        offset = (ch - 3.5) * width
        ch_samples = channels[ch]
        ax2.bar(x_elev + offset, ch_samples, width=width, 
               label=f'Ch{ch}', color=channel_colors[ch], alpha=0.7)
    
    ax2.axhline(y=target_samples, color='red', linestyle='--', linewidth=2, label=f'Target: {target_samples}')
    ax2.set_xlabel('Elevation Index (within channel)', fontsize=11)
    ax2.set_ylabel('Sample Count', fontsize=11)
    ax2.set_title('Samples per Elevation (by Channel)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_elev)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(ncol=3, fontsize=8)
    
    # Plot 3: Statistics per channel
    ax3 = plt.subplot(2, 2, 3)
    
    ch_means = [np.mean(ch_samples) for ch_samples in channels]
    ch_mins = [np.min(ch_samples) for ch_samples in channels]
    ch_maxs = [np.max(ch_samples) for ch_samples in channels]
    
    x_ch = np.arange(8)
    ax3.bar(x_ch, ch_means, color=channel_colors, alpha=0.7, label='Mean')
    ax3.scatter(x_ch, ch_mins, color='red', s=100, marker='v', label='Min', zorder=3)
    ax3.scatter(x_ch, ch_maxs, color='green', s=100, marker='^', label='Max', zorder=3)
    ax3.axhline(y=target_samples, color='red', linestyle='--', linewidth=2, label=f'Target: {target_samples}')
    
    ax3.set_xlabel('Channel', fontsize=11)
    ax3.set_ylabel('Sample Count', fontsize=11)
    ax3.set_title('Statistics per Channel', fontsize=12, fontweight='bold')
    ax3.set_xticks(x_ch)
    ax3.set_xticklabels([f'Ch{i}' for i in range(8)])
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend()
    
    # Plot 4: Coverage quality heatmap
    ax4 = plt.subplot(2, 2, 4)
    
    # Create 2D array: channels x elevations
    heatmap_data = np.array(channels)
    
    im = ax4.imshow(heatmap_data, aspect='auto', cmap='RdYlGn', 
                    vmin=0, vmax=target_samples*1.5)
    ax4.set_xlabel('Elevation Index (within channel)', fontsize=11)
    ax4.set_ylabel('Channel', fontsize=11)
    ax4.set_title('Coverage Heatmap (samples per elevation)', fontsize=12, fontweight='bold')
    ax4.set_yticks(range(8))
    ax4.set_yticklabels([f'Ch{i}' for i in range(8)])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('Sample Count', fontsize=10)
    cbar.ax.axhline(y=target_samples, color='red', linestyle='--', linewidth=2)
    
    # Annotate cells with values
    for ch in range(8):
        for elev in range(16):
            text = ax4.text(elev, ch, int(heatmap_data[ch, elev]),
                           ha="center", va="center", color="black", fontsize=7)
    
    plt.tight_layout()
    plt.savefig(f'coverage_analysis_{distance_m}m.png', dpi=150, bbox_inches='tight')
    print(f"Saved coverage_analysis_{distance_m}m.png")
    plt.close()
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"COVERAGE STATISTICS")
    print(f"{'='*60}")
    print(f"Target samples per elevation: {target_samples}")
    print(f"\nOverall:")
    print(f"  Total samples collected: {np.sum(sample_counts)}")
    print(f"  Mean samples per elevation: {np.mean(sample_counts[sample_counts > 0]):.1f}")
    print(f"  Min samples: {np.min(sample_counts[sample_counts > 0])}")
    print(f"  Max samples: {np.max(sample_counts)}")
    print(f"  Elevations with data: {np.sum(sample_counts > 0)}/128")
    
    print(f"\nPer Channel:")
    for ch in range(8):
        ch_samples = channels[ch]
        active_elevations = np.sum(ch_samples > 0)
        if active_elevations > 0:
            print(f"  Channel {ch}: Mean={np.mean(ch_samples[ch_samples > 0]):.1f}, "
                  f"Min={np.min(ch_samples[ch_samples > 0])}, "
                  f"Max={np.max(ch_samples)}, "
                  f"Active elevations={active_elevations}/16")
        else:
            print(f"  Channel {ch}: No data")
    
    # Check if target met
    print(f"\nTarget Achievement:")
    active_samples = sample_counts[sample_counts > 0]
    meeting_target = np.sum(active_samples >= target_samples)
    total_active = len(active_samples)
    print(f"  Elevations meeting target ({target_samples}+ samples): {meeting_target}/{total_active} ({100*meeting_target/total_active:.1f}%)")


# Test at multiple distances
if __name__ == "__main__":
    test_distances = [5, 10, 50, 70, 100, 150]
    
    for distance in test_distances:
        sample_counts, setup, positions = analyze_coverage(distance, samp_per_channel=400)
        plot_coverage_analysis(distance, sample_counts)
    
    print(f"\n{'='*60}")
    print(f"✓ Analysis complete for all distances")
    print(f"{'='*60}")
