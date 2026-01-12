"""
Visualize LiDAR sampling results from gimbal_offsets.csv
Replays gimbal positions to reconstruct sample data for visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import csv
from point_counter import FlatTargetHitCounter

def load_gimbal_offsets(csv_path):
    """Load gimbal offsets from CSV file."""
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        
        # Skip to calibration section
        next(reader)  # "Calibration Info"
        next(reader)  # header
        calib_line = next(reader)
        dphi_calib = float(calib_line[0])
        dtheta_calib = float(calib_line[1])
        
        # Skip empty line and offsets header
        next(reader)
        next(reader)
        
        # Read all offset positions
        offsets = []
        for row in reader:
            if row:  # Skip empty lines
                dphi_rel = float(row[1])
                dtheta_rel = float(row[2])
                offsets.append((dphi_rel, dtheta_rel))
        
        return dphi_calib, dtheta_calib, offsets

def reconstruct_samples(counter, offsets, spot_radius_m=0.01):
    """
    Replay gimbal positions to reconstruct all samples.
    Returns: list of (x, y, channel, elevation) tuples
    """
    samples = []
    collected = set()  # Track (x, y) pairs to avoid duplicates
    min_sep2 = (2.0 * spot_radius_m) ** 2 if spot_radius_m > 0.0 else 0.0
    
    for dphi, dtheta in offsets:
        # Project beams at this gimbal position
        X, Y = counter.project_to_target(dphi, dtheta)
        mask = counter.inside_mask(X, Y)
        
        # Get valid hits
        xs = X[mask].ravel()
        ys = Y[mask].ravel()
        
        # Get beam indices and map to channels/elevations
        beam_indices = np.where(mask)
        elev_beam_idx = beam_indices[0]
        channels = elev_beam_idx // counter.elevations_per_channel
        elevs = elev_beam_idx % counter.elevations_per_channel
        
        # Collect unique samples
        for x, y, ch, elev in zip(xs, ys, channels, elevs):
            # Round to avoid floating point issues
            x_key = round(x, 6)
            y_key = round(y, 6)
            
            if (x_key, y_key) not in collected:
                # Check minimum spacing
                too_close = False
                if min_sep2 > 0.0:
                    for prev_x, prev_y, _, _ in samples:
                        dist2 = (x - prev_x)**2 + (y - prev_y)**2
                        if dist2 < min_sep2:
                            too_close = True
                            break
                
                if not too_close:
                    samples.append((x, y, ch, elev))
                    collected.add((x_key, y_key))
    
    return samples

def plot_target_surface(samples, target_width, target_height):
    """Plot 1: Target surface with samples - each channel gets unique color, elevations within channel get shading."""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Draw target outline
    target = Rectangle((-target_width/2, -target_height/2), 
                       target_width, target_height, 
                       fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(target)
    
    # Define distinct colors for each channel
    channel_colors = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple', 'magenta']
    
    # Plot each channel separately with elevation-based shading
    for ch in range(8):
        # Get samples for this channel
        ch_samples = [s for s in samples if s[2] == ch]
        if not ch_samples:
            continue
        
        xs = [s[0] for s in ch_samples]
        ys = [s[1] for s in ch_samples]
        elevs = [s[3] for s in ch_samples]  # 0-15
        
        # Use elevation to control brightness within channel
        # Elevation 0 = darker, elevation 15 = lighter
        alphas = [0.3 + 0.7 * (e / 15.0) for e in elevs]
        
        for x, y, alpha in zip(xs, ys, alphas):
            ax.scatter(x, y, c=channel_colors[ch], s=15, alpha=alpha, edgecolors='none')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=channel_colors[ch], label=f'Ch {ch}') for ch in range(8)]
    ax.legend(handles=legend_elements, loc='upper right', title='Channels')
    
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title('Target Surface Coverage\n(Color = Channel, Brightness = Elevation within channel)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_channel_heatmap(samples):
    """Plot 2: Channel×Elevation heatmap showing sample counts."""
    # Count samples per (channel, elevation)
    counts = np.zeros((8, 16), dtype=int)
    for _, _, ch, elev in samples:
        counts[ch, elev] += 1
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    im = ax.imshow(counts, aspect='auto', cmap='YlOrRd', 
                   interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Sample Count', rotation=270, labelpad=20)
    
    # Labels
    ax.set_xlabel('Elevation (0-15)')
    ax.set_ylabel('Channel (0-7)')
    ax.set_title('Sample Distribution: 8 Channels × 16 Elevations')
    
    # Set ticks
    ax.set_xticks(range(16))
    ax.set_yticks(range(8))
    
    # Add text annotations
    for ch in range(8):
        for elev in range(16):
            text = ax.text(elev, ch, f'{counts[ch, elev]}',
                          ha="center", va="center", color="black", fontsize=8)
    
    plt.tight_layout()
    return fig

def plot_gimbal_trajectory(offsets):
    """Plot 3: Gimbal trajectory showing scanning pattern."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    dphi_vals = [off[0] for off in offsets]
    dtheta_vals = [off[1] for off in offsets]
    
    # Plot trajectory with color gradient by sequence
    scatter = ax.scatter(np.degrees(dphi_vals), 
                        np.degrees(dtheta_vals),
                        c=range(len(offsets)), 
                        cmap='plasma', s=20, alpha=0.7)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Position Sequence', rotation=270, labelpad=20)
    
    ax.set_xlabel('Azimuth Offset (degrees)')
    ax.set_ylabel('Elevation Offset (degrees)')
    ax.set_title(f'Gimbal Trajectory ({len(offsets)} positions)')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Calibration')
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    ax.legend()
    
    plt.tight_layout()
    return fig

def plot_per_channel_histograms(samples):
    """Plot 4: Per-channel histograms showing elevation distribution."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    
    for ch in range(8):
        ax = axes[ch]
        
        # Get samples for this channel
        elevations = [s[3] for s in samples if s[2] == ch]
        
        # Count per elevation
        counts = [elevations.count(e) for e in range(16)]
        
        # Plot bar chart
        ax.bar(range(16), counts, color=f'C{ch}', alpha=0.7)
        ax.set_title(f'Channel {ch}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Elevation')
        ax.set_ylabel('Sample Count')
        ax.set_xticks(range(16))
        ax.set_ylim(0, max(counts) * 1.1 if counts else 1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add stats
        total = sum(counts)
        min_val = min(counts) if counts else 0
        max_val = max(counts) if counts else 0
        avg_val = total / 16 if total > 0 else 0
        ax.text(0.98, 0.95, f'Total={total}\nMin={min_val}\nMax={max_val}\nAvg={avg_val:.1f}',
                transform=ax.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Sample Distribution per Channel', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def main():
    print("=" * 80)
    print("LiDAR Sampling Visualization")
    print("=" * 80)
    
    # Load configuration
    csv_path = "gimbal_offsets.csv"
    distance_m = 50.0
    target_width_m = 1.8
    target_height_m = 1.3
    
    print(f"\nLoading gimbal offsets from: {csv_path}")
    dphi_calib, dtheta_calib, offsets = load_gimbal_offsets(csv_path)
    print(f"✓ Loaded {len(offsets)} gimbal positions")
    print(f"  Calibration: dphi={np.degrees(dphi_calib):.3f}°, dtheta={np.degrees(dtheta_calib):.3f}°")
    
    # Create sensor model
    print("\nInitializing sensor model...")
    counter = FlatTargetHitCounter(
        target_width_m=target_width_m,
        target_height_m=target_height_m,
        distance_m=distance_m,
        sensor_height_offset_m=0.0,
        sensor_width_offset_m=0.0,
        azimuth_res_rad=np.radians(0.25),
        elevation_res_rad=np.radians(30.0 / 127),
        sensor_hfov_deg=45.0,
        sensor_vfov_deg=30.0
    )
    counter.gimbal_dphi_calib = dphi_calib
    counter.gimbal_dtheta_calib = dtheta_calib
    
    # Reconstruct samples
    print("Reconstructing samples from gimbal positions...")
    samples = reconstruct_samples(counter, offsets)
    print(f"✓ Reconstructed {len(samples)} samples")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    print("  1. Target surface coverage...")
    fig1 = plot_target_surface(samples, target_width_m, target_height_m)
    fig1.savefig('viz_target_surface.png', dpi=150, bbox_inches='tight')
    print("     ✓ Saved: viz_target_surface.png")
    
    print("  2. Channel×Elevation heatmap...")
    fig2 = plot_channel_heatmap(samples)
    fig2.savefig('viz_channel_heatmap.png', dpi=150, bbox_inches='tight')
    print("     ✓ Saved: viz_channel_heatmap.png")
    
    print("  3. Gimbal trajectory...")
    fig3 = plot_gimbal_trajectory(offsets)
    fig3.savefig('viz_gimbal_trajectory.png', dpi=150, bbox_inches='tight')
    print("     ✓ Saved: viz_gimbal_trajectory.png")
    
    print("  4. Per-channel histograms...")
    fig4 = plot_per_channel_histograms(samples)
    fig4.savefig('viz_channel_histograms.png', dpi=150, bbox_inches='tight')
    print("     ✓ Saved: viz_channel_histograms.png")
    
    print("\n" + "=" * 80)
    print("✓ Visualization complete!")
    print("=" * 80)
    
    # Show plots
    plt.show()

if __name__ == "__main__":
    main()
