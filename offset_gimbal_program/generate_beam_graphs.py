"""Test the get_positions procedure with visualization at multiple distances.
Ben Sheppard


"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import to_rgb
from testing import Test_Setup

# Helper function to lighten a color

def lighten_color(color, amount=0.5):
    """Lighten a color by blending with white"""
    rgb = to_rgb(color)
    return tuple(rgb[i] + (1 - rgb[i]) * amount for i in range(3))

# Diagnostics and plotting functions moved from new_point_counter.py
def init_diagnostics():
    import numpy as np
    return {
        "channel_samples": np.zeros(8, dtype=float),
        "elevation_samples": np.zeros((8, 16), dtype=float),
    }

def sensor_index_to_channel_elev(sensor_idx):
    channel_idx = 7 - (sensor_idx // 16)
    elevation_idx = 15 - (sensor_idx % 16)
    return channel_idx, elevation_idx

def record_samples(diag, hit_indices, n_azimuth, microsteps, channel_filter=None):
    if diag is None or hit_indices is None or n_azimuth <= 0 or microsteps <= 0:
        return
    sample_amount = n_azimuth * microsteps
    for idx in hit_indices:
        channel_idx, elevation_idx = sensor_index_to_channel_elev(int(idx))
        if channel_idx < 0 or channel_idx >= 8:
            continue
        if channel_filter is not None and channel_idx != channel_filter:
            continue
        diag["elevation_samples"][channel_idx, elevation_idx] += sample_amount
        diag["channel_samples"][channel_idx] += sample_amount

def plot_diagnostics(diag, label=None):
    if diag is None:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping diagnostic plots.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(np.arange(8), diag["channel_samples"])
    axes[0].set_xlabel("Channel")
    axes[0].set_ylabel("Total samples")
    axes[0].set_title("Samples per channel")
    im = axes[1].imshow(diag["elevation_samples"], aspect="auto", cmap="viridis")
    axes[1].set_xlabel("Elevation index")
    axes[1].set_ylabel("Channel")
    axes[1].set_title("Samples per elevation")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    if label:
        import os
        outdir = 'photos'
        os.makedirs(outdir, exist_ok=True)
        safe_label = str(label).replace(" ", "_")
        outfile = os.path.join(outdir, f"diagnostics_{safe_label}.png")
        fig.savefig(outfile, dpi=200, bbox_inches="tight")
        print(f"Saved diagnostics plot to {outfile}")
    plt.show()

# Test at multiple distances

test_distances = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # in meters
spot_diameter_m=0.0135  # 1.35cm diameter


# Channel colors (8 distinct colors)
channel_colors = ['#e41a1c', '#377eb8', '#4daf4a', "#91499c", '#ff7f00', '#ffff33', '#a65628', "#af5083"]

for distance_m in test_distances:
    print(f"\n{'='*60}")
    print(f"Testing at distance: {distance_m}m")
    print(f"{'='*60}")
    
    # Create setup at specified distance
    setup = Test_Setup(
        target_width_m=1.8,
        target_height_m=1.3,
        distance_m=distance_m,
        sensor_height_offset_m=0.0,
        sensor_width_offset_m=0.0,
        num_azimuth_beams=181,
        num_elevation_beams=128,
        samp_per_channel=1000,
        buffer_m=0.01,
    )

    # Set calibration
    calib = setup.compute_calibration_offset()
    setup.set_calibration(calib)
    print(f"Calibration: {np.degrees(calib[1]):.3f} 0\n")

    # Diagnostics
    diag = init_diagnostics()
    # Run the test
    # Note: Removed arguments that might not be in the imported class definition
    positions, positions_map = setup.get_positions()
    
    # Record samples (assuming hit_indices, n_azimuth, microsteps available)
    # record_samples(diag, hit_indices, n_azimuth, microsteps)
    # Plot diagnostics
    plot_diagnostics(diag, label=f"{distance_m}m")

    # Get unique vertical positions (one plot per macrostep frame)
    unique_v_offsets = sorted(set([v for h, v in positions]))
    print(f"\n{len(unique_v_offsets)} unique vertical positions (frames)")

    # Separate initial and stepped positions
    initial_positions = [(h, v) for h, v in positions if h == 0.0]
    stepped_positions = [(h, v) for h, v in positions if h != 0.0]

    print(f"Initial positions: {len(initial_positions)}")
    print(f"Stepped positions: {len(stepped_positions)}")

    # Create one plot per vertical position
    for frame_idx, v_offset in enumerate(unique_v_offsets):
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Determine which channel is being viewed at this v_offset
        # Each channel is 16 elevations = 16 * elevation_res apart
        # Channel 0 is at the top (highest angles) at calibration
        channel_num = int(round(v_offset / (16 * setup.elevation_res_rad)))
        
        fig.suptitle(f'Distance: {distance_m}m | Frame {frame_idx}: Channel {channel_num} at v_offset={np.degrees(v_offset):.2f}°', 
                     fontsize=14, fontweight='bold')
        
        # Plot target boundary
        half_w = setup.target_width_m / 2.0
        half_h = setup.target_height_m / 2.0
        target_rect = Rectangle(
            (-half_w, -half_h), 
            setup.target_width_m, 
            setup.target_height_m,
            fill=False, 
            edgecolor='black', 
            linewidth=3
        )
        ax.add_patch(target_rect)
        
        # Plot buffer zone (usable area boundary)
        if setup.buffer_m > 0:
            usable_half_w = half_w - setup.buffer_m
            usable_half_h = half_h - setup.buffer_m
            buffer_rect = Rectangle(
                (-usable_half_w, -usable_half_h),
                2 * usable_half_w,
                2 * usable_half_h,
                fill=False,
                edgecolor='red',
                linewidth=2,
                linestyle='--',
                alpha=0.7
            )
            ax.add_patch(buffer_rect)
        
        # Track which channels have been added to legend
        channels_plotted = set()
        
        # Plot initial position at this v_offset
        initial_at_v = [h for h, v in initial_positions if v == v_offset]
        if initial_at_v:
            h_offset = initial_at_v[0]  # Should be 0.0
            
            # Generate beam angles at this position
            phis = setup.azimuth_angles(h_offset)
            thetas = setup.elevation_angles(setup.gimbal_v_offset_rad + v_offset)
            
            # Project to target
            PHI, THETA = np.meshgrid(phis, thetas)
            X = setup.distance_m * np.tan(PHI)
            Y = setup.distance_m * np.tan(THETA)
            
            # Apply buffer to mask (usable area only)
            usable_half_w = half_w - setup.buffer_m
            usable_half_h = half_h - setup.buffer_m
            mask = (np.abs(X) <= usable_half_w) & (np.abs(Y) <= usable_half_h)
            
            # Plot beams by channel (INITIAL - thick, dark)
            for ch in range(8):
                # NOTE: The loop here iterates LOGICAL channels (0=Top).
                # But the map uses RAW channels/elevations.
                # Visualization matches the scatter plot logic.
                ch_start = 128 - (ch + 1) * 16
                ch_end = 128 - ch * 16
                
                ch_mask = mask[ch_start:ch_end, :]
                ch_X = X[ch_start:ch_end, :]
                ch_Y = Y[ch_start:ch_end, :]
                
                if np.any(ch_mask):
                    # Calculate proper circle size: convert spot diameter to display coordinates
                    # Figure is 12 inches wide showing ~2m of data, so need to scale properly
                    # Use diameter directly in data coordinates, scale by fig size
                    fig_width_inches = 12
                    data_width = 2.6  # actual x-axis range (1.3 * target width)
                    points_per_data_unit = (fig_width_inches * 72) / data_width  # 72 pts/inch
                    circle_size = (spot_diameter_m * points_per_data_unit) ** 2
                    
                    # Only add label once per channel
                    label = f'Ch{ch}' if ch not in channels_plotted else None
                    if label:
                        channels_plotted.add(ch)
                    
                    ax.scatter(ch_X[ch_mask], ch_Y[ch_mask], 
                              s=circle_size,
                              facecolors='none', 
                              edgecolors=channel_colors[ch],
                              linewidths=2.0,
                              alpha=1.0,
                              zorder=2,
                              label=label)
        
        # Plot stepped positions at this v_offset
        stepped_at_v = [h for h, v in stepped_positions if v == v_offset]
        for h_offset in stepped_at_v:
            
            phis = setup.azimuth_angles(h_offset)
            thetas = setup.elevation_angles(setup.gimbal_v_offset_rad + v_offset)
            
            PHI, THETA = np.meshgrid(phis, thetas)
            X = setup.distance_m * np.tan(PHI)
            Y = setup.distance_m * np.tan(THETA)
            
            # Apply buffer to mask (usable area only)
            usable_half_w = half_w - setup.buffer_m
            usable_half_h = half_h - setup.buffer_m
            mask = (np.abs(X) <= usable_half_w) & (np.abs(Y) <= usable_half_h)
            
            # Plot beams by channel (STEPPED - thin, light)
            for ch in range(8):
                ch_start = 128 - (ch + 1) * 16
                ch_end = 128 - ch * 16
                
                ch_mask = mask[ch_start:ch_end, :]
                ch_X = X[ch_start:ch_end, :]
                ch_Y = Y[ch_start:ch_end, :]
                
                if np.any(ch_mask):
                    # Calculate proper circle size (same as initial)
                    fig_width_inches = 12
                    data_width = 2.6
                    points_per_data_unit = (fig_width_inches * 72) / data_width
                    circle_size = (spot_diameter_m * points_per_data_unit) ** 2
                    
                    # Use lighter shade of the same channel color
                    light_color = lighten_color(channel_colors[ch], amount=0.5)
                    
                    # Only add label once per channel (if not already added)
                    label = f'Ch{ch}' if ch not in channels_plotted else None
                    if label:
                        channels_plotted.add(ch)
                    
                    ax.scatter(ch_X[ch_mask], ch_Y[ch_mask], 
                              s=circle_size,
                              facecolors='none', 
                              edgecolors=light_color,
                              linewidths=2.0,
                              alpha=1.0,
                              zorder=1,
                              label=label)
        
        # Label elevations directly from map data
        # Recalculate geometric projection for labeling
        phis = setup.azimuth_angles(0.0)
        thetas = setup.elevation_angles(setup.gimbal_v_offset_rad + v_offset)
        PHI, THETA = np.meshgrid(phis, thetas)
        X = setup.distance_m * np.tan(PHI)
        Y = setup.distance_m * np.tan(THETA)
        usable_half_w = half_w - setup.buffer_m
        usable_half_h = half_h - setup.buffer_m
        mask = (np.abs(X) <= usable_half_w) & (np.abs(Y) <= usable_half_h)
        
        deg_key = round(np.degrees(v_offset), 5)
        
        if deg_key in positions_map:
            active_channels, active_elevs = positions_map[deg_key]
            
            # Iterate through the actual beams recorded in the map
            for i in range(len(active_channels)):
                raw_ch = active_channels[i]
                map_elev = active_elevs[i] # This is sensor_idx % 16
                
                # Calculate sensor_idx to find geometric position
                # map_elev (0..15) + raw_ch (0..7) * 16 = sensor_idx (0..127)
                sensor_idx = int(raw_ch * 16 + map_elev)
                
                # Verify this beam has pixels on target in the projection
                if 0 <= sensor_idx < 128 and np.any(mask[sensor_idx, :]):
                    hit_cols = np.where(mask[sensor_idx, :])[0]
                    if len(hit_cols) > 0:
                        col_idx = hit_cols[len(hit_cols)//2]
                        y_pos = Y[sensor_idx, col_idx]
                        
                        # Position label
                        x_pos = half_w * 1.15
                        
                        # Use the EXACT value from the map as the label
                        label_str = str(map_elev)
                        
                        ax.text(x_pos, y_pos, label_str, 
                               fontsize=8, ha='left', va='center',
                               color='black', weight='bold')

        ax.set_xlabel('Horizontal Position (m)', fontsize=11)
        ax.set_ylabel('Vertical Position (m)', fontsize=11)
        ax.set_aspect('equal')
        
        # Only show legend if there are entries
        if channels_plotted:
            ax.legend(fontsize=9, loc='upper left', ncol=1)
        
        # Zoom to target area
        ax.set_xlim(-half_w * 1.3, half_w * 1.3)
        ax.set_ylim(-half_h * 1.3, half_h * 1.3)
            
        plt.tight_layout()
        outdir = 'photos2'
        import os
        os.makedirs(outdir, exist_ok=True)
        outpath = os.path.join(outdir, f'{distance_m}m_frame_{frame_idx:02d}_channel_{channel_num}.png')
        plt.savefig(outpath, dpi=150, bbox_inches='tight')
        print(f"Saved {outpath}")
        plt.close()

    print(f"\n✓ Generated {len(unique_v_offsets)} frame visualizations for {distance_m}m")

print(f"\n{'='*60}")
print(f"✓ Completed testing for all distances: {test_distances}")
print(f"{'='*60}")