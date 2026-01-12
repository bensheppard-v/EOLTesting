"""Test the run_test procedure with visualization at multiple distances."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import to_rgb
from new_point_counter import Test_Setup

# Helper function to lighten a color
def lighten_color(color, amount=0.5):
    """Lighten a color by blending with white"""
    rgb = to_rgb(color)
    return tuple(rgb[i] + (1 - rgb[i]) * amount for i in range(3))

# Test at multiple distances
test_distances = [100, 150]
#test_distances = [5, 10, 50, 70]


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
        samp_per_channel=400,
        buffer_m=0.01,
        spot_diameter_m=0.0135  # 1.35cm diameter
    )

    # Set calibration
    calib = setup.compute_calibration_offset()
    setup.set_calibration(calib)
    print(f"Calibration: {np.degrees(calib[1]):.3f}°\n")

    # Run the test
    positions = setup.run_test(diagnostics=True, plot_hist=True, diag_label=f"{distance_m}m")

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
                    circle_size = (setup.spot_diameter_m * points_per_data_unit) ** 2
                    
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
                    circle_size = (setup.spot_diameter_m * points_per_data_unit) ** 2
                    
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
        
        # Label elevations only on initial position
        phis = setup.azimuth_angles(0.0)
        thetas = setup.elevation_angles(setup.gimbal_v_offset_rad + v_offset)
        PHI, THETA = np.meshgrid(phis, thetas)
        X = setup.distance_m * np.tan(PHI)
        Y = setup.distance_m * np.tan(THETA)
        # Apply buffer to mask
        usable_half_w = half_w - setup.buffer_m
        usable_half_h = half_h - setup.buffer_m
        mask = (np.abs(X) <= usable_half_w) & (np.abs(Y) <= usable_half_h)
        
        for ch in range(8):
            ch_start = 128 - (ch + 1) * 16
            for elev in range(16):
                elev_idx = ch_start + elev
                if np.any(mask[elev_idx, :]):
                    hit_cols = np.where(mask[elev_idx, :])[0]
                    if len(hit_cols) > 0:
                        # Get y position from center of beam pattern
                        col_idx = hit_cols[len(hit_cols)//2]
                        y_pos = Y[elev_idx, col_idx]
                        # Place label to the right of the target
                        x_pos = half_w * 1.15
                        ax.text(x_pos, y_pos, str(elev), 
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
        plt.savefig(f'{distance_m}m_frame_{frame_idx:02d}_channel_{channel_num}.png', dpi=150, bbox_inches='tight')
        print(f"Saved {distance_m}m_frame_{frame_idx:02d}_channel_{channel_num}.png")
        plt.close()

    print(f"\n✓ Generated {len(unique_v_offsets)} frame visualizations for {distance_m}m")

print(f"\n{'='*60}")
print(f"✓ Completed testing for all distances: {test_distances}")
print(f"{'='*60}")
