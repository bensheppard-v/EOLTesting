"""Debug which elevation indices are hitting at different v_offsets."""

import numpy as np
from new_point_counter import Test_Setup

distance_m = 70

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
    spot_diameter_m=0.0135
)

calib = setup.compute_calibration_offset()
setup.set_calibration(calib)

# Test a few v_offsets
test_v_offsets = [0.0, np.radians(2.303), np.radians(16.122)]  # Channel 0, Channel 1, Channel 7
channel_names = ["Channel 0", "Channel 1", "Channel 7"]

half_w = setup.target_width_m / 2.0
half_h = setup.target_height_m / 2.0
usable_half_w = half_w - setup.buffer_m
usable_half_h = half_h - setup.buffer_m

for v_offset, ch_name in zip(test_v_offsets, channel_names):
    print(f"\n{ch_name} (v_offset={np.degrees(v_offset):.3f}°):")
    
    phis = setup.azimuth_angles(0.0)
    thetas = setup.elevation_angles(setup.gimbal_v_offset_rad + v_offset)
    
    PHI, THETA = np.meshgrid(phis, thetas)
    X = setup.distance_m * np.tan(PHI)
    Y = setup.distance_m * np.tan(THETA)
    
    mask = (np.abs(X) <= usable_half_w) & (np.abs(Y) <= usable_half_h)
    
    hitting_indices = []
    for elev_idx in range(128):
        if np.any(mask[elev_idx, :]):
            hitting_indices.append(elev_idx)
    
    print(f"  Elevation indices hitting target: {hitting_indices}")
    print(f"  Count: {len(hitting_indices)}")
    
    # Which "channels" do these belong to?
    channels_hit = set()
    for idx in hitting_indices:
        # Channel numbering: 0-15 = Ch7, 16-31 = Ch6, ... 112-127 = Ch0
        ch_num = 7 - (idx // 16)
        channels_hit.add(ch_num)
    print(f"  These correspond to channel(s): {sorted(channels_hit)}")
