import numpy as np

from point_counter import FlatTargetHitCounter


"""
collect_data: Collects data by stepping the gimbal through positions defined by a FlatTargetHitCounter.
Parameters:
    min_num_samples: Minimum number of samples to collect.
    target_width_m: Width of the target in meters.
    target_height_m: Height of the target in meters.
    edge_margin_m: Margin from the edge of the target in meters (need to do testing to determine appropriate margin)
    distance_m: Distance to the target in meters.
    azimuth_res_rad: Azimuth resolution in radians. (system defined)
    elevation_res_rad: Elevation resolution in radians. (system defined)
    spot_radius_m: Radius of the spot to consider for hits in meters. (system defined)
"""
def collect_data(min_num_samples,target_width_m, target_height_m, edge_margin_m, distance_m, azimuth_res_rad, elevation_res_rad, spot_radius_m):
    test_setup = FlatTargetHitCounter(
        target_width_m=target_width_m,
        target_height_m=target_height_m,
        distance_m=distance_m,
        azimuth_res_rad=azimuth_res_rad,
        elevation_res_rad=elevation_res_rad,
    )
    offsets, *_ = test_setup.autofill_to_min_samples(min_samples=min_num_samples, margin_m=edge_margin_m,spot_radius_m=spot_radius_m)

    np_offsets = np.array(offsets)
    offsets_deg = np.rad2deg(np_offsets)
    test_setup.save_csv_offsets("gimbal_offsets_deg.csv", offsets_deg)
    return offsets_deg
    
if __name__ == '__main__':

    HFOV = 45
    VFOV = 30
    HFOV_RES = 45/181
    VFOV_RES = 30/128  
    SPOT_RADIUS_M = 0.0135 / 2 # diameter is 1.35 cm
    
    offsets = collect_data(
        min_num_samples=300,
        target_width_m=1.2,
        target_height_m=1.2,
        edge_margin_m=0.02,
        distance_m=200,
        azimuth_res_rad=np.deg2rad(HFOV_RES),
        elevation_res_rad=np.deg2rad(VFOV_RES),
        spot_radius_m=SPOT_RADIUS_M
    )

    print(f"Collected {len(offsets)} offsets.")
    print(offsets)