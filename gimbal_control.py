# Want to do a simple test case of gimbal control when located 20m away from target
from EC_XGRSTDEGimbal import EC_XGRSTDEGimbal
import numpy as np


def order_offsets_start_at_base(offsets):
    """Sort offsets lexicographically by (dtheta, dphi) then rotate so (0.0,0.0) is first.

    offsets: list of (dphi, dtheta) in radians.
    Returns a new list where the relative order is preserved but starts at base.
    """
    ordered = sorted(offsets, key=lambda o: (o[1], o[0]))
    if (0.0, 0.0) in ordered:
        i = ordered.index((0.0, 0.0))
        ordered = ordered[i:] + ordered[:i]
    return ordered


import time
if __name__ == '__main__':
    gimbal = EC_XGRSTDEGimbal('COM4')
    time.sleep(2)  # wait for gimbal to initialize

    # Set some limits
    gimbal.set_horizontal_axis_limits(-90, 90)
    gimbal.set_vertical_axis_limits(-45, 45)

    # Move to initial position
    gimbal.move_to_spot_relative(0, 0)
    time.sleep(2)

    # Move to a new position
    gimbal.move_to_spot_relative(30, 15)
    time.sleep(2)

    # Move back to center
    gimbal.move_to_spot_relative(0, 0)
    time.sleep(2)

    # Demo: fetch offsets from the sampler and execute them starting at base
    # WARNING: this will move the gimbal through several positions. Enable only when ready.
    run_demo = False
    if run_demo:
        sampler = FlatTargetHitCounter(
            target_width_m=1.2,
            target_height_m=1.2,
            distance_m=20,
            azimuth_res_rad=np.deg2rad(45/181),
            elevation_res_rad=np.deg2rad(30/128),
        )
        offsets, p0, p_ext, dphi_sub, dtheta_sub, j, i = sampler.autofill_to_min_samples(min_samples=100, margin_m=0.02)
        ordered = order_offsets_start_at_base(offsets)
        for dphi, dtheta in ordered:
            az_deg = np.rad2deg(dphi)
            el_deg = np.rad2deg(dtheta)
            # Respect gimbal limits and give it time to settle between moves
            gimbal.move_to_spot_relative(az_deg, el_deg)
            time.sleep(0.3)

    print("Gimbal test completed.")