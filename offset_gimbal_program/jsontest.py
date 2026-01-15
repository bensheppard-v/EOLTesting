
import json
import os
import datetime
from new_point_counter import Test_Setup

target_width_m = 1.8
target_height_m = 1.3
distance_m = 100
sensor_height_offset_m = 0
sensor_width_offset_m = 0
num_azimuth_beams = 181
num_elevation_beams = 128
samp_per_channel = 400
buffer_m = 0.01

if __name__ == '__main__':
    # Define the test setup
    setup = Test_Setup(target_width_m, target_height_m,
                       distance_m, sensor_height_offset_m, sensor_width_offset_m,
                       num_azimuth_beams, num_elevation_beams, samp_per_channel, buffer_m)

    # Prepare results directory (expand ~) and a timestamped test-case folder
    all_results_folder = os.path.expanduser('~/Desktop/EOLTesting/results')
    os.makedirs(all_results_folder, exist_ok=True)

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    test_case_path = os.path.join(all_results_folder, f'test_{ts}')
    os.makedirs(test_case_path, exist_ok=True)

    # Save a json with the setup conditions into the test-case folder
    out_file = os.path.join(test_case_path, 'test_setup.json')
    try:
        with open(out_file, 'w') as f:
            json.dump(setup.__dict__, f, indent=4, default=str)
    except TypeError:
        # As a fallback, stringify any non-serializable values
        with open(out_file, 'w') as f:
            json.dump({k: str(v) for k, v in setup.__dict__.items()}, f, indent=4)

    print(f'Saved test setup to {out_file}')

