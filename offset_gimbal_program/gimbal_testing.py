import sys
import os
import time
import numpy as np
import shutil
import pyautogui
from zaber_motion import Units, Library
from zaber_motion.ascii import Connection
from new_point_counter import Test_Setup

# script_dir = os.path.dirname(os.path.abspath(__file__))
# lib_path = os.path.join(script_dir, '..', 'lib')
# sys.path.append(lib_path)
from EC_XGRSTDEGimbal import EC_XGRSTDEGimbal



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
    gimbal = EC_XGRSTDEGimbal('COM5')
    print("Homing both axes...")
    gimbal.voyant_home_both_axes()
    time.sleep(3)

    # Define the test setup
    setup = Test_Setup(target_width_m, target_height_m,
                       distance_m, sensor_height_offset_m, sensor_width_offset_m,
                       num_azimuth_beams, num_elevation_beams, samp_per_channel, buffer_m)
    
    # Compute and set calibration; move gimbal to position
    calib = setup.compute_calibration_offset()
    setup.set_calibration(calib)
    print(f"Calibration: {np.degrees(calib[1]):.3f}°\n")
    gimbal.move_to_spot_H_relatively(-np.degrees(setup.get_v_calibration()))

    # Get the positions for gimbal to move to
    positions, positions_info = setup.get_positions()

    # may need to negate positions depending on gimbal orientation
    print(f"Total positions to visit: {len(positions)}")
    print(np.degrees(positions))

    # linux
    # gui_result_folder = '~/git/carbon_motor_drive/results/integration_capture/'
    # all_results_folder = '~/Desktop/EOLTesting/results/'
    # windows
    gui_result_folder = 'C:\\Users\\BenSheppard\\git\\carbon_motor_drive\\results\\integration_capture\\'
    all_results_folder = 'C:\\Users\\BenSheppard\\Desktop\\EOLTesting\\results\\'

    # need to add code to make another folder which is data for that very test case

    test_case_path = os.path.join(all_results_folder, 'test')
    frame = 0
    for dphi, dtheta in positions:
        # note:: they are swapped because gimbal's horizontal axis corresponds to elevation in our setup, and vertical axis
        # corresponds to azimuth in our setup.
        gimbal.move_to_spot_relative(np.degrees(dtheta),np.degrees(dphi)) # it is blocking, so will wait
        print(f"Moved to position: H={np.degrees(dphi):.3f} deg, V={np.degrees(dtheta):.3f} deg")
        time.sleep(0.25)  # wait a bit at each position
        pyautogui.click() # Trigger lidar capture. Add the correct mousclick coordinates as parameter
        
        frame_result = os.path.join(test_case_path, str(frame).zfill(3))
        shutil.move(gui_result_folder, frame_result)
        time.sleep(0.25)  # wait a bit after capture
        frame +=1




    # Example of moving gimbal to each position and capturing data
    # for dphi, dtheta in offsets:
    # target_az = base_az + dphi     # radians -> convert if needed
    # target_el = base_el + dtheta
    # gimbal.move_to(target_az, target_el, speed=...)
    # gimbal.wait_until_settled()
    # lidar.capture()

    # gimbal.move_to_spot_H_relatively(-10)
    # gimbal.move_to_spot_V_relatively(-10)
    # time.sleep(3)
    # gimbal.move_horizontal_axis_relative(-40)
    # gimbal.voyant_home_horizontal_axis()
    # gimbal.move_vertical_axis_relative(-1.88)
    # gimbal.voyant_home_vertical_axis()
    # time.sleep(2)
    # gimbal.move_to_spot_relative(15,15)
    # gimbal.voyant_home_both_axes()
    # gimbal.home_vertical_axis()
    # gimbal.home_ho_axis()

