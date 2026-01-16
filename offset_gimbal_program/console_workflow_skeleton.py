"""
Console-based skeleton for test/calibration workflow.
"""
import json
import os
import shutil
import time

import pyautogui
from offset_gimbal_program.gimbal_testing import Test_Setup
import EC_XGRSTDEGimbal
import numpy as np
def main():
    # Precompute or load your test setup objects here
    test_setups = generate_test_setups(
        target_width_m=1.8,
        target_height_m=1.3,
        sensor_height_offset_m=0.0,
        num_azimuth_beams=181,
        num_elevation_beams=128,
        samp_per_channel=1000,
        buffer_m=0.02)
    
    print("Attempting to connect to gimbal...")
    try:
        gimbal = EC_XGRSTDEGimbal.EC_XGRSTDEGimbal('COM5')
    except Exception:
        print("Failed to connect to gimbal. Please check the connection and try again.")
        return
    print("Gimbal connected successfully.")
    while True:
        print("\nGimbal Test Program")

        print("Homing both axes...")
        gimbal.voyant_home_both_axes()
        print("Gimbal homed successfully.")

        # Get test number
        print(f"\nPlease enter a number between 0 and {len(test_setups)-1} to select a test to run and click enter. -1 to exit:")
        test_choice = input("Select a test to run (number): ")
        try:
            choice_num = int(test_choice)
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue
        if choice_num == -1:
            print("Exiting program.")
            break
        if not (0 <= choice_num < len(test_setups)):
            print("Invalid selection. Please try again.")
            continue


        selected_test = test_setups[choice_num]
        calibrate_device(selected_test, gimbal)

        if not confirm_auto_mouse():
            print("Auto mouse not ready. Returning to test selection.")
            continue
        run_test(selected_test)

def generate_test_setups(target_width_m, target_height_m, sensor_height_offset_m, num_azimuth_beams, num_elevation_beams, samp_per_channel, buffer_m):
    """Generate a list of test setups with varying parameters."""
    setups = []
    ranges_m = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    azimuth_settings_percent = [-75, 0, 75]
    HFOVs_deg = [45, 120]
    test_id = 0
    for dist in ranges_m:
        for az in azimuth_settings_percent:
            for HFOV in HFOVs_deg:
                setup = Test_Setup(test_id, target_width_m, target_height_m, dist,
                                    sensor_height_offset_m,
                                    num_azimuth_beams, num_elevation_beams,
                                    samp_per_channel,
                                    buffer_m)
                setups.append(setup)
                test_id += 1
    return setups

def calibrate_device(test, gimbal):
    print(f"\n--- Calibration Step ---")
    print("\nCalibrating azimuth axis...")
    gimbal.move_vertical_axis_relative(-np.degrees(test.gimbal_h_offset_rad)) # very confusing that horizonatal gimbal axis is vertical in my model
    print("\nAziumuth calibration complete.")
    print("Please perform the necessary vertical calibration now.")

    input("Press Enter when calibration is complete...")
    # note set_vert_calibration expects degrees
    test.set_vert_calibration(gimbal.get_horizontal_axis_position()) # very confusing that horizonatal gimbal axis is vertical in my model
    print("Calibration complete.")

def confirm_auto_mouse():
    print("\n--- Auto Mouse Clicker Preparation ---")
    while True:
        ready = input("Is the gui ready for auto mouse clicking? (y/n): ").strip().lower()
        if ready == 'y':
            return True
        elif ready == 'n':
            return False
        else:
            print("Please enter 'y' or 'n'.")
def copy_capture(gui_result_folder, test_case_path, frame,
                 fixed_wait):
    """Simplified move: wait a fixed time then move the captured subfolder or files.

    Returns True if anything was moved, False otherwise.
    """

    frame_result_dir = os.path.join(test_case_path, f'frame_{frame:04d}')
    # Simple fixed wait (caller chooses duration)
    time.sleep(fixed_wait)
    # Check if source actually has files before copying
    if not os.path.exists(gui_result_folder) or not os.listdir(gui_result_folder):
        print(f"Error: No files found in {gui_result_folder}. Capture likely failed.")
        return False
    
    try:
        # Copy folder
        shutil.copytree(gui_result_folder, frame_result_dir)
        time.sleep(fixed_wait)
        print(f"Copied {gui_result_folder} inside {frame_result_dir}")
        # SAFETY CLEAR: Delete source files so we don't copy them again if next capture fails
        for filename in os.listdir(gui_result_folder):
            file_path = os.path.join(gui_result_folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to cleanup {file_path}: {e}')

        return True
    except FileExistsError:
        print(f"Frame result directory {frame_result_dir} already exists. Skipping copy.")
        return False
    except Exception as e:
        print(f"Move failed: {e}")
        return False

def run_test(test, gimbal):
    print(f"\n--- Running Test ---")
    # Get the positions for gimbal to move to
    positions, positions_info = test.get_positions()

    # may need to negate positions depending on gimbal orientation
    print(f"Total positions to visit: {len(positions)}")
    print(np.degrees(positions))

    gui_result_folder = os.path.expanduser('~/git/carbon_motor_drive/results/integration_capture')
    all_results_folder = os.path.expanduser('~/Desktop/EOLTesting/results')

    # need to add code to make another folder which is data for that very test case

    test_folder = f"test_{test.test_id}_dist{test.distance_m}m_azset{test.azimuth_setting_percent}pct_HFOV{test.hfov_deg}deg"
    test_case_path = os.path.join(all_results_folder, test_folder) 
    os.makedirs(test_case_path, exist_ok=True)

    # Save a json with the setup conditions
    json_file_name = f"test_setup_{test.test_id}.json"
    out_file = os.path.join(test_case_path, json_file_name)


    with open(out_file, 'w') as f:
        json.dump(test.__dict__, f, indent=4, default=str)
    print(f'Saved test setup to {out_file}')

    frame = 0
    for dphi, dtheta in positions:
        # note:: they are swapped when passing to move_to_spot_relative because gimbal's horizontal axis corresponds to elevation in our setup, and vertical axis
        # corresponds to azimuth in our setup.
        gimbal.move_to_spot_relative(np.degrees(dtheta),np.degrees(dphi)) # it is blocking, so will wait
        print(f"Moved to position: H={np.degrees(dphi):.3f} deg, V={np.degrees(dtheta):.3f} deg")
        time.sleep(0.25)  # wait a bit at each position
        pyautogui.click() # Trigger lidar capture. Add the correct mouseclick coordinates as parameter

        # Move the capture into a numbered frame folder (fixed short wait)
        moved = copy_capture(gui_result_folder, test_case_path, frame, fixed_wait=0.5)
        if not moved:
            print(f'Warning: nothing moved for frame {frame:04d}')

        frame += 1



    print("Test is running...")
    # Simulate test duration
    input("Press Enter when test is finished...")
    print(f"{test['name']} complete.")

def review_results():
    print("\n--- Review Results ---")
    # Insert result review logic here
    input("Press Enter to return to menu...")

if __name__ == "__main__":
    main()
