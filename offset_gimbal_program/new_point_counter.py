import numpy as np
# Lawrence Tzuang and Ben Sheppard
class Test_Setup:
    def __init__(
        self, 
        target_width_m, 
        target_height_m, 
        distance_m, 
        sensor_height_offset_m,
        sensor_width_offset_m,
        num_azimuth_beams,
        num_elevation_beams,
        samp_per_channel,
        buffer_m=0.0,
        ):

        self.target_width_m = target_width_m
        self.target_height_m = target_height_m
        self.distance_m = distance_m
        self.sensor_height_offset_m = sensor_height_offset_m
        self.sensor_width_offset_m = sensor_width_offset_m
        self.num_azimuth_beams = num_azimuth_beams
        self.num_elevation_beams = num_elevation_beams

        self.samp_per_channel = samp_per_channel
        self.buffer_m = buffer_m
        
        # Sensor FOV parameters
        self.hfov_rad = np.radians(45.0)  # 45° horizontal FOV
        self.vfov_rad = np.radians(22.5)  # 22.5° vertical FOV
        self.theta0 = 0.0  # Uncalibrated center elevation angle (radians)
        self.azimuth_res_rad = np.radians(45) / (num_azimuth_beams - 1)
        self.elevation_res_rad = np.radians(22.5) / (num_elevation_beams - 1)
        # Gimbal calibration offsets (will be set manually for each distance test)
        self.gimbal_h_offset_rad = 0.0
        self.gimbal_v_offset_rad = 0.0

    def _angular_limits(self):
        """Return half-angles of sensor FOV in radians."""
        phi_half = self.hfov_rad / 2.0
        theta_half = self.vfov_rad / 2.0
        return phi_half, theta_half
    
    def compute_calibration_offset(self):
        """Compute gimbal offset to align highest beam with top of target. Does this with vertical gimbal offset. Doesn't touch horizontal offset.
        
        note: I will need to do this manually for each distance in the test script.
        However, this function can provide the geometrical calculation.
        """
        _, theta_half = self._angular_limits()
        # Align highest beam with actual top of target 
        target_top = self.target_height_m / 2.0
        angle_to_top = np.arctan((target_top - self.sensor_height_offset_m) / self.distance_m)
        dtheta_calib = angle_to_top - (self.theta0 + theta_half) - np.arctan(self.buffer_m / self.distance_m)
        return 0.0, dtheta_calib # Horizontal offset is zero for this setup
   
    def set_calibration(self, calib_offset):
        """Set gimbal calibration offsets in radians."""
        dphi_calib, dtheta_calib = calib_offset # Unpack calib_offset tuple
        self.gimbal_h_offset_rad = dphi_calib
        self.gimbal_v_offset_rad = dtheta_calib

    def get_v_calibration(self):
        """Get current vertical gimbal calibration offset in radians."""
        return self.gimbal_v_offset_rad
    
    def azimuth_angles(self, dphi_offset=0.0):
        """Generate all azimuth (horizontal) beam angles in radians."""
        phi_half, _ = self._angular_limits()
        return np.linspace(-phi_half, phi_half, self.num_azimuth_beams) + dphi_offset
    
    def elevation_angles(self, dtheta_offset=0.0):
        """Generate all elevation (vertical) beam angles in radians."""
        _, theta_half = self._angular_limits()
        return self.theta0 + np.linspace(-theta_half, theta_half, self.num_elevation_beams) + dtheta_offset
    
    def count_beams_on_target(self, dphi_rel=0.0, dtheta_rel=0.0, return_indices=False):
        """
        Count how many horizontal and vertical beams hit the target.
        
        Parameters
        ----------
        dphi_rel : float
            Relative azimuth offset from calibration (radians)
        dtheta_rel : float
            Relative elevation offset from calibration (radians)
        
        Returns
        -------
        n_azimuth : int
            Number of horizontal beams hitting target
        n_elevation : int
            Number of vertical beams hitting target
        """
        # Convert relative offsets to absolute
        dphi_abs = self.gimbal_h_offset_rad + dphi_rel
        dtheta_abs = self.gimbal_v_offset_rad + dtheta_rel
        
        # Generate beam angles
        phis = self.azimuth_angles(dphi_abs)
        thetas = self.elevation_angles(dtheta_abs)
        
        # Create meshgrid for all beam combinations
        PHI, THETA = np.meshgrid(phis, thetas)
        
        # Project onto target plane (distance D away)
        X = self.distance_m * np.tan(PHI)
        Y = self.distance_m * np.tan(THETA)
        
        # Check which points are inside target boundaries (accounting for buffer)
        half_w = self.target_width_m / 2.0 - self.buffer_m
        half_h = self.target_height_m / 2.0 - self.buffer_m
        mask = (np.abs(X) <= half_w) & (np.abs(Y) <= half_h)
        
        # Count unique beams hitting (not total points)
        # Sum along each axis to see which beams hit
        azimuth_hits = np.any(mask, axis=0)  # Which horizontal beams hit
        elevation_hits = np.any(mask, axis=1)  # Which vertical beams hit

        index_elevation_hits = np.flatnonzero(elevation_hits) # returns list of indices of elevation beams that hit
        n_azimuth = int(np.sum(azimuth_hits))
        n_elevation = int(np.sum(elevation_hits))

        if return_indices:
            hit_indices = np.flatnonzero(elevation_hits)
            return n_azimuth, n_elevation, hit_indices
        return n_azimuth, n_elevation

    def _microsteps_needed(self, n_azimuth, n_elevation):
        """Return microsteps so each elevation contributes its sample share."""
        if n_azimuth <= 0 or n_elevation <= 0:
            return 0
        samples_needed = (n_elevation / 16.0) * self.samp_per_channel
        samples_per_microstep = n_azimuth * n_elevation
        steps = int(np.ceil(samples_needed / samples_per_microstep))
        return max(1, steps)

    def _microstep_offsets(self, steps):
        """Return symmetric azimuth offsets for microstepping."""
        if steps <= 1:
            return [0.0]
        step_size = self.azimuth_res_rad / 20.0 # This is a parameter we can play with. Currently stepping to the right by 1/20 of azimuth resolution. Can't go too big or we hit same spot on target more than once.
        center = (steps - 1) / 2.0
        return [(idx - center) * step_size for idx in range(steps)]

    def _sensor_index_to_channel_elev(self, sensor_idx):
        channel_idx = 7 - (sensor_idx // 16)
        # Convert matrix index (0 bottom) to channel-local index from top (0..15)
        elevation_idx = 15 - (sensor_idx % 16)
        return channel_idx, elevation_idx

    def get_positions(self):
        """
        Run the test procedure following the pseudocode.
        Returns list of relative gimbal positions visited as well as a map of covered indices at each vertical position.
        """

        # Count beams at calibration
        _, n_vertical = self.count_beams_on_target(0, 0)
        if n_vertical == 0:
            print("No vertical beams on target at calibration. Abort test.")
            return [], {}
        print(f"Starting test at calibration: {n_vertical} vertical beams visible")


        # Track all positions visited
        positions = []
        # Track covered indices for each position
        positions_map = {}        
        # Current relative vertical offset from calibration (BASE position)
        current_v_rel = 0.0  
        


        # There are 2 cases. One where full channel is visible at once, and one where partial channel is visible.
        if n_vertical >= 16:
            print("Case: n_vertical >= 16 (full channel visible)")

            channel = 0
            while channel < 8:
                print(f"\nChannel {channel}:")
                current_position_v = current_v_rel

                n_azimuth, n_elevation, channel_elev_indices = self.count_beams_on_target(0, current_position_v, return_indices=True)
                print("channel elev:",channel_elev_indices)
                # Get the channel/elevation pair from the actual elevation indices
                channels = channel_elev_indices // 16
                elevations = channel_elev_indices % 16
                angle = round(np.degrees(current_position_v), 5) 
                if angle not in positions_map:
                        positions_map[angle] = (channels, elevations)
                if n_azimuth <= 0 or n_elevation <= 0:
                    print("  Warning: no beams on target at this pose. Skipping channel.")
                    channel += 1
                    current_v_rel += 16 * self.elevation_res_rad
                    continue

                full_channels_visible = n_elevation // 16
                if full_channels_visible == 0:
                    full_channels_visible = 1
                full_channels_visible = min(full_channels_visible, 8 - channel)
                elevations_for_sampling = 16 if n_elevation >= 16 else n_elevation

                microsteps_needed = self._microsteps_needed(n_azimuth, elevations_for_sampling)
                print(
                    f"  Azimuth beams: {n_azimuth}, covering {full_channels_visible} channel(s),"
                    f" microsteps: {microsteps_needed}"
                )

                for step, h_offset in enumerate(self._microstep_offsets(microsteps_needed)):
                    print(
                        f"  Position {step}: v_offset={np.degrees(current_position_v):.3f}°," 
                        f" h_offset={np.degrees(h_offset):.3f}°"
                    )
                    positions.append((h_offset, current_position_v))


                channels_covered = full_channels_visible
                channel += channels_covered
                if channel < 8:
                    macrostep_size = channels_covered * 16 * self.elevation_res_rad
                    current_v_rel += macrostep_size
                    print(
                        f"  Macrostep UP by {channels_covered * 16} elevations"
                        f" (+{np.degrees(macrostep_size):.3f}°)"
                    )

        else:
            print(f"Case: n_vertical < 16 (partial channel visible)")
            
            for channel in range(8):
                print(f"\nChannel {channel}:")
                
                # We track how many beams we have successfully processed in this channel
                beams_processed_in_channel = 0
                offset_applications = 0
                
                # Iterate until we have covered all 16 elevations in this channel
                while beams_processed_in_channel < 16:
                    
                    # We subtract the offset to move beams DOWN (or target UP relative to beams)
                    # This ensures the "leading edge" (top of the next chunk) stays safely inside the target
                    # instead of drifting off the top.
                    vertical_offset = offset_applications * (self.elevation_res_rad / 16.0) # KEY PARAMETER TO PLAY WITH!!!!!
                    current_position_v = current_v_rel - vertical_offset
                    
                    # 2. Count visible beams at this dithered position
                    n_azimuth, actual_elevations, actual_elevation_indices = self.count_beams_on_target(0, current_position_v, return_indices=True)
                    print("channel elev:", actual_elevation_indices)

                    # Map for diagnostics
                    channels = actual_elevation_indices // 16
                    elevations = actual_elevation_indices % 16
                    angle = round(np.degrees(current_position_v), 5) 
                    
                    if angle not in positions_map:
                        positions_map[angle] = (channels, elevations)

                    if actual_elevations <= 0 or n_azimuth <= 0:
                         print(f"  Warning: No beams on target at v_offset={np.degrees(current_position_v):.3f}°. Advancing by estimate...")
                         step_size_beams = max(1, n_vertical)
                         current_v_rel += step_size_beams * self.elevation_res_rad
                         beams_processed_in_channel += step_size_beams
                         offset_applications += 1
                         continue

                    # 3. Perform Microsteps
                    microsteps_needed = self._microsteps_needed(n_azimuth, actual_elevations)
                    print(
                        f"  Chunk start {beams_processed_in_channel}: "
                        f"{actual_elevations} visible. "
                        f"Microsteps {microsteps_needed} at v_offset={np.degrees(current_position_v):.3f}° "
                        f"(base={np.degrees(current_v_rel):.3f} - off={np.degrees(vertical_offset):.4f})"
                    )

                    for step, h_offset in enumerate(self._microstep_offsets(microsteps_needed)):
                        print(
                            f"    Position {step}: v_offset={np.degrees(current_position_v):.3f}°,"
                            f" h_offset={np.degrees(h_offset):.3f}°"
                        )
                        positions.append((h_offset, current_position_v))

                    # 4. Strict Stepping without overlap
                    # Since we are drifting DOWN (into the target), the top beam of the next chunk
                    # should be safely inside. We can step by the full 'actual_elevations'.
                    step_beams = actual_elevations
                    step_rad = step_beams * self.elevation_res_rad
                    
                    current_v_rel += step_rad
                    beams_processed_in_channel += step_beams
                    offset_applications += 1
                    
                    print(f"  Macrostep UP by {step_beams} beams (+{np.degrees(step_rad):.3f}°)")

                # End of channel loop. Align to exact next channel start.
                next_channel_start_target = (channel + 1) * 16 * self.elevation_res_rad
                drift = current_v_rel - next_channel_start_target
                if abs(drift) > 1e-6:
                     print(f"  Aligning next channel. Drift was {np.degrees(drift):.5f}°")
                     current_v_rel = next_channel_start_target

        
        print(f"\nTest complete! Visited {len(positions)} positions")
        return positions, positions_map


if(__name__ == "__main__"):
    # Create setup at specified distance
    setup = Test_Setup(
        target_width_m = 1.8,
        target_height_m = 1.3,
        distance_m = 100,
        sensor_height_offset_m = 0.0,
        sensor_width_offset_m = 0.0,
        num_azimuth_beams = 181,
        num_elevation_beams = 128,
        samp_per_channel = 400,
        buffer_m = 0.01,
    )
    calib = setup.compute_calibration_offset()
    setup.set_calibration(calib)
    # Get positions and coverage info
    positions, positions_info = setup.get_positions()
    positions_deg = [(np.degrees(h), np.degrees(v)) for h, v in positions]
    print("Positions (degrees):", positions_deg)
    print("\n")
    print("Positions info (covered):", positions_info)

        # Test procedure

    """
    Assume it was initially calibrated (set_calibration called) for the given distance. save calibration of initial point

    Then compute how many vertical beams are landing on the target (n_vertical)

    If n_vertical >= 16 (full channel or more in view):
        For each of 8 channels:
            Micro step to collect num_samples for the channel 
            Macro step UP by 16 elevations so next channel is at top of target
    
    If n_vertical < 16 (partial channel in view):
        For each of 8 channels:
            Calculate subdivisions_needed = ceil(16 / n_vertical)
            For each subdivision:
                Calculate how many elevations visible at this position:
                    - All positions except last: n_vertical elevations
                    - Last position: remaining elevations = 16 - (n_vertical * (subdivisions_needed - 1))
                Micro step to collect (elevations_visible / 16) * num_samples
                If not last subdivision of channel:
                    Macro step UP by n_vertical elevations so next chunk is at top of target
                Else (last subdivision):
                    Macro step UP by remaining elevations to align next channel at top of target
    
    Notes:
        - Each of the 16 elevations in a channel gets equal sampling weight
        - Example with n_vertical = 7: step by 7, 7, then 2 to complete channel and align next
        - Micro step = collect samples at current gimbal position
        - Macro step = move gimbal to next position (positive offset = tilt up = pattern moves up)
    """