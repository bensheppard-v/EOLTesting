import numpy as np

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
        spot_diameter_m=0.0135,

        
        
        ):
        self.target_width_m = target_width_m
        self.target_height_m = target_height_m
        self.distance_m = distance_m
        self.sensor_height_offset_m = sensor_height_offset_m
        self.sensor_width_offset_m = sensor_width_offset_m
        self.num_azimuth_beams = num_azimuth_beams
        self.num_elevation_beams = num_elevation_beams
        self.azimuth_res_rad = np.radians(45) / (num_azimuth_beams - 1)
        self.elevation_res_rad = np.radians(22.5) / (num_elevation_beams - 1)
        self.samp_per_channel = samp_per_channel
        self.buffer_m = buffer_m
        self.spot_diameter_m = spot_diameter_m
        self.spot_radius_m = spot_diameter_m / 2.0
        
        # Sensor FOV parameters
        self.hfov_rad = np.radians(45.0)  # 45° horizontal FOV
        self.vfov_rad = np.radians(22.5)  # 22.5° vertical FOV
        self.theta0 = 0.0  # Uncalibrated center elevation angle (radians)

        # Gimbal calibration offsets (will be set manually for each distance test)
        self.gimbal_h_offset_rad = 0.0
        self.gimbal_v_offset_rad = 0.0

    def _angular_limits(self):
        """Return half-angles of sensor FOV in radians."""
        phi_half = self.hfov_rad / 2.0
        theta_half = self.vfov_rad / 2.0
        return phi_half, theta_half
    
    def compute_calibration_offset(self):
        """Compute gimbal offset to align highest beam with top of target.
        
        note: I will need to do this manually for each distance in the test script.
        However, this function can provide the rough calcualion. I also a
        """
        _, theta_half = self._angular_limits()
        # Align highest beam with actual top of target (
        target_top = self.target_height_m / 2.0
        angle_to_top = np.arctan((target_top - self.sensor_height_offset_m) / self.distance_m)
        dtheta_calib = angle_to_top - (self.theta0 + theta_half) - np.arctan(self.buffer_m / self.distance_m)
        return 0.0, dtheta_calib # Horizontal offset is zero for this setup
   
    def set_calibration(self, calib_offset):
        """Set gimbal calibration offsets in radians."""
        dphi_calib, dtheta_calib = calib_offset # Unpack calib_offset tuple
        self.gimbal_h_offset_rad = dphi_calib
        self.gimbal_v_offset_rad = dtheta_calib
    
    def azimuth_angles(self, dphi_offset=0.0):
        """Generate all azimuth (horizontal) beam angles in radians."""
        phi_half, _ = self._angular_limits()
        return np.linspace(-phi_half, phi_half, self.num_azimuth_beams) + dphi_offset
    
    def elevation_angles(self, dtheta_offset=0.0):
        """Generate all elevation (vertical) beam angles in radians."""
        _, theta_half = self._angular_limits()
        return self.theta0 + np.linspace(-theta_half, theta_half, self.num_elevation_beams) + dtheta_offset
    
    def count_beams_on_target(self, dphi_rel=0.0, dtheta_rel=0.0):
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
        
        n_azimuth = int(np.sum(azimuth_hits))
        n_elevation = int(np.sum(elevation_hits))
        
        return n_azimuth, n_elevation
    
    def microstep_case1(self):
        """
          Case 1: n_vertical >= 16 (full channel visible)
        Should microstep until samp_per_channel is reached

        """

    def run_test(self):
        """
        Run the test procedure following the pseudocode.
        For now: simplified microstepping (one step to the right).
        Returns list of gimbal positions visited.
        """
        # Store initial calibration
        initial_h_offset = self.gimbal_h_offset_rad
        initial_v_offset = self.gimbal_v_offset_rad
        
        # Count vertical beams at calibration
        _, n_vertical = self.count_beams_on_target(0, 0)
        print(f"Starting test at calibration: {n_vertical} vertical beams visible")
        
        # Track all positions visited
        positions = []
        # Current relative offset from calibration
        current_v_rel = 0.0  # Vertical offset relative to calibration
        
        if n_vertical >= 16:
            print("Case: n_vertical >= 16 (full channel visible)")
            
            # Calculate how many microsteps needed so one elevation gets 19 samples
            # Count azimuth beams hitting at calibration position
            n_azimuth, _ = self.count_beams_on_target(0, 0)
            # Need 19 samples per elevation, each position gives n_azimuth samples per elevation
            microsteps_needed = max(1, int(np.ceil(self.samp_per_channel / 16 / n_azimuth)))
            print(f"Azimuth beams per position: {n_azimuth}, Microsteps needed: {microsteps_needed} (to get {self.samp_per_channel / 16:.0f} samples per elevation)")
            
            # Full channel visible at once
            for channel in range(8):
                print(f"\nChannel {channel}:")
                
                # Multiple microsteps at this channel position
                for step in range(microsteps_needed):
                    h_offset = step * self.azimuth_res_rad / 20.0
                    print(f"  Position {step}: v_offset={np.degrees(current_v_rel):.3f}°, h_offset={np.degrees(h_offset):.3f}°")
                    positions.append((h_offset, current_v_rel))
                
                # Macrostep UP by 16 elevations (except after last channel)
                if channel < 7:
                    macrostep_size = 16 * self.elevation_res_rad
                    current_v_rel += macrostep_size
                    print(f"  Macrostep UP by 16 elevations (+{np.degrees(macrostep_size):.3f}°)")

        else:
            print(f"Case: n_vertical < 16 (partial channel visible)")
            # Partial channel visible - need subdivisions
            subdivisions_needed = int(np.ceil(16 / n_vertical))
            # Calculate microsteps needed: 19 samples per elevation line
            # Count azimuth beams hitting at calibration
            n_azimuth, _ = self.count_beams_on_target(0, 0)
            samples_per_elevation = self.samp_per_channel / 16  # 300/16 = 18.75 ≈ 19
            microsteps_per_subdivision = max(1, int(np.ceil(samples_per_elevation / n_azimuth)))
            print(f"Azimuth beams per position: {n_azimuth}, Microsteps per subdivision: {microsteps_per_subdivision} (to get {samples_per_elevation:.0f} samples per elevation)")
            
            for channel in range(8):
                print(f"\nChannel {channel}:")
                
                for sub in range(subdivisions_needed):
                    # Apply small vertical offset between subdivisions within channel
                    # This prevents same elevations from landing on same spots
                    v_offset_adjustment = sub * self.elevation_res_rad / 64
                    current_position_v = current_v_rel + v_offset_adjustment
                    
                    # Count how many elevations are ACTUALLY visible at this position
                    elevations_visible, _ = self.count_beams_on_target(0, current_position_v)
                    
                    print(f"  Subdivision {sub}: {elevations_visible} elevations visible at v_offset={np.degrees(current_position_v):.3f}°")
                    
                    # Multiple microsteps to reach sample target
                    for step in range(microsteps_per_subdivision):
                        h_offset = step * self.azimuth_res_rad / 20.0
                        print(f"    Position {step}: v_offset={np.degrees(current_position_v):.3f}°, h_offset={np.degrees(h_offset):.3f}°")
                        positions.append((h_offset, current_position_v))
                    
                    # Macrostep UP - overlap by 1 elevation to avoid losing samples at edges
                    if sub < subdivisions_needed - 1:
                        # Within channel: step by (elevations_visible - 1) to create overlap
                        # But if only 1 elevation, can't create overlap - just step by 1
                        overlap_step = max(1, elevations_visible - 1)
                        macrostep_size = overlap_step * self.elevation_res_rad
                        current_v_rel += macrostep_size
                        if overlap_step < elevations_visible:
                            print(f"    Macrostep UP by {overlap_step} elevations (+{np.degrees(macrostep_size):.3f}°) [1 elevation overlap]")
                        else:
                            print(f"    Macrostep UP by {overlap_step} elevations (+{np.degrees(macrostep_size):.3f}°)")
                    elif channel < 7:
                        # Between channels: step by full elevations_visible to align next channel
                        macrostep_size = elevations_visible * self.elevation_res_rad
                        current_v_rel += macrostep_size
                        print(f"    Macrostep UP by {elevations_visible} elevations to next channel (+{np.degrees(macrostep_size):.3f}°)")
        
        print(f"\nTest complete! Visited {len(positions)} positions")
        return positions

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
