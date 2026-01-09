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
        self.elevation_res_rad = np.radians(30) / (num_elevation_beams - 1)
        self.samp_per_channel = samp_per_channel
        self.buffer_m = buffer_m
        self.spot_diameter_m = spot_diameter_m
        self.spot_radius_m = spot_diameter_m / 2.0
        
        # Sensor FOV parameters
        self.hfov_rad = np.radians(45.0)  # 45° horizontal FOV
        self.vfov_rad = np.radians(30.0)  # 30° vertical FOV
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
    
    def calculate_microsteps_needed(self, v_offset_rel=0.0, samples_per_elevation=19):
        """
        Calculate how many horizontal microsteps are needed to achieve target samples per elevation.
        
        Parameters
        ----------
        v_offset_rel : float
            Vertical offset (radians) relative to calibration
        samples_per_elevation : int
            Desired number of samples per elevation beam (default: 19)
        
        Returns
        -------
        n_microsteps : int
            Number of microstep positions needed (including initial position)
        
        Notes
        -----
        Accounts for beams entering/exiting as we microstep by checking beam count
        at multiple positions and summing actual samples collected.
        Microsteps are always 1/32 of azimuth_res_rad apart.
        """
        microstep_size = self.azimuth_res_rad / 32.0
        
        # Start with initial estimate
        n_azimuth_initial, _ = self.count_beams_on_target(0.0, v_offset_rel)
        
        if n_azimuth_initial <= 0:
            return 1  # At least one position even if no beams hit
        
        # Iteratively check if we have enough samples
        for n_microsteps in range(1, 50):  # Reasonable upper limit
            total_samples = 0
            for i in range(n_microsteps):
                h_offset = i * microstep_size
                n_azimuth, _ = self.count_beams_on_target(h_offset, v_offset_rel)
                total_samples += n_azimuth
            
            if total_samples >= samples_per_elevation:
                return n_microsteps
        
        # Fallback: use initial estimate if loop doesn't converge
        return int(np.ceil(samples_per_elevation / n_azimuth_initial))
    
    def generate_microstep_positions(self, n_microsteps, v_offset_rel):
        """
        Generate list of (h_offset, v_offset) positions for microstepping.
        
        Parameters
        ----------
        n_microsteps : int
            Number of microstep positions to generate
        v_offset_rel : float
            Vertical offset (radians) relative to calibration
        
        Returns
        -------
        positions : list of tuples
            List of (h_offset_rad, v_offset_rad) positions
        
        Notes
        -----
        Microsteps are always 1/32 of azimuth_res_rad apart horizontally.
        First position is always at h_offset = 0.0.
        """
        microstep_size = self.azimuth_res_rad / 32.0
        positions = []
        
        for i in range(n_microsteps):
            h_offset = i * microstep_size
            positions.append((h_offset, v_offset_rel))
        
        return positions
    
    def run_test(self, samples_per_elevation=19):
        """
        Run the test procedure with automatic microstepping.
        
        Parameters
        ----------
        samples_per_elevation : int
            Target number of samples per elevation beam (default: 19)
        
        Returns
        -------
        positions : list of tuples
            List of (h_offset_rad, v_offset_rad) gimbal positions visited
        """
        # Count beams at calibration
        n_azimuth, n_vertical = self.count_beams_on_target(0, 0)
        print(f"Starting test at calibration:")
        print(f"  Horizontal beams: {n_azimuth}")
        print(f"  Vertical beams: {n_vertical}")
        
        # Calculate microsteps needed (will check beam count at each position)
        n_microsteps = self.calculate_microsteps_needed(0.0, samples_per_elevation)
        
        # Calculate actual samples that will be collected
        microstep_size = self.azimuth_res_rad / 32.0
        total_samples = sum([self.count_beams_on_target(i * microstep_size, 0.0)[0] for i in range(n_microsteps)])
        
        print(f"  Microsteps per position: {n_microsteps}")
        print(f"  Samples per elevation: {total_samples}\n")
        
        # Track all positions visited
        positions = []
        current_v_rel = 0.0  # Vertical offset relative to calibration
        
        if n_vertical >= 16:
            print("Case: n_vertical >= 16 (full channel visible)")
            # Full channel visible at once
            for channel in range(8):
                print(f"\nChannel {channel}:")
                
                # Recalculate microsteps for this specific v_offset (beam count may vary)
                n_microsteps_ch = self.calculate_microsteps_needed(current_v_rel, samples_per_elevation)
                
                # Generate microstep positions for this channel
                channel_positions = self.generate_microstep_positions(n_microsteps_ch, current_v_rel)
                positions.extend(channel_positions)
                
                # Calculate actual samples for this channel
                total_samples_ch = sum([self.count_beams_on_target(i * microstep_size, current_v_rel)[0] 
                                       for i in range(n_microsteps_ch)])
                
                print(f"  Added {len(channel_positions)} positions at v_offset={np.degrees(current_v_rel):.3f}°")
                print(f"  Total samples for this channel: {total_samples_ch}")
                
                # Macrostep UP by 16 elevations (except after last channel)
                if channel < 7:
                    macrostep_size = 16 * self.elevation_res_rad
                    current_v_rel += macrostep_size
                    print(f"  Macrostep UP by 16 elevations (+{np.degrees(macrostep_size):.3f}°)")

        else:
            print(f"Case: n_vertical < 16 (partial channel visible)")
            # Partial channel visible - need subdivisions
            subdivisions_needed = int(np.ceil(16 / n_vertical))
            print(f"Subdivisions per channel: {subdivisions_needed}")
            
            for channel in range(8):
                print(f"\nChannel {channel}:")
                
                for sub in range(subdivisions_needed):
                    # Calculate elevations visible at this subdivision
                    if sub < subdivisions_needed - 1:
                        elevations_visible = n_vertical
                    else:
                        elevations_visible = 16 - (n_vertical * (subdivisions_needed - 1))
                    
                    print(f"  Subdivision {sub}: {elevations_visible} elevations visible")
                    
                    # Recalculate microsteps for this specific position
                    n_microsteps_sub = self.calculate_microsteps_needed(current_v_rel, samples_per_elevation)
                    
                    # Generate microstep positions for this subdivision
                    subdivision_positions = self.generate_microstep_positions(n_microsteps_sub, current_v_rel)
                    positions.extend(subdivision_positions)
                    
                    # Calculate actual samples for this subdivision
                    total_samples_sub = sum([self.count_beams_on_target(i * microstep_size, current_v_rel)[0] 
                                            for i in range(n_microsteps_sub)])
                    
                    print(f"    Added {len(subdivision_positions)} positions at v_offset={np.degrees(current_v_rel):.3f}°")
                    print(f"    Total samples for this subdivision: {total_samples_sub}")
                    
                    # Macrostep UP
                    if sub < subdivisions_needed - 1:
                        # Not last subdivision - step by n_vertical
                        macrostep_size = n_vertical * self.elevation_res_rad
                        current_v_rel += macrostep_size
                        print(f"    Macrostep UP by {n_vertical} elevations (+{np.degrees(macrostep_size):.3f}°)")
                    elif channel < 7:
                        # Last subdivision but not last channel - step by remaining elevations
                        macrostep_size = elevations_visible * self.elevation_res_rad
                        current_v_rel += macrostep_size
                        print(f"    Macrostep UP by {elevations_visible} elevations (+{np.degrees(macrostep_size):.3f}°)")
        
        print(f"\n✓ Test complete! Visited {len(positions)} positions")
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
