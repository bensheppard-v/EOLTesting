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
        spot_diameter_m=0.0135,

        
        
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
        self.spot_diameter_m = spot_diameter_m
        self.spot_radius_m = spot_diameter_m / 2.0
        
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
        However, this function can provide the rough calculation.
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

    def _init_diagnostics(self):
        return {
            "channel_samples": np.zeros(8, dtype=float),
            "elevation_samples": np.zeros((8, 16), dtype=float),
        }

    def _sensor_index_to_channel_elev(self, sensor_idx):
        channel_idx = 7 - (sensor_idx // 16)
        # Convert matrix index (0 bottom) to channel-local index from top (0..15)
        elevation_idx = 15 - (sensor_idx % 16)
        return channel_idx, elevation_idx

    def _record_samples(self, diag, hit_indices, n_azimuth, microsteps, channel_filter=None):
        if diag is None or hit_indices is None or n_azimuth <= 0 or microsteps <= 0:
            return
        sample_amount = n_azimuth * microsteps
        for idx in hit_indices:
            channel_idx, elevation_idx = self._sensor_index_to_channel_elev(int(idx))
            if channel_idx < 0 or channel_idx >= 8:
                continue
            if channel_filter is not None and channel_idx != channel_filter:
                continue
            diag["elevation_samples"][channel_idx, elevation_idx] += sample_amount
            diag["channel_samples"][channel_idx] += sample_amount

    def _plot_diagnostics(self, diag, label=None):
        if diag is None:
            return
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available; skipping diagnostic plots.")
            return
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].bar(np.arange(8), diag["channel_samples"])
        axes[0].set_xlabel("Channel")
        axes[0].set_ylabel("Total samples")
        axes[0].set_title("Samples per channel")
        im = axes[1].imshow(diag["elevation_samples"], aspect="auto", cmap="viridis")
        axes[1].set_xlabel("Elevation index")
        axes[1].set_ylabel("Channel")
        axes[1].set_title("Samples per elevation")
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        fig.tight_layout()
        if label:
            import os
            outdir = 'photos'
            os.makedirs(outdir, exist_ok=True)
            safe_label = str(label).replace(" ", "_")
            outfile = os.path.join(outdir, f"diagnostics_{safe_label}.png")
            fig.savefig(outfile, dpi=200, bbox_inches="tight")
            print(f"Saved diagnostics plot to {outfile}")
        plt.show()

    def get_positions(self, diagnostics=False, plot_hist=False, diag_label=None):
        """
        Run the test procedure following the pseudocode.
        Returns list of relative gimbal positions visited.
        Set diagnostics=True to capture per-channel sample counts.
        """

        # Count beams at calibration
        _, n_vertical = self.count_beams_on_target(0, 0)
        if n_vertical == 0:
            print("No vertical beams on target at calibration. Abort test.")
            return []
        print(f"Starting test at calibration: {n_vertical} vertical beams visible")

        track_diag = diagnostics or plot_hist
        diag = self._init_diagnostics() if track_diag else None
        self.last_diagnostics = diag

        # Track all positions visited
        positions = []
        # Current relative vertical offset from calibration
        current_v_rel = 0.0  # Vertical offset relative to calibration
        

        # There are 2 cases. One where full channel is visible at once, and one where partial channel is visible.
        if n_vertical >= 16:
            print("Case: n_vertical >= 16 (full channel visible)")

            channel = 0
            while channel < 8:
                print(f"\nChannel {channel}:")
                current_position_v = current_v_rel
                if diag is not None:
                    n_azimuth, n_elevation, hit_indices = self.count_beams_on_target(
                        0, current_position_v, return_indices=True
                    )
                else:
                    n_azimuth, n_elevation = self.count_beams_on_target(0, current_position_v)
                    hit_indices = None
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

                if diag is not None:
                    for ch_offset in range(full_channels_visible):
                        channel_idx = channel + ch_offset
                        if channel_idx >= 8:
                            break
                        channel_hits = [
                            idx for idx in hit_indices
                            if self._sensor_index_to_channel_elev(int(idx))[0] == channel_idx
                        ]
                        self._record_samples(
                            diag,
                            channel_hits,
                            n_azimuth,
                            microsteps_needed,
                            channel_filter=channel_idx,
                        )

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
            print(
                f"  Estimated subdivisions per channel: {int(np.ceil(16 / max(1, n_vertical)))}"
            )
            
            for channel in range(8):
                print(f"\nChannel {channel}:")
                channel_start_v = current_v_rel
                offset_applications = 0
                sub_start_index = 0
                subdivision = 0
                
                while sub_start_index < 16:
                    remaining_rows = 16 - sub_start_index
                    target_elevations = min(n_vertical, remaining_rows)
                    base_position_v = channel_start_v + sub_start_index * self.elevation_res_rad
                    vertical_offset = offset_applications * (self.elevation_res_rad / 16.0) # This vertical step is another parameter we can play with. Needs to be small enough to not skip beams.
                    current_position_v = base_position_v + vertical_offset

                    # Count how many beams are actually visible
                    if diag is not None:
                        n_azimuth, actual_elevations, hit_indices = self.count_beams_on_target(
                            0, current_position_v, return_indices=True
                        )
                    else:
                        n_azimuth, actual_elevations = self.count_beams_on_target(0, current_position_v)
                        hit_indices = None
                    elevations_visible = min(target_elevations, actual_elevations)
                    if elevations_visible <= 0 or n_azimuth <= 0:
                        print(
                            f"  Subdivision {subdivision}: no beams on target (v_offset={np.degrees(current_position_v):.3f}°). Skipping."
                        )
                        sub_start_index += 1
                        offset_applications += 1
                        subdivision += 1
                        continue

                    microsteps_needed = self._microsteps_needed(n_azimuth, elevations_visible)
                    print(
                        f"  Subdivision {subdivision}: {elevations_visible} elevations, {n_azimuth} az beams,"
                        f" microsteps {microsteps_needed} at v_offset={np.degrees(current_position_v):.3f}°"
                    )

                    for step, h_offset in enumerate(self._microstep_offsets(microsteps_needed)):
                        print(
                            f"    Position {step}: v_offset={np.degrees(current_position_v):.3f}°,"
                            f" h_offset={np.degrees(h_offset):.3f}°"
                        )
                        positions.append((h_offset, current_position_v))

                    channel_hits = []
                    if diag is not None:
                        channel_hits = [
                            idx for idx in hit_indices
                            if self._sensor_index_to_channel_elev(int(idx))[0] == channel
                        ]
                        self._record_samples(
                            diag,
                            channel_hits,
                            n_azimuth,
                            microsteps_needed,
                            channel_filter=channel,
                        )

                    if diag is not None and channel_hits:
                        local_indices = [self._sensor_index_to_channel_elev(int(idx))[1] for idx in channel_hits]
                    else:
                        local_indices = [sub_start_index + i for i in range(elevations_visible)]

                    max_local = max(local_indices)
                    if max_local >= 15:
                        sub_start_index = 16
                    else:
                        overlap = 1
                        next_start = max_local + 1 - overlap
                        if next_start <= sub_start_index:
                            next_start = sub_start_index + max(1, elevations_visible - overlap)
                        sub_start_index = min(next_start, 16)
                    offset_applications += 1
                    subdivision += 1

                # After completing subdivisions, align to next channel start
                if channel < 7:
                    next_channel_offset = channel_start_v + 16 * self.elevation_res_rad
                    macrostep_size = next_channel_offset - current_v_rel
                    current_v_rel = next_channel_offset
                    print(f"  Macrostep to next channel (+{np.degrees(macrostep_size):.3f}°)")
                else:
                    current_v_rel = channel_start_v + 16 * self.elevation_res_rad
        
        print(f"\nTest complete! Visited {len(positions)} positions")

        if plot_hist:
            self._plot_diagnostics(diag, diag_label)

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
