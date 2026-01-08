import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class FlatTargetHitCounter:
    """
    FMCW LiDAR sampling on a flat rectangular target with gimbal autofill.
    
    =============================================================================
    OVERVIEW
    =============================================================================
    
    This class models a LiDAR sensor with:
    - 128 elevation beams organized into 8 vertical channels (16 elevations each)
    - 181 azimuth beams
    - Fixed horizontal FOV (default 45°) and vertical FOV (default 30°)
    - Gimbal control for micro/macro positioning
    
    GOAL: Collect 300 samples per channel with equal representation from all 
          16 elevations within that channel (~19 samples per elevation).
    
    =============================================================================
    WORKFLOW
    =============================================================================
    
    1. CALIBRATION:
       - Manually align the TOP of the sensor's vertical FOV with the TOP of target
       - Record this gimbal position as calibration reference (0, 0)
       - All subsequent autofill offsets are relative to this position
    
    2. AUTOFILL STRATEGY:
       
       CLOSE TARGETS (many elevations visible):
         → Microstep within visible range to densify samples
         
       FAR TARGETS (few elevations visible):
         → Microstep visible elevations to get their share of samples
         → Macrostep vertically to bring new elevation groups onto target
         → Repeat until all 16 elevations per channel are adequately sampled
    
    3. OUTPUT:
       - Relative offsets: gimbal movements from calibration position
       - Absolute offsets: actual gimbal commands (includes calibration)
       - Per-bin counts: samples organized by (channel, elevation)
    
    =============================================================================

    Parameters
    ----------
    target_width_m : float 
        Width of the target in metres.
    target_height_m : float
        Height of the target in metres.
    distance_m : float  
        Distance to the target in metres.
    sensor_height_offset_m : float
        Vertical offset of the sensor from the target center in metres.
    sensor_width_offset_m : float
        Horizontal offset of the sensor from the target center in metres.
    azimuth_res_rad : float
        Azimuth resolution in radians.
    elevation_res_rad : float
        Elevation resolution in radians.
    elevation_center_rad : float, optional
        Center elevation angle in radians (default is 0.0).
    num_channels : int, optional
        Number of azimuth channels (default is 8).
    elevations_per_channel : int, optional
        Number of elevation beams per channel (default is 16).
    sensor_hfov_deg : float, optional
        Horizontal field of view in degrees (default is 45.0).
    sensor_vfov_deg : float, optional
        Vertical field of view in degrees (default is 30.0).
    """

    def __init__(
        self,
        target_width_m,
        target_height_m,
        distance_m,
        sensor_height_offset_m,
        sensor_width_offset_m, 
        azimuth_res_rad,
        elevation_res_rad,
        elevation_center_rad=0.0,
        num_channels=8,
        elevations_per_channel=16,
        sensor_hfov_deg=45.0,
        sensor_vfov_deg=30.0,
    ):
        self.W = float(target_width_m)
        self.H = float(target_height_m)
        self.D = float(distance_m)
        self.sensor_height_offset_m = float(sensor_height_offset_m)
        self.sensor_width_offset_m = float(sensor_width_offset_m)
        self.dphi = float(azimuth_res_rad)
        self.dtheta = float(elevation_res_rad)
        self.theta0 = float(elevation_center_rad)
        self.num_channels = int(num_channels)
        self.elevations_per_channel = int(elevations_per_channel)
        
        # Sensor fixed field-of-view (convert degrees to radians)
        self.hfov_rad = np.radians(sensor_hfov_deg)
        self.vfov_rad = np.radians(sensor_vfov_deg)
        
        # Calibration tracking: gimbal position where top-of-FOV aligns with top-of-target
        self.gimbal_dphi_calib = 0.0  # Calibration azimuth offset
        self.gimbal_dtheta_calib = 0.0  # Calibration elevation offset
        self.is_calibrated = False

    # ------------------------------------------------------------
    # Angular grid
    # ------------------------------------------------------------

    # phi is azimuth, theta is elevation
    def _angular_limits(self):
        """Return half-angles of sensor's fixed field-of-view."""
        phi_half = self.hfov_rad / 2.0
        theta_half = self.vfov_rad / 2.0
        return phi_half, theta_half

    def azimuth_angles(self, dphi_offset=0.0):
        phi_half, _ = self._angular_limits()
        # Generate exactly 181 beams spanning the full horizontal FOV
        # Beams at edges: -22.5° to +22.5°
        expected_beams = int(np.round(2 * phi_half / self.dphi)) + 1  # +1 for edge-inclusive
        return np.linspace(-phi_half, phi_half, expected_beams) + dphi_offset

    def elevation_angles(self, dtheta_offset=0.0):
        _, theta_half = self._angular_limits()
        # Generate exactly 128 beams spanning the full vertical FOV
        # Beams at edges: -15° to +15° (relative to theta0)
        expected_beams = int(np.round(2 * theta_half / self.dtheta)) + 1  # +1 for edge-inclusive
        return self.theta0 + np.linspace(-theta_half, theta_half, expected_beams) + dtheta_offset

    # ------------------------------------------------------------
    # Channel and elevation mapping
    # ------------------------------------------------------------

    def get_channel_bounds(self):
        """Return elevation range (min, max) for each channel (indices 0-15 within each channel)."""
        # Channels are organized vertically: 8 channels × 16 elevations each
        bounds = []
        for ch in range(self.num_channels):
            elev_min = ch * self.elevations_per_channel
            elev_max = (ch + 1) * self.elevations_per_channel - 1
            bounds.append((elev_min, elev_max))
        return bounds

    def assign_to_channel(self, theta):
        """Assign elevation angle(s) to channel index 0..num_channels-1.
        
        Channel 0 = HIGHEST elevation angles (top of FOV)
        Channel 7 = LOWEST elevation angles (bottom of FOV)
        """
        # Channels are vertical: split 128 total elevations into 8 groups of 16
        _, theta_half = self._angular_limits()
        theta_span = 2 * theta_half
        channel_height = theta_span / self.num_channels
        
        # Theta ranges from theta0 - theta_half to theta0 + theta_half
        theta_max = self.theta0 + theta_half  # Highest angle
        
        # Handle both scalar and array inputs
        theta_arr = np.asarray(theta)
        # Invert: highest angles → channel 0, lowest angles → channel 7
        channel = np.floor((theta_max - theta_arr) / channel_height).astype(int)
        channel = np.clip(channel, 0, self.num_channels - 1)
        
        return channel if theta_arr.shape else int(channel)

    def get_elevation_index(self, theta):
        """Map elevation angle(s) to elevation index 0..elevations_per_channel-1 within its channel.
        
        Within each channel:
        Elevation 0 = HIGHEST angle in that channel
        Elevation 15 = LOWEST angle in that channel
        """
        _, theta_half = self._angular_limits()
        theta_span = 2 * theta_half
        bin_height = theta_span / (self.num_channels * self.elevations_per_channel)
        
        # Theta ranges from theta0 - theta_half to theta0 + theta_half
        theta_max = self.theta0 + theta_half
        
        # Get global elevation index (0-127), inverted so highest angle = 0
        theta_arr = np.asarray(theta)
        global_elev_idx = np.floor((theta_max - theta_arr) / bin_height).astype(int)
        global_elev_idx = np.clip(global_elev_idx, 0, self.num_channels * self.elevations_per_channel - 1)
        
        # Convert to local index within channel (0-15)
        local_elev_idx = global_elev_idx % self.elevations_per_channel
        
        return local_elev_idx if theta_arr.shape else int(local_elev_idx)

    def get_max_elevation_offset_within_channel(self):
        """Maximum elevation offset to stay within current channel (16 beams)."""
        _, theta_half = self._angular_limits()
        theta_span = 2 * theta_half
        channel_height = theta_span / self.num_channels
        return channel_height / 2.0  # Half channel height to stay centered in channel

    # ------------------------------------------------------------
    # Calibration methods
    # ------------------------------------------------------------

    def set_calibration(self, gimbal_dphi_calib, gimbal_dtheta_calib):
        """
        Record the gimbal offset corresponding to top-of-FOV aligned with top-of-target.
        
        This calibration point becomes the reference (0, 0) for all subsequent autofill operations.
        
        Parameters
        ----------
        gimbal_dphi_calib : float
            Gimbal azimuth offset (radians) at calibration.
        gimbal_dtheta_calib : float
            Gimbal elevation offset (radians) at calibration (top-of-target alignment).
        """
        self.gimbal_dphi_calib = float(gimbal_dphi_calib)
        self.gimbal_dtheta_calib = float(gimbal_dtheta_calib)
        self.is_calibrated = True

    def get_calibration(self):
        """Return the current calibration offset."""
        return (self.gimbal_dphi_calib, self.gimbal_dtheta_calib)

    def relative_to_absolute_offset(self, dphi_rel, dtheta_rel):
        """
        Convert relative offsets (from autofill) to absolute gimbal commands.
        
        Parameters
        ----------
        dphi_rel : float or array
            Relative azimuth offset(s) from autofill.
        dtheta_rel : float or array
            Relative elevation offset(s) from autofill.
        
        Returns
        -------
        dphi_abs : float or array
            Absolute gimbal azimuth command(s).
        dtheta_abs : float or array
            Absolute gimbal elevation command(s).
        """
        dphi_abs = self.gimbal_dphi_calib + dphi_rel
        dtheta_abs = self.gimbal_dtheta_calib + dtheta_rel
        return dphi_abs, dtheta_abs

    def absolute_to_relative_offset(self, dphi_abs, dtheta_abs):
        """
        Convert absolute gimbal commands back to relative offsets (for verification).
        
        Parameters
        ----------
        dphi_abs : float or array
            Absolute gimbal azimuth command(s).
        dtheta_abs : float or array
            Absolute gimbal elevation command(s).
        
        Returns
        -------
        dphi_rel : float or array
            Relative azimuth offset(s) from calibration.
        dtheta_rel : float or array
            Relative elevation offset(s) from calibration.
        """
        dphi_rel = dphi_abs - self.gimbal_dphi_calib
        dtheta_rel = dtheta_abs - self.gimbal_dtheta_calib
        return dphi_rel, dtheta_rel

    # ------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------

    def project_to_target(self, dphi_offset=0.0, dtheta_offset=0.0):
        phis = self.azimuth_angles(dphi_offset)
        thetas = self.elevation_angles(dtheta_offset)

        PHI, THETA = np.meshgrid(phis, thetas)

        X = self.D * np.tan(PHI)
        Y = self.D * np.tan(THETA)

        return X, Y

    def project_with_angles(self, dphi_offset=0.0, dtheta_offset=0.0):
        """Project to target and also return angle grids for channel/elevation assignment."""
        phis = self.azimuth_angles(dphi_offset)
        thetas = self.elevation_angles(dtheta_offset)

        PHI, THETA = np.meshgrid(phis, thetas)

        X = self.D * np.tan(PHI)
        Y = self.D * np.tan(THETA)

        return X, Y, PHI, THETA

    # ------------------------------------------------------------
    # Mask + counts
    # ------------------------------------------------------------

    def inside_mask(self, X, Y, margin_m=0.0):
        """Return boolean mask for points inside the target minus an optional border margin.

        Parameters
        - X, Y: arrays of coordinates on target (metres)
        - margin_m: margin from the rectangle edges to exclude (metres)
        """
        half_w = max(0.0, self.W / 2 - float(margin_m))
        half_h = max(0.0, self.H / 2 - float(margin_m))
        return (np.abs(X) <= half_w) & (np.abs(Y) <= half_h)


    def identify_reachable_channels(self, margin_m=0.0):
        """Determine which elevation-based channels have at least one beam hitting the target."""
        X, Y, PHI, THETA = self.project_with_angles()
        mask = self.inside_mask(X, Y, margin_m=margin_m)
        
        if not np.any(mask):
            return []
        
        # Get elevation angles for all beams that hit
        theta_hit = THETA[mask]
        channels_hit = self.assign_to_channel(theta_hit)
        
        # Return sorted unique channels
        return sorted(np.unique(channels_hit).tolist())
    def total_points_single_frame(self):
        X, Y = self.project_to_target()
        return int(np.sum(self.inside_mask(X, Y)))

    def azimuth_points_per_elevation(self):
        X, Y = self.project_to_target()
        mask = self.inside_mask(X, Y)

        counts = mask.sum(axis=1)
        thetas = self.elevation_angles()

        return {thetas[i]: int(counts[i]) for i in range(len(thetas)) if counts[i] > 0}

    # ------------------------------------------------------------
    # Gimbal autofill (core requirement)
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # Per-channel, per-elevation autofill (MAIN METHOD)
    # ------------------------------------------------------------

    def autofill_per_channel_elevation(
        self,
        samples_per_bin=19,
        max_fine_subdiv=16,
        max_coarse_steps=20,
        margin_m=0.0,
        spot_radius_m=0.0,
        tolerance=2,
    ):
        """
        Autofill samples to achieve uniform coverage across all channels and elevations.
        
        STRATEGY:
        After calibration (top of FOV aligned with top of target), we start at offset (0,0).
        
        For CLOSE targets (many elevations visible):
          - Microstep within the visible elevations to densify them
          
        For FAR targets (few elevations visible):
          - Microstep the visible elevations to densify them
          - Then MACROSTEP vertically to bring new elevations onto target
          - Repeat until all 16 elevations per channel have adequate samples
        
        The code proceeds in stages:
          1. BASE: Sample at calibration position (0, 0)
          2. MICROSTEP: Fine azimuth/elevation offsets to densify visible elevations
          3. MACROSTEP: Large elevation shifts to scan through all elevations
        
        Goal: 300 samples per channel = ~19 samples per elevation (16 elevations/channel)

        Parameters
        ----------
        samples_per_bin : int
            Target samples per (channel, elevation) bin (default 19 for 300÷16).
        max_fine_subdiv : int
            Maximum microstepping subdivisions (default 16).
        max_coarse_steps : int
            Maximum macrostepping elevation shifts (default 20).
        margin_m : float
            Border margin to exclude points near edge (metres).
        spot_radius_m : float
            Laser spot radius - spots closer than 2×radius are rejected (metres).
        tolerance : int
            Allow bins to exceed target by this many samples (default 2).

        Returns
        -------
        offsets_rel : list[(dphi, dtheta)]
            Relative offsets from calibration position.
        offsets_abs : list[(dphi, dtheta)]
            Absolute gimbal commands (includes calibration offset).
        samples : dict[channel][elev_idx] -> list[(x, y)]
            All samples organized by channel and elevation.
        counts : dict[channel][elev_idx] -> int
            Count per bin.
        reachable_channels : list[int]
            Channels that can reach target (0-7).
        summary : dict
            Statistics: total_samples, bins_satisfied, completion_rate, etc.
        """
        print(f"DEBUG START: samples_per_bin={samples_per_bin}, tolerance={tolerance}")
        import sys; sys.stdout.flush()
        
        # ============================================================
        # INITIALIZATION
        # ============================================================
        # Initialize ALL 8 channels (not just initially reachable ones)
        # We'll make other channels reachable by macrostepping the gimbal
        all_channels = list(range(self.num_channels))
        
        # Storage: samples[channel][elevation_index] = [(x, y), ...]
        samples = {ch: {e: [] for e in range(self.elevations_per_channel)} 
                   for ch in all_channels}
        counts = {ch: {e: 0 for e in range(self.elevations_per_channel)} 
                  for ch in all_channels}
        
        # Track all accepted points using spatial grid for fast lookup
        offsets = []
        min_sep = 2.0 * float(spot_radius_m) if spot_radius_m > 0.0 else 0.0
        
        # Spatial hash grid: divide space into cells of size min_sep
        # Each cell stores list of points in that cell
        from collections import defaultdict
        grid = defaultdict(list)
        cell_size = max(min_sep, 0.01) if min_sep > 0 else 0.01
        
        def _get_cell(x, y):
            """Get grid cell for point."""
            return (int(x / cell_size), int(y / cell_size))
        
        def _accept(xp, yp):
            """Check if point doesn't overlap with existing points using spatial grid."""
            if min_sep <= 0.0:
                return True
            
            # Check only nearby cells (9 cells: current + 8 neighbors)
            cx, cy = _get_cell(xp, yp)
            min_sep2 = min_sep * min_sep
            
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    cell = (cx + dx, cy + dy)
                    for xa, ya in grid[cell]:
                        if (xa - xp)**2 + (ya - yp)**2 < min_sep2:
                            return False
            return True
        
        def _add_sample(xp, yp, ch, elev):
            """Try to add sample to bin if bin needs more and point doesn't overlap."""
            if counts[ch][elev] >= samples_per_bin + tolerance:
                return False
            if _accept(xp, yp):
                samples[ch][elev].append((xp, yp))
                counts[ch][elev] += 1
                # Add to spatial grid
                cell = _get_cell(xp, yp)
                grid[cell].append((xp, yp))
                return True
            return False
        
        def _all_bins_satisfied():
            """Check if all (channel, elevation) bins have reached target."""
            for ch in all_channels:
                for e in range(self.elevations_per_channel):
                    if counts[ch][e] < samples_per_bin:
                        return False
            print(f"DEBUG: All bins satisfied! samples_per_bin={samples_per_bin}, Total offsets: {len(offsets)}")
            import sys; sys.stdout.flush()
            return True
        
        def _process_frame(dphi, dtheta):
            """
            Process one gimbal position: project beams, check which hit target,
            assign to (channel, elevation) bins, and add non-overlapping samples.
            
            The key insight: channel assignment must be based on the PHYSICAL BEAM INDEX (0-127),
            not the angle where it points. The meshgrid row index IS the beam index.
            
            Parameters dphi, dtheta are RELATIVE offsets from calibration position.
            We convert to absolute before projecting.
            
            Returns number of samples added.
            """
            # Convert relative offset to absolute gimbal position
            dphi_abs, dtheta_abs = self.relative_to_absolute_offset(dphi, dtheta)
            
            X, Y, PHI, THETA = self.project_with_angles(dphi_abs, dtheta_abs)
            mask = self.inside_mask(X, Y, margin_m=margin_m)
            
            if not np.any(mask):
                return 0
            
            # Get coordinates of beams that hit target
            xs = X[mask].ravel()
            ys = Y[mask].ravel()
            
            # CRITICAL: Get the beam indices (row, col) from the mask
            # The row index (elevation beam index) determines the channel!
            beam_indices = np.where(mask)
            elev_beam_idx = beam_indices[0]  # Row indices = elevation beam 0-127
            
            # Map beam index to (channel, elevation_in_channel)
            # Beam 0 = LOWEST angle (-15°) = channel 7, elevation 15
            # Beam 127 = HIGHEST angle (+15°) = channel 0, elevation 0
            # INVERT: highest beams → channel 0
            channels = (self.num_channels - 1) - (elev_beam_idx // self.elevations_per_channel)
            elevs = (self.elevations_per_channel - 1) - (elev_beam_idx % self.elevations_per_channel)
            
            added = 0
            for x, y, ch, elev in zip(xs, ys, channels, elevs):
                # Accept samples from any channel (0-7)
                if _add_sample(x, y, ch, elev):
                    added += 1
            
            return added
        
        # ============================================================
        # STAGE 1: BASE FRAME (calibration position)
        # ============================================================
        # Start at (0, 0) relative offset - the calibrated position
        # where top of FOV aligns with top of target
        added = _process_frame(0.0, 0.0)
        if added > 0:
            offsets.append((0.0, 0.0))
        
        if _all_bins_satisfied():
            return self._build_return(offsets, samples, counts, all_channels)
        
        # ============================================================
        # STAGE 2: MICROSTEPPING (densify visible elevations)
        # ============================================================
        # Microstep within currently-visible elevation range
        # This works well when target is close and many elevations are visible
        
        # Elevation constraint: stay within same channel to avoid cross-channel issues
        max_elevation_offset = self.get_max_elevation_offset_within_channel()
        
        n_elev_micro = max_fine_subdiv
        n_azi_micro = max_fine_subdiv
        
        dtheta_micro = max_elevation_offset / max(1, n_elev_micro)
        dphi_micro = self.dphi / (n_azi_micro + 1)
        
        for j in range(n_elev_micro + 1):
            for i in range(-n_azi_micro, n_azi_micro + 1):
                if i == 0 and j == 0:
                    continue  # Already did (0, 0) in base frame
                
                dtheta = j * dtheta_micro
                dphi = i * dphi_micro
                
                added = _process_frame(dphi, dtheta)
                if added > 0:
                    offsets.append((dphi, dtheta))
                
                if _all_bins_satisfied():
                    return self._build_return(offsets, samples, counts, all_channels)
        
        # ============================================================
        # STAGE 3: MACROSTEPPING (scan through all elevations)
        # ============================================================
        # For far targets, only a few elevations are visible at any gimbal position.
        # Make large vertical steps to bring different elevation groups onto target.
        # At each macro position, do microstepping to densify.
        
        # Calculate how many elevations (beams) hit the target at once
        vfov_height_at_distance = 2.0 * self.D * np.tan(self.vfov_rad / 2.0)
        num_elevation_beams = self.num_channels * self.elevations_per_channel  # 128
        elevations_visible = (self.H / vfov_height_at_distance) * num_elevation_beams
        
        # Calculate how many azimuth beams hit the target width at once
        hfov_width_at_distance = 2.0 * self.D * np.tan(self.hfov_rad / 2.0)
        phi_half, theta_half = self._angular_limits()
        num_azimuth_beams = int(np.round(2 * phi_half / self.dphi)) + 1  # 181
        azimuths_visible = (self.W / hfov_width_at_distance) * num_azimuth_beams
        
        # Calculate angular spacing per elevation
        theta_range = self.elevation_angles().max() - self.elevation_angles().min()
        angular_spacing_per_elevation = theta_range / num_elevation_beams
        
        # Step by the number of elevations visible at once
        # This moves to the next non-overlapping group of elevations
        macro_step_size = elevations_visible * angular_spacing_per_elevation
        
        # Calculate how many steps needed to scan all elevations
        steps_needed = int(np.ceil(num_elevation_beams / elevations_visible))
        max_empty_macrosteps = max(3, steps_needed // 2)  # Allow some empty steps
        consecutive_empty_macrosteps = 0
        
        # Adjust microstepping based on how many beams are visible
        # Fewer beams visible → MORE microstepping needed (inverse relationship)
        elevations_per_channel = num_elevation_beams / self.num_channels  # = 16
        
        # Scale inversely: if only 25% of elevations visible, use more microstepping
        elev_visibility_ratio = elevations_visible / num_elevation_beams
        azi_visibility_ratio = azimuths_visible / num_azimuth_beams
        
        # Inverse scaling: fewer visible → more microsteps
        n_elev_micro_macro = max(2, int(max_fine_subdiv / max(0.1, elev_visibility_ratio)))
        n_azi_micro_macro = max(2, int(max_fine_subdiv / max(0.1, azi_visibility_ratio)))
        
        # Cap at max_fine_subdiv to avoid excessive computation
        n_elev_micro_macro = min(n_elev_micro_macro, max_fine_subdiv)
        n_azi_micro_macro = min(n_azi_micro_macro, max_fine_subdiv)
        
        print(f"DEBUG: elev_visible={elevations_visible:.1f}, azi_visible={azimuths_visible:.1f}, macro_step={np.degrees(macro_step_size):.2f}°, steps_needed={steps_needed}, elev_micro={n_elev_micro_macro}, azi_micro={n_azi_micro_macro}")
        import sys; sys.stdout.flush()
        
        # At calibration, channels 0-1 are visible (top of target).
        # Need to explore BOTH directions since only middle is initially visible:
        # - Step DOWN (-) to bring lower beam indices (higher channels) onto target  
        # - Step UP (+) may be needed depending on initial coverage
        # Bidirectional ensures we don't miss any channels due to geometry.
        
        # Recalculate micro step sizes for macro loop
        dtheta_micro_macro = max_elevation_offset / max(1, n_elev_micro_macro)
        dphi_micro_macro = self.dphi / (n_azi_micro_macro + 1)
        
        consecutive_empty_macrosteps = 0
        
        for macro_step in range(1, max_coarse_steps + 1):
            # Try both directions each iteration
            for direction_sign in [+1, -1]:
                dtheta_macro = direction_sign * macro_step * macro_step_size
                
                # Track if this macro step added anything
                macro_added = 0
                
                # At this macro position, do microstepping
                for j in range(n_elev_micro_macro + 1):
                    for i in range(-n_azi_micro_macro, n_azi_micro_macro + 1):
                        dtheta_total = dtheta_macro + j * dtheta_micro_macro
                        dphi = i * dphi_micro_macro
                        
                        added = _process_frame(dphi, dtheta_total)
                        if added > 0:
                            offsets.append((dphi, dtheta_total))
                            macro_added += added
                        
                        # Check after every frame
                        if _all_bins_satisfied():
                            return self._build_return(offsets, samples, counts, all_channels)
                
                # Count empty steps
                if macro_added == 0:
                    consecutive_empty_macrosteps += 1
                    # Stop early if we've had too many consecutive empty steps
                    if consecutive_empty_macrosteps >= max_empty_macrosteps * 2:  # *2 since doing 2 directions
                        print(f"DEBUG: Stopping at macro_step {macro_step} - no progress for {consecutive_empty_macrosteps} steps")
                        import sys; sys.stdout.flush()
                        return self._build_return(offsets, samples, counts, all_channels)
                else:
                    consecutive_empty_macrosteps = 0  # Reset counter when we make progress
        
        # Reached max iterations without fully satisfying all bins
        return self._build_return(offsets, samples, counts, all_channels)
    
    def _build_return(self, offsets, samples, counts, all_channels):
        """Build return structure with summary statistics."""
        # Determine which channels actually got samples
        reachable_channels = [ch for ch in all_channels 
                             if sum(counts[ch].values()) > 0]
        
        total_samples = sum(sum(counts[ch].values()) for ch in all_channels)
        bins_satisfied = sum(1 for ch in all_channels
                           for e in range(self.elevations_per_channel) 
                           if counts[ch][e] >= 19)
        total_bins = len(all_channels) * self.elevations_per_channel
        
        # Convert relative offsets to absolute gimbal commands if calibrated
        if self.is_calibrated:
            offsets_absolute = [self.relative_to_absolute_offset(dphi, dtheta) 
                               for dphi, dtheta in offsets]
        else:
            offsets_absolute = offsets  # Return as-is if not calibrated
        
        summary = {
            "total_samples": total_samples,
            "total_offsets": len(offsets),
            "bins_satisfied": bins_satisfied,
            "total_bins": total_bins,
            "reachable_channels": reachable_channels,
            "completion_rate": bins_satisfied / max(1, total_bins),
            "is_calibrated": self.is_calibrated,
            "gimbal_calib_offset": (self.gimbal_dphi_calib, self.gimbal_dtheta_calib) if self.is_calibrated else None,
        }
        
        return offsets, offsets_absolute, samples, counts, reachable_channels, summary

    # ------------------------------------------------------------
    # CSV export methods
    # ------------------------------------------------------------
    def save_csv_offsets(self, filename, offsets):
        """Save offsets to CSV file."""
        import csv
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['dphi_rad', 'dtheta_rad'])
            for dphi, dtheta in offsets:
                writer.writerow([dphi, dtheta])

    def save_offsets_with_calibration(self, filename, offsets_relative, offsets_absolute=None):
        """
        Save both relative and absolute gimbal offsets with calibration info.
        
        Parameters
        ----------
        filename : str
            Output CSV filename.
        offsets_relative : list[(dphi, dtheta)]
            Relative offsets from autofill.
        offsets_absolute : list[(dphi, dtheta)], optional
            Absolute gimbal commands. If None, will be computed.
        """
        import csv
        
        if offsets_absolute is None:
            offsets_absolute = [self.relative_to_absolute_offset(dphi, dtheta) 
                               for dphi, dtheta in offsets_relative]
        
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # Header with calibration info
            writer.writerow(['Calibration Info'])
            writer.writerow(['gimbal_dphi_calib_rad', 'gimbal_dtheta_calib_rad'])
            writer.writerow([self.gimbal_dphi_calib, self.gimbal_dtheta_calib])
            writer.writerow([])
            
            # Offset data
            writer.writerow(['Index', 'dphi_rel_rad', 'dtheta_rel_rad', 'dphi_abs_rad', 'dtheta_abs_rad'])
            for idx, (dphi_rel, dtheta_rel) in enumerate(offsets_relative):
                dphi_abs, dtheta_abs = offsets_absolute[idx]
                writer.writerow([idx, dphi_rel, dtheta_rel, dphi_abs, dtheta_abs])