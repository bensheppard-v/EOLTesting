import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class FlatTargetHitCounter:
    """
    FMCW LiDAR sampling on a flat rectangular target with gimbal autofill.

    - Angular sampling (real LiDAR beams)
    - Counts points inside rectangle
    - Gimbal micro-stepping to reach target sample count
    - No overlapping beams per frame
    """

    def __init__(
        self,
        target_width_m,
        target_height_m,
        distance_m,
        azimuth_res_rad,
        elevation_res_rad,
        elevation_center_rad=0.0,
    ):
        self.W = float(target_width_m)
        self.H = float(target_height_m)
        self.D = float(distance_m)
        self.dphi = float(azimuth_res_rad)
        self.dtheta = float(elevation_res_rad)
        self.theta0 = float(elevation_center_rad)

    # ------------------------------------------------------------
    # Angular grid
    # ------------------------------------------------------------

    # phi is azimuth, theta is elevation
    def _angular_limits(self):
        phi_half = np.arctan((self.W / 2) / self.D)
        theta_half = np.arctan((self.H / 2) / self.D)
        return phi_half, theta_half

    def azimuth_angles(self, dphi_offset=0.0):
        phi_half, _ = self._angular_limits()
        n = int(np.ceil(phi_half / self.dphi)) # num steps from center to one edge
        return np.arange(-n, n + 1) * self.dphi + dphi_offset # creates angle array

    def elevation_angles(self, dtheta_offset=0.0):
        _, theta_half = self._angular_limits()
        n = int(np.ceil(theta_half / self.dtheta))
        return self.theta0 + np.arange(-n, n + 1) * self.dtheta + dtheta_offset

    # ------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------

    def project_to_target(self, dphi_offset=0.0, dtheta_offset=0.0):
        phis = self.azimuth_angles(dphi_offset)
        thetas = self.elevation_angles(dtheta_offset)

        PHI, THETA = np.meshgrid(phis, thetas)

        X = self.D * np.tan(PHI)
        Y = self.D * np.tan(THETA - self.theta0)

        return X, Y

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

    def autofill_to_min_samples(self,
                                min_samples=300,
                                max_subdiv=32,
                                margin_m=0.0,
                                spot_radius_m=0.0):
        """
        Autofill samples using gimbal micro-stepping.

        Parameters
        - min_samples: target unique samples for the whole target
        - max_subdiv: cap on micro-step subdivisions
        - margin_m: border margin to exclude points near the edge (metres)
        - spot_radius_m: radius of a laser spot (metres). Points closer than 2*spot_radius_m
          are considered overlapping and will be rejected to ensure non-overlap.

        Returns
        -------
        offsets : list[(dphi, dtheta)]
        base_points : (x0, y0)
        filled_points : (xf, yf)
        """

        # ---- Base frame ----
        X0, Y0 = self.project_to_target()
        mask0 = self.inside_mask(X0, Y0, margin_m=margin_m)

        x0_raw = X0[mask0].ravel()
        y0_raw = Y0[mask0].ravel()

        # enforce non-overlap on base points
        accepted_x = []
        accepted_y = []
        base_x = []
        base_y = []

        min_sep2 = (2.0 * float(spot_radius_m)) ** 2 if spot_radius_m > 0.0 else 0.0

        def _accept(xp, yp):
            if min_sep2 <= 0.0:
                return True
            for xa, ya in zip(accepted_x, accepted_y):
                dx = xa - xp
                dy = ya - yp
                if dx * dx + dy * dy < min_sep2:
                    return False
            return True

        for xv, yv in zip(x0_raw, y0_raw):
            if _accept(xv, yv):
                accepted_x.append(xv); accepted_y.append(yv)
                base_x.append(xv); base_y.append(yv)
            if len(accepted_x) >= min_samples:
                return [(0.0, 0.0)], (np.array(base_x), np.array(base_y)), (np.array([]), np.array([])), 0, 0, 0, 0

        # ---- Autofill ----
        xf, yf = [], []  # store added points only
        offsets = [(0.0, 0.0)]

        n0 = len(base_x)  # number of accepted base samples
        frames_needed = int(np.ceil(min_samples / max(1, n0)))  # num frames of base needed
        n_side = int(np.ceil(np.sqrt(frames_needed)))  # near-square grid
        n_side = min(n_side, max_subdiv)

        dphi_sub = self.dphi / (n_side + 1)
        dtheta_sub = self.dtheta / (n_side + 1)

        last_j, last_i = -1, -1

        for j in range(n_side + 1):
            for i in range(n_side + 1):
                if i == 0 and j == 0:
                    continue

                dphi = i * dphi_sub
                dtheta = j * dtheta_sub

                X, Y = self.project_to_target(dphi, dtheta) # applies offset
                mask = self.inside_mask(X, Y, margin_m=margin_m)

                xs = X[mask].ravel()
                ys = Y[mask].ravel()
                added = 0
                for xv, yv in zip(xs, ys):
                    if _accept(xv, yv):
                        # accept non-overlapping point
                        accepted_x.append(xv); accepted_y.append(yv)
                        xf.append(xv); yf.append(yv)
                        added += 1
                        if len(accepted_x) >= min_samples:
                            offsets.append((dphi, dtheta))
                            return offsets, (np.array(base_x), np.array(base_y)), (np.array(xf), np.array(yf)), dphi_sub, dtheta_sub, j, i

                if added > 0:
                    offsets.append((dphi, dtheta))
                    last_j, last_i = j, i

        return offsets, (np.array(base_x), np.array(base_y)), (np.array(xf), np.array(yf)), dphi_sub, dtheta_sub, last_j, last_i

    def save_csv_offsets(self, filename, offsets):
        """Save offsets to CSV file."""
        import csv
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['dphi_rad', 'dtheta_rad'])
            for dphi, dtheta in offsets:
                writer.writerow([dphi, dtheta])
    

