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


    # ------------------------------------------------------------
    # Overlap checking helpers
    # ------------------------------------------------------------
    def _min_pairwise_dist2(self, xs, ys):
        """Return minimum squared pairwise distance between points in xs, ys."""
        xs = np.asarray(xs).ravel()
        ys = np.asarray(ys).ravel()
        n = xs.size
        if n < 2:
            return float('inf')
        min_sq = float('inf')
        for i in range(n - 1):
            dx = xs[i+1:] - xs[i]
            dy = ys[i+1:] - ys[i]
            if dx.size > 0:
                d2 = dx * dx + dy * dy
                m = float(np.min(d2))
                if m < min_sq:
                    min_sq = m
        return min_sq

    def validate_no_overlap(self, xs, ys, spot_radius_m, tol=1e-12):
        """Return (ok, min_sq, required_sq). ok True if min_sq + tol >= (2*spot_radius_m)**2."""
        min_sq = self._min_pairwise_dist2(xs, ys)
        required = (2.0 * float(spot_radius_m)) ** 2
        ok = (min_sq + tol) >= required
        return bool(ok), float(min_sq), float(required)

    # ------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------

    def plot_single_frame(self, ax=None, margin_m=0.0):
        X, Y = self.project_to_target()
        mask = self.inside_mask(X, Y, margin_m=margin_m)

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))

        # outer target outline
        ax.add_patch(
            Rectangle(
                (-self.W / 2, -self.H / 2),
                self.W,
                self.H,
                fill=False,
                linewidth=2,
            )
        )

        # optional inner valid area when margin is used
        if margin_m > 0:
            inner_w = max(0.0, self.W - 2 * margin_m)
            inner_h = max(0.0, self.H - 2 * margin_m)
            ax.add_patch(
                Rectangle(
                    (-self.W / 2 + margin_m, -self.H / 2 + margin_m),
                    inner_w,
                    inner_h,
                    fill=False,
                    linewidth=1,
                    linestyle='--',
                    edgecolor='C1',
                )
            )

        ax.scatter(X[mask], Y[mask], s=10)
        ax.set_aspect("equal")
        ax.set_title(f"Single Frame Hits ({np.sum(mask)} pts)")
        ax.grid(True)
        return ax

    def plot_autofilled(self, min_samples=300, margin_m=0.0, spot_radius_m=0.0):
        offsets, (x0, y0), (xf, yf), dphi_sub, dtheta_sub, j, i = self.autofill_to_min_samples(min_samples=min_samples, margin_m=margin_m, spot_radius_m=spot_radius_m)

        # Full base grid (for visual comparison)
        X0, Y0 = self.project_to_target()
        mask0 = self.inside_mask(X0, Y0, margin_m=margin_m)
        x_full = X0[mask0].ravel()
        y_full = Y0[mask0].ravel()

        fig, ax = plt.subplots(figsize=(6, 6))

        # Target outline
        ax.add_patch(
            Rectangle(
                (-self.W / 2, -self.H / 2),
                self.W,
                self.H,
                fill=False,
                linewidth=2,
                edgecolor="k",
            )
        )

        # optional inner valid area when margin is used
        if margin_m > 0:
            inner_w = max(0.0, self.W - 2 * margin_m)
            inner_h = max(0.0, self.H - 2 * margin_m)
            ax.add_patch(
                Rectangle(
                    (-self.W / 2 + margin_m, -self.H / 2 + margin_m),
                    inner_w,
                    inner_h,
                    fill=False,
                    linewidth=1,
                    linestyle='--',
                    edgecolor='C1',
                )
            )

        # Plot full frame faintly so users can compare with accepted base
        if len(x_full) > 0:
            ax.scatter(x_full, y_full, s=6, c='0.7', alpha=0.4, label=f"Full frame ({len(x_full)})", zorder=1)

        # Accepted base-frame points (after enforcing non-overlap)
        ax.scatter(
            x0,
            y0,
            s=12,
            c="tab:blue",
            label=f"Accepted base ({len(x0)})",
            zorder=3,
        )

        # Autofilled points
        if len(xf) > 0:
            ax.scatter(
                xf,
                yf,
                s=8,
                c="tab:red",
                label=f"Gimbal fill ({len(xf)})",
                zorder=2,
            )

        ax.set_aspect("equal")
        ax.set_xlabel("X on target (m)")
        ax.set_ylabel("Y on target (m)")
        title = (
            f"FMCW Autofill Sampling\n"
            f"Full: {len(x_full)} pts, Accepted base: {len(x0)}, Added: {len(xf)}, Offsets: {len(offsets)}"
        )
        if len(offsets) <= 1 and len(xf) == 0:
            title += "\nNo gimbal fill required (base meets min_samples)"

        ax.set_title(title)
        ax.grid(True)
        ax.legend()
        plt.show()

        return offsets, (x0, y0), (xf, yf), dphi_sub, dtheta_sub, j, i

    # def autofill_per_elevation(self,
    #                             per_row=300,
    #                             spot_radius_m=0.0,
    #                             margin_m=0.0,
    #                             max_subdiv=32):
    #     """
    #     Autofill samples per elevation row until each row has at least `per_row` samples.

    #     Simple implementation (sweep, azimuth-only micro-steps):
    #     - For each elevation row, accepts base hits then applies symmetric azimuth offsets
    #       (positive and negative) until per_row unique, non-overlapping points are collected
    #       or subdivisions are exhausted.
    #     - `spot_radius_m` defines the optical spot radius; any candidate point closer than
    #       `2 * spot_radius_m` to an accepted point for the same row will be rejected.

    #     Returns
    #     -------
    #     rows : list[dict]
    #         Each dict has keys: 'theta' (rad), 'base' (n), 'added' (n), 'total' (n),
    #         'offsets' (list of dphi offsets used), 'x0', 'y0', 'xf', 'yf' (numpy arrays)
    #     """
    #     # Project base grid
    #     X0, Y0 = self.project_to_target()
    #     n_rows, n_cols = X0.shape

    #     results = []

    #     for r in range(n_rows):
    #         X_row = X0[r, :]
    #         Y_row = Y0[r, :]
    #         # mask for points inside the (inner) rectangle
    #         mask_base = self.inside_mask(X_row, Y_row, margin_m=margin_m)
    #         xs_base = X_row[mask_base].ravel().tolist()
    #         ys_base = Y_row[mask_base].ravel().tolist()

    #         # Build accepted list enforcing minimal spacing
    #         accepted_x = []
    #         accepted_y = []

    #         def accept_point(xp, yp):
    #             if spot_radius_m <= 0.0:
    #                 return True
    #             min_sep2 = (2.0 * spot_radius_m) ** 2
    #             for xa, ya in zip(accepted_x, accepted_y):
    #                 dx = xa - xp
    #                 dy = ya - yp
    #                 if dx * dx + dy * dy < min_sep2:
    #                     return False
    #             return True

    #         # accept base points in order
    #         for xb, yb in zip(xs_base, ys_base):
    #             if accept_point(xb, yb):
    #                 accepted_x.append(xb); accepted_y.append(yb)
    #             if len(accepted_x) >= per_row:
    #                 break

    #         offsets_used = []

    #         if len(accepted_x) < per_row:
    #             # need to micro-step in azimuth only
    #             base_count = max(1, len(accepted_x))
    #             frames_needed = int(np.ceil(per_row / base_count))
    #             n_side = int(np.ceil(np.sqrt(frames_needed)))
    #             n_side = min(n_side, max_subdiv)
    #             dphi_sub = self.dphi / (n_side + 1)

    #             # symmetric offset order: 1, -1, 2, -2, ... up to n_side
    #             seq = []
    #             for k in range(1, n_side + 1):
    #                 seq.append(k)
    #                 seq.append(-k)

    #             for i_off in seq:
    #                 dphi = i_off * dphi_sub
    #                 # project with dphi offset and extract same row
    #                 X_off, Y_off = self.project_to_target(dphi_offset=dphi)
    #                 Xr = X_off[r, :]
    #                 Yr = Y_off[r, :]
    #                 mask = self.inside_mask(Xr, Yr, margin_m=margin_m)
    #                 xs = Xr[mask].ravel().tolist()
    #                 ys = Yr[mask].ravel().tolist()

    #                 added_this_offset = 0
    #                 for xc, yc in zip(xs, ys):
    #                     if accept_point(xc, yc):
    #                         accepted_x.append(xc); accepted_y.append(yc); added_this_offset += 1
    #                         if len(accepted_x) >= per_row:
    #                             break
    #                 if added_this_offset > 0:
    #                     offsets_used.append((dphi, added_this_offset))
    #                 if len(accepted_x) >= per_row:
    #                     break

    #         results.append({
    #             'theta': float(self.elevation_angles()[r]),
    #             'base': len(xs_base),
    #             'added': max(0, len(accepted_x) - len(xs_base)),
    #             'total': len(accepted_x),
    #             'offsets': offsets_used,
    #             'x0': np.array(xs_base),
    #             'y0': np.array(ys_base),
    #             'xf': np.array(accepted_x[len(xs_base):]),
    #             'yf': np.array(accepted_y[len(xs_base):]),
    #         })

    #     return results


if __name__ == '__main__':
    import pandas as pd
    
    # do HFOV as both 120 deg and 45 deg
    HFOV = 45
    VFOV = 30
    
    TARGET_WIDTH = 1.2
    TARGET_HEIGHT = 1.2
    SPOT_DIAMETER_M= 0.0135
    SPOT_RADIUS = SPOT_DIAMETER_M / 2  # What is this actual value?

    
    TARGET_DISTANCE_LIST = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # TARGET_DISTANCE_LIST = [30]
    # splits up the HFOV and VFOV into discrete angular steps
    AZ_RES_DEG = HFOV/181
    EL_RES_DEG = VFOV/128
    
    results = []
    
    for TARGET_DISTANCE in TARGET_DISTANCE_LIST:
        sampler = FlatTargetHitCounter(target_width_m=TARGET_WIDTH,
                                       target_height_m=TARGET_HEIGHT,
                                       distance_m=TARGET_DISTANCE,
                                       azimuth_res_rad=np.deg2rad(AZ_RES_DEG),
                                       elevation_res_rad=np.deg2rad(EL_RES_DEG))
    
        single_frame_points = sampler.total_points_single_frame()
        num_of_elevations = len(sampler.azimuth_points_per_elevation())
        print("Single-frame points:", single_frame_points)
        print("Number of Elevations:", num_of_elevations)
        print("Points per Elevation:", int(single_frame_points/num_of_elevations))
        
        sampler.plot_single_frame(margin_m=0.02)

        offsets, p0, p_ext, dphi_sub, dtheta_sub, j, i = sampler.plot_autofilled(min_samples=300, margin_m=0.02, spot_radius_m=SPOT_RADIUS)
        num_of_gimbal_steps = len(offsets)
        num_of_points_with_gimbal_stepping = len(p0[0]) + len(p_ext[0])
        print('Number of Gimbal Steps (to reach 300 points)', num_of_gimbal_steps)
        print('Number of points post gimbal stepping', num_of_points_with_gimbal_stepping)

        # Optional: validate no overlap for all accepted points (disabled by default)
        run_overlap_check = True
        if run_overlap_check:
            all_x = np.concatenate([p0[0], p_ext[0]])
            all_y = np.concatenate([p0[1], p_ext[1]])
            ok, min_sq, required_sq = sampler.validate_no_overlap(all_x, all_y, SPOT_RADIUS)
            if not ok:
                print(f"ERROR: Overlap detected! min_dist={np.sqrt(min_sq):.6f} m, required >= {np.sqrt(required_sq):.6f} m")
                raise AssertionError("Overlap detected in sampling for spot_radius_m={}".format(SPOT_RADIUS))
            else:
                print(f"No overlap: min_dist={np.sqrt(min_sq):.6f} m, required >= {np.sqrt(required_sq):.6f} m")
        
        num_of_az_steps = len(set([item[0] for item in offsets]))
    
        results.append((TARGET_WIDTH, TARGET_HEIGHT, HFOV, VFOV, 
                        TARGET_DISTANCE, single_frame_points, num_of_elevations,
                        num_of_gimbal_steps, num_of_points_with_gimbal_stepping,
                        np.rad2deg(dphi_sub), np.rad2deg(dtheta_sub), num_of_az_steps))
        
        
    # save to csv
    results_dict = {}
    results_dict['Target Width [m]'] = [item[0] for item in results]
    results_dict['Target Height [m]'] = [item[1] for item in results]
    results_dict['HFOV [deg]'] = [item[2] for item in results]
    results_dict['VFOV [deg]'] = [item[3] for item in results]
    results_dict['Target Distance [m]'] = [item[4] for item in results]
    results_dict['Number of Hits (Single)'] = [item[5] for item in results]
    results_dict['Number of Elevations'] = [item[6] for item in results]
    results_dict['Number of Gimbal Steps (300 Hits)'] = [item[7] for item in results]
    results_dict['Number of Hits (Gimbal)'] = [item[8] for item in results]
    results_dict['Azimuth Stepping [deg]'] = [item[9] for item in results]
    results_dict['Elevation Stepping [deg]'] = [item[10] for item in results]
    results_dict['Number of Azimuth Stepping [deg]'] = [item[11] for item in results]
    
    df_results = pd.DataFrame(data=results_dict, index=None)
    df_results.to_csv('results.csv', index=False)

    
    

