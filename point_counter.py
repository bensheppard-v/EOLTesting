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

    def _angular_limits(self):
        phi_half = np.arctan((self.W / 2) / self.D)
        theta_half = np.arctan((self.H / 2) / self.D)
        return phi_half, theta_half

    def azimuth_angles(self, dphi_offset=0.0):
        phi_half, _ = self._angular_limits()
        n = int(np.ceil(phi_half / self.dphi))
        return np.arange(-n, n + 1) * self.dphi + dphi_offset

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

    def inside_mask(self, X, Y):
        return (np.abs(X) <= self.W / 2) & (np.abs(Y) <= self.H / 2)

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
                                max_subdiv=8,):
        """
        Autofill samples using gimbal micro-stepping.
    
        Returns
        -------
        offsets : list[(dphi, dtheta)]
        base_points : (x0, y0)
        filled_points : (xf, yf)
        """
    
        # ---- Base frame ----
        X0, Y0 = self.project_to_target()
        mask0 = self.inside_mask(X0, Y0)
    
        x0 = X0[mask0].ravel()
        y0 = Y0[mask0].ravel()
    
        if len(x0) >= min_samples:
            return [(0.0, 0.0)], (x0, y0), (np.array([]), np.array([])), 0, 0, 0, 0
    
        # ---- Autofill ----
        xf, yf = [], []
        offsets = [(0.0, 0.0)]
    
        n0 = len(x0)
        frames_needed = int(np.ceil(min_samples / n0))
        n_side = int(np.ceil(np.sqrt(frames_needed)))
        n_side = min(n_side, max_subdiv)
    
        dphi_sub = self.dphi / (n_side + 1)
        dtheta_sub = self.dtheta / (n_side + 1)
    
        for j in range(n_side + 1):
            for i in range(n_side + 1):
                if i == 0 and j == 0:
                    continue
    
                dphi = i * dphi_sub
                dtheta = j * dtheta_sub
    
                X, Y = self.project_to_target(dphi, dtheta)
                mask = self.inside_mask(X, Y)
    
                xf.extend(X[mask].ravel())
                yf.extend(Y[mask].ravel())
                offsets.append((dphi, dtheta))
    
                if len(x0) + len(xf) >= min_samples:
                    return offsets, (x0, y0), (np.array(xf), np.array(yf)), dphi_sub, dtheta_sub, j, i
    
        return offsets, (x0, y0), (np.array(xf), np.array(yf)), dphi_sub, dtheta_sub, j, i


    # ------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------

    def plot_single_frame(self, ax=None):
        X, Y = self.project_to_target()
        mask = self.inside_mask(X, Y)

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))

        ax.add_patch(
            Rectangle(
                (-self.W / 2, -self.H / 2),
                self.W,
                self.H,
                fill=False,
                linewidth=2,
            )
        )

        ax.scatter(X[mask], Y[mask], s=10)
        ax.set_aspect("equal")
        ax.set_title(f"Single Frame Hits ({np.sum(mask)} pts)")
        ax.grid(True)
        return ax

    def plot_autofilled(self, min_samples=300):
        offsets, (x0, y0), (xf, yf), dphi_sub, dtheta_sub, j, i = self.autofill_to_min_samples(min_samples)
    
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
    
        # Base-frame points
        ax.scatter(
            x0,
            y0,
            s=12,
            c="tab:blue",
            label=f"Base frame ({len(x0)})",
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
        ax.set_title(
            f"FMCW Autofill Sampling\n"
            f"Total: {len(x0) + len(xf)} pts, {len(offsets)} gimbal steps"
        )
        ax.grid(True)
        ax.legend()
        plt.show()

        return offsets, (x0, y0), (xf, yf), dphi_sub, dtheta_sub, j, i
    

if __name__ == '__main__':
    import pandas as pd
    
    HFOV = 120
    VFOV = 30
    
    TARGET_WIDTH = 1.2
    TARGET_HEIGHT = 1.2
    
    TARGET_DISTANCE_LIST = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # TARGET_DISTANCE_LIST = [30]
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
        
        sampler.plot_single_frame()
    
        offsets, p0, p_ext, dphi_sub, dtheta_sub, j, i = sampler.plot_autofilled(min_samples=300)
        num_of_gimbal_steps = len(offsets)
        num_of_points_with_gimbal_stepping = len(p0[0]) + len(p_ext[0])
        print('Number of Gimbal Steps (to reach 300 points)', num_of_gimbal_steps)
        print('Number of points post gimbal stepping', num_of_points_with_gimbal_stepping)
        
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

    
    

