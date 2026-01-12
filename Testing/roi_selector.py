# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 07:45:33 2025

@author: LawrenceTzuang
"""

import numpy as np
import matplotlib.pyplot as plt

class PointCloudROISelector:
    """
    Interactive 3-view (XY, XZ, YZ) ROI selector for point clouds
    using Matplotlib.

    - Right-click to select points
    - Max 4 points per view (FIFO)
    - Grid enabled
    """

    MAX_CLICKS_PER_VIEW = 4

    def __init__(self):
        self.points = None
        self.x = self.y = self.z = None

        self.clicks = {"XY": [], "XZ": [], "YZ": []}
        self.click_artists = {"XY": [], "XZ": [], "YZ": []}

        self.bounds = {
            "x": {"min": [], "max": []},
            "y": {"min": [], "max": []},
            "z": {"min": [], "max": []},
        }

        self.roi_points = None
        self.roi_bounds = None

        self.fig = None
        self.axs = None

        self.VIEW_INFO = {
            "XY": {"ax": 0, "color": "#39FF14"},  # neon green
            "XZ": {"ax": 1, "color": "#FF00FF"},  # neon magenta
            "YZ": {"ax": 2, "color": "#FFFF33"},  # neon yellow
        }

    # -------------------------------------------------
    # Data loading
    # -------------------------------------------------
    def load_points(self, df, frame_sparse=1):
        # Validate
        required = {"x", "y", "z", "frame_idx"}
        if not required.issubset(df.columns):
            raise ValueError("DataFrame must contain columns: x, y, z, frame_idx")
    
        # frame sparse
        if frame_sparse >= 1:
            df = df[(df['frame_idx'] % frame_sparse) == 0]
        
        # Authoritative point cloud
        self.points = df[["x", "y", "z"]].to_numpy()
    
        # Convenience views (no copy)
        self.x, self.y, self.z = self.points.T
    
        # Reset interaction state ONLY
        self._reset_interaction_state()
    
        # Build figure
        self._setup_figure()

    # -------------------------------------------------
    # Figure / UI setup
    # -------------------------------------------------
    def _setup_figure(self):
        self.fig, self.axs = plt.subplots(1, 3, figsize=(15, 5))

        self.axs[0].scatter(self.x, self.y, s=1)
        self.axs[0].set_title("XY view")
        self.axs[0].set_xlabel("X")
        self.axs[0].set_ylabel("Y")

        self.axs[1].scatter(self.x, self.z, s=1)
        self.axs[1].set_title("XZ view")
        self.axs[1].set_xlabel("X")
        self.axs[1].set_ylabel("Z")

        self.axs[2].scatter(self.y, self.z, s=1)
        self.axs[2].set_title("YZ view")
        self.axs[2].set_xlabel("Y")
        self.axs[2].set_ylabel("Z")

        for ax in self.axs:
            ax.set_aspect("equal")
            ax.grid(True, linestyle="--", alpha=0.5)

        self.fig.suptitle(
            "RIGHT click to select (max 4/view) | ENTER = finalize | r = reset",
            fontsize=12
        )

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    # -------------------------------------------------
    # Drawing helpers
    # -------------------------------------------------
    def _draw_cross(self, ax, x0, y0, color):
        artist, = ax.plot(
            x0, y0,
            marker="+",
            markersize=10,
            markeredgewidth=2,
            color=color
        )
        return artist

    # -------------------------------------------------
    # Event handlers
    # -------------------------------------------------
    def _on_click(self, event):
        # Only RIGHT mouse button (avoid zoom/pan conflicts)
        if event.button != 3 or event.inaxes is None:
            return

        if event.inaxes == self.axs[0]:
            view = "XY"
            x0, y0 = event.xdata, event.ydata
        elif event.inaxes == self.axs[1]:
            view = "XZ"
            x0, y0 = event.xdata, event.ydata
        elif event.inaxes == self.axs[2]:
            view = "YZ"
            x0, y0 = event.xdata, event.ydata
        else:
            return

        # Enforce max clicks (FIFO)
        if len(self.clicks[view]) >= self.MAX_CLICKS_PER_VIEW:
            self.clicks[view].pop(0)
            old_artist = self.click_artists[view].pop(0)
            old_artist.remove()

        self.clicks[view].append((x0, y0))
        artist = self._draw_cross(
            event.inaxes,
            x0, y0,
            self.VIEW_INFO[view]["color"]
        )
        self.click_artists[view].append(artist)

        event.canvas.draw_idle()

    def _on_key(self, event):
        if event.key == "enter":
            self._finalize_roi()
            plt.close(self.fig)
        elif event.key == "r":
            self.reset()

    # -------------------------------------------------
    # ROI logic
    # -------------------------------------------------
    def _finalize_roi(self):
        self.roi_bounds = self._compute_bounds()

        xmin, xmax = self.roi_bounds["x"]
        ymin, ymax = self.roi_bounds["y"]
        zmin, zmax = self.roi_bounds["z"]

        mask = (
            (self.x >= xmin) & (self.x <= xmax) &
            (self.y >= ymin) & (self.y <= ymax) &
            (self.z >= zmin) & (self.z <= zmax)
        )

        self.roi_points = self.points[mask]

        print("\nFinal ROI bounds:")
        for k, v in self.roi_bounds.items():
            print(f"  {k}: {v}")
        print(f"ROI contains {self.roi_points.shape[0]} points")

    def _compute_bounds(self):
        # XY → X, Y
        if self.clicks["XY"]:
            xs, ys = zip(*self.clicks["XY"])
            self.bounds["x"]["min"].append(min(xs))
            self.bounds["x"]["max"].append(max(xs))
            self.bounds["y"]["min"].append(min(ys))
            self.bounds["y"]["max"].append(max(ys))

        # XZ → X, Z
        if self.clicks["XZ"]:
            xs, zs = zip(*self.clicks["XZ"])
            self.bounds["x"]["min"].append(min(xs))
            self.bounds["x"]["max"].append(max(xs))
            self.bounds["z"]["min"].append(min(zs))
            self.bounds["z"]["max"].append(max(zs))

        # YZ → Y, Z
        if self.clicks["YZ"]:
            ys, zs = zip(*self.clicks["YZ"])
            self.bounds["y"]["min"].append(min(ys))
            self.bounds["y"]["max"].append(max(ys))
            self.bounds["z"]["min"].append(min(zs))
            self.bounds["z"]["max"].append(max(zs))

        return {
            axis: (min(v["min"]), max(v["max"]))
            for axis, v in self.bounds.items()
            if v["min"]
        }
    
    def _reset_interaction_state(self):
        """Reset ROI interaction state but KEEP loaded point data."""
        self.clicks = {"XY": [], "XZ": [], "YZ": []}
        self.click_artists = {"XY": [], "XZ": [], "YZ": []}
    
        self.bounds = {
            "x": {"min": [], "max": []},
            "y": {"min": [], "max": []},
            "z": {"min": [], "max": []},
        }
    
        self.roi_points = None
        self.roi_bounds = None

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------
    def show(self):
        if self.fig is None:
            raise RuntimeError("No data loaded. Call load_points() first.")
        plt.show()

    def reset(self, clear_figure=False):
        self.clicks = {"XY": [], "XZ": [], "YZ": []}
        self.click_artists = {"XY": [], "XZ": [], "YZ": []}
        self.bounds = {
            "x": {"min": [], "max": []},
            "y": {"min": [], "max": []},
            "z": {"min": [], "max": []},
        }
        self.roi_points = None
        self.roi_bounds = None

        if clear_figure and self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.axs = None
            
    def _draw_roi_bbox_3d(self, ax):
        xmin, xmax = self.roi_bounds["x"]
        ymin, ymax = self.roi_bounds["y"]
        zmin, zmax = self.roi_bounds["z"]
    
        corners = np.array([
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmax, ymax, zmin],
            [xmin, ymax, zmin],
            [xmin, ymin, zmax],
            [xmax, ymin, zmax],
            [xmax, ymax, zmax],
            [xmin, ymax, zmax],
        ])
    
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
    
        for i, j in edges:
            ax.plot(
                [corners[i, 0], corners[j, 0]],
                [corners[i, 1], corners[j, 1]],
                [corners[i, 2], corners[j, 2]],
                color="r",
                linewidth=1.5
            )

            
    def show_roi_3d(self, show_bbox=True, point_size=2):
        """
        Visualize the finalized ROI points in 3D.
    
        Parameters
        ----------
        show_bbox : bool
            Whether to draw the ROI bounding box
        point_size : int
            Scatter point size
        """
        if self.roi_points is None or self.roi_bounds is None:
            raise RuntimeError(
                "ROI not finalized. Press ENTER in the selector before calling show_roi_3d()."
            )
    
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    
        pts = self.roi_points
        x, y, z = pts.T
    
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
    
        ax.scatter(x, y, z, s=point_size, alpha=0.8)
    
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("ROI – 3D View")
    
        ax.set_box_aspect([1, 1, 1])
        ax.grid(True)
    
        if show_bbox:
            self._draw_roi_bbox_3d(ax)
    
        plt.show()


if __name__ == '__main__':
    import pandas as pd
    
    # Create selector
    selector = PointCloudROISelector()
    
    # Load data from wherever you want
    filepath = r"C:\Users\LawrenceTzuang\Desktop\pc_validation\CES3\pointCloud_20251218_153128_CES3_5m_middle_beams_results_normal.csv"
    df = pd.read_csv(filepath)
    selector.load_points(df, frame_sparse=10)
    
    # Interact
    selector.show()
    
    # # Results
    roi_bounds = selector.roi_bounds
    roi_pts = selector.roi_points
    
    # # visualize selection    
    selector.show()
    # roi_pts = selector.roi_points
    selector.show_roi_3d() # visualize cropped result
