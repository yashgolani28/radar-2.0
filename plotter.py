# plotter.py
import matplotlib
matplotlib.use("Agg")  # headless rendering

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D proj)
import numpy as np
import threading
import time
import os
import tempfile


class Live3DPlotter:
    """
    Background renderer for dashboard assets:
      - static/scatter_3d.png
      - static/heatmap_3d.png  (this class is the ONLY writer)
      - static/heatmap_xy.png / heatmap_xz.png / heatmap_yz.png (voxel projections)
    """
    def __init__(self):
        self.objects = []          # [{x,y,z,vx,vy,vz,ax,ay,az,type,speed}]
        self.heatmap_data = None   # last valid 2D uint8 heatmap (for fallback)
        self.lock = threading.Lock()
        self.running = True

        # Save frequency for scatter (seconds)
        self._last_scatter_save = 0.0
        self._scatter_save_every = 0.6

        # Throttle voxel generation
        self._last_voxel_ts = 0.0
        self._voxel_every = 1.0

        # Shared static dir (same for radar + web). Override with ISK_STATIC_DIR.
        self.static_dir = os.environ.get("ISK_STATIC_DIR", os.path.join(os.path.dirname(__file__), "static"))
        os.makedirs(self.static_dir, exist_ok=True)

        self.fig = plt.figure(figsize=(8, 7))
        self.ax = self.fig.add_subplot(111, projection="3d")

        self.thread = threading.Thread(target=self._render_loop, daemon=True)
        self.thread.start()

    # ---------- public API ----------

    def update(self, tracked_objects):
        """
        Update the live 3D scatter state (objects list).
        Each object may contain: x,y,z, velX/velY/velZ, accX/accY/accZ, type, speed[_kmh]
        """
        with self.lock:
            objs = []
            for o in tracked_objects or []:
                if all(k in o for k in ("x", "y", "z")):
                    objs.append({
                        "x": float(o.get("x", 0.0)),
                        "y": float(o.get("y", 0.0)),
                        "z": float(o.get("z", 0.0)),
                        "vx": float(o.get("velX", 0.0)),
                        "vy": float(o.get("velY", 0.0)),
                        "vz": float(o.get("velZ", 0.0)),
                        "ax": float(o.get("accX", 0.0)),
                        "ay": float(o.get("accY", 0.0)),
                        "az": float(o.get("accZ", 0.0)),
                        "type": str(o.get("type", "UNKNOWN")),
                        "speed": float(o.get("speed_kmh", o.get("speed", 0.0))),
                    })
            self.objects = objs

            now = time.time()
            if now - self._last_voxel_ts > self._voxel_every:
                self._generate_voxel_heatmaps()
                self._last_voxel_ts = now

    # replace your update_heatmap() with this
    def update_heatmap(self, heatmap):
        try:
            # ------ to float array ------
            if heatmap is None:
                return
            import numpy as np, tempfile, os, cv2, time
            if isinstance(heatmap, list):
                arr = np.asarray(heatmap, dtype=np.float32).ravel()
            elif isinstance(heatmap, (bytes, bytearray, memoryview)):
                arr = np.frombuffer(heatmap, dtype=np.int16).astype(np.float32).ravel()
            elif isinstance(heatmap, np.ndarray):
                arr = heatmap.astype(np.float32, copy=False).ravel()
            else:
                return
            if arr.size == 0:
                return
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            arr[arr < 0] = 0.0

            # ------ reshape (prefer square) ------
            n = int(np.sqrt(arr.size))
            target = None
            if n * n == arr.size:
                target = arr.reshape(n, n)
            if target is None:
                for rows in [32, 48, 64, 96, 128, 240, 256]:
                    if arr.size % rows == 0:
                        target = arr.reshape(rows, arr.size // rows)
                        break
            if target is None or min(target.shape) <= 1:
                return

            # ------ robust contrast (avoid all-black) ------
            # use 1–99th percentile stretch on non-zero pixels
            nz = target[target > 0]
            if nz.size >= 10:
                lo, hi = np.percentile(nz, [1, 99])
            else:
                lo, hi = float(target.min()), float(target.max())
            if hi <= lo:
                norm = np.zeros_like(target, dtype=np.uint8)
            else:
                norm = np.clip((target - lo) / (hi - lo), 0, 1)
                norm = (norm * 255.0).astype(np.uint8)

            # ------ atomic save to ISK_STATIC_DIR/heatmap_3d.png ------
            final_path = os.path.join(self.static_dir, "heatmap_3d.png")
            fd, tmp = tempfile.mkstemp(prefix=".hmap_", suffix=".png", dir=self.static_dir)
            os.close(fd)
            try:
                cv2.imwrite(tmp, norm)
                os.replace(tmp, final_path)
                try:
                    sz = os.stat(final_path).st_size
                    print(f"[HEATMAP_SAVE] {final_path} shape={norm.shape} size={sz}B")
                except Exception:
                    pass
            finally:
                try:
                    if os.path.exists(tmp): os.remove(tmp)
                except Exception:
                    pass

            # keep last in memory (optional)
            self.heatmap_data = norm
            self.last_heatmap_timestamp = time.time()

        except Exception as e:
            print(f"[PLOTTER ERROR] heatmap update failed: {e}")

    def stop(self):
        self.running = False
        try:
            self.thread.join(timeout=1.0)
        except Exception:
            pass
        plt.close(self.fig)

    # ---------- internal ----------

    def _render_loop(self):
        while self.running:
            try:
                self._render_once()
            except Exception as e:
                print(f"[PLOTTER ERROR] render loop: {e}")
            time.sleep(0.1)

    def _render_once(self):
        with self.lock:
            self.ax.clear()
            self.ax.set_xlim([-5, 5])
            self.ax.set_ylim([0, 10])
            self.ax.set_zlim([-2, 2])
            self.ax.set_xlabel("X (m)")
            self.ax.set_ylabel("Y (m)")
            self.ax.set_zlabel("Z (m)")
            self.ax.set_title("IWR6843 Real-time 3D Object Cloud")

            for o in self.objects:
                c = self._color(o["type"])
                self.ax.scatter(o["x"], o["y"], o["z"], c=c, s=60)
                self.ax.text(o["x"], o["y"], o["z"], f"{o['type']} | {o['speed']:.1f} km/h",
                             fontsize=8, color="black")
                if o["vx"] or o["vy"] or o["vz"]:
                    self.ax.quiver(o["x"], o["y"], o["z"], o["vx"], o["vy"], o["vz"],
                                   length=0.5, normalize=True, color=c, linewidth=1.5)

            now = time.time()
            if now - self._last_scatter_save >= self._scatter_save_every:
                try:
                    os.makedirs("static", exist_ok=True)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png", dir="static") as tmp:
                        self.fig.savefig(tmp.name, dpi=100, bbox_inches="tight")
                        os.replace(tmp.name, "static/scatter_3d.png")
                except Exception as e:
                    print(f"[SCATTER SAVE ERROR] {e}")


    def _generate_voxel_heatmaps(self):
        """
        Simple occupancy grid from current objects; saves XY/XZ/YZ projections.
        """
        if not self.objects:
            return

        vx = 0.5  # meter bin
        x_bounds, y_bounds, z_bounds = (-5, 5), (0, 10), (-2, 2)
        x_bins = int((x_bounds[1] - x_bounds[0]) / vx)
        y_bins = int((y_bounds[1] - y_bounds[0]) / vx)
        z_bins = int((z_bounds[1] - z_bounds[0]) / vx)

        grid = np.zeros((x_bins, y_bins, z_bins), dtype=np.float32)
        for o in self.objects:
            i = int((o["x"] - x_bounds[0]) / vx)
            j = int((o["y"] - y_bounds[0]) / vx)
            k = int((o["z"] - z_bounds[0]) / vx)
            if 0 <= i < x_bins and 0 <= j < y_bins and 0 <= k < z_bins:
                grid[i, j, k] += 1.0

        xy = np.sum(grid, axis=2)
        xz = np.sum(grid, axis=1)
        yz = np.sum(grid, axis=0)

        try:
            for name, arr in (("heatmap_xy.png", xy.T),
                              ("heatmap_xz.png", xz.T),
                              ("heatmap_yz.png", yz.T)):
                tmp = os.path.join(self.static_dir, f".{name}.tmp")
                dst = os.path.join(self.static_dir, name)
                plt.imsave(tmp, arr, cmap="hot", origin="lower")
                os.replace(tmp, dst)
        except Exception as e:
            print(f"[VOXEL SAVE ERROR] {e}")

    def _color(self, t: str):
        cmap = {
            "CAR": "red", "TRUCK": "orange", "BUS": "purple",
            "HUMAN": "blue", "PERSON": "blue",
            "BIKE": "green", "BICYCLE": "cyan", "UNKNOWN": "gray",
        }
        return cmap.get(t.upper(), "black")
