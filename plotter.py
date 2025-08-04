import matplotlib
matplotlib.use('Agg')  # Headless backend for environments without display

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import threading
import time
import os

class Live3DPlotter:
    def __init__(self):
        self.objects = []
        self.running = True
        self.heatmap_data = None
        self.last_heatmap_timestamp = 0
        self.lock = threading.Lock()

        self.fig = plt.figure(figsize=(8, 7))
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.thread = threading.Thread(target=self._render_loop, daemon=True)
        self.thread.start()

    def update(self, tracked_objects):
        with self.lock:
            self.objects = []
            for obj in tracked_objects:
                if all(k in obj for k in ("x", "y", "z", "vx", "vy", "vz")):
                    self.objects.append({
                        "x": obj["x"],
                        "y": obj["y"],
                        "z": obj["z"],
                        "vx": obj.get("vx", 0.0),
                        "vy": obj.get("vy", 0.0),
                        "vz": obj.get("vz", 0.0),
                        "id": obj.get("object_id", "obj"),
                        "type": obj.get("type", "UNKNOWN"),
                        "speed": obj.get("speed_kmh", 0.0)
                    })

    def _render_loop(self):
        while self.running:
            self.render()
            time.sleep(0.1)

    def stop(self):
        self.running = False
        plt.close()

    def render(self):
        with self.lock:
            self.ax.clear()
            self.ax.set_xlim([-5, 5])
            self.ax.set_ylim([0, 10])
            self.ax.set_zlim([-2, 2])
            self.ax.set_xlabel("X (m)")
            self.ax.set_ylabel("Y (m)")
            self.ax.set_zlabel("Z (m)")
            self.ax.set_title("IWR6843 Real-time 3D Tracking")

            for obj in self.objects:
                color = self._get_color(obj["type"])
                self.ax.scatter(obj["x"], obj["y"], obj["z"], c=color, s=60)
                label = f"{obj['type']} | {obj['speed']:.1f} km/h"
                self.ax.text(obj["x"], obj["y"], obj["z"], label, fontsize=8, color='black')
                self.ax.quiver(
                    obj["x"], obj["y"], obj["z"],
                    obj["vx"], obj["vy"], obj["vz"],
                    length=0.5, normalize=True, color=color, linewidth=1.5
                )

            # Save Doppler heatmap as image
            if self.heatmap_data is not None:
                try:
                    heatmap = self.heatmap_data
                    if heatmap.size == 0:
                        return

                    reshaped = None
                    for rows in [32, 64, 128]:
                        if heatmap.size % rows == 0:
                            try:
                                reshaped = heatmap.reshape((rows, -1))
                                break
                            except Exception:
                                continue
                    if reshaped is None:
                        print(f"[HEATMAP WARNING] Cannot reshape heatmap of size {heatmap.size}")
                        return
                    if reshaped.size == 0 or min(reshaped.shape) <= 1:
                        return

                    # Apply percentile-based normalization to stretch contrast
                    vmin = np.percentile(reshaped, 1)
                    vmax = np.percentile(reshaped, 99)

                    # Prevent degenerate case
                    if vmax - vmin < 1e-3:
                        vmax = vmin + 1

                    fig, ax = plt.subplots(figsize=(6, 4))
                    im = ax.imshow(reshaped, cmap='hot', interpolation='nearest', origin='lower',
                    aspect='auto', vmin=vmin, vmax=vmax)
                    ax.set_title("Range-Doppler Heatmap")
                    ax.set_xlabel("Doppler Bins")
                    ax.set_ylabel("Range Bins")
                    fig.colorbar(im, ax=ax, label='Intensity')
                    fig.tight_layout()

                    os.makedirs("static", exist_ok=True)
                    fig.savefig("static/heatmap.png")
                    plt.close(fig)

                except Exception as e:
                    print(f"[HEATMAP ERROR] {e}")

    def update_heatmap(self, heatmap):
        with self.lock:
            try:
                self.heatmap_data = np.array(heatmap, dtype=np.int16)
                self.last_heatmap_timestamp = time.time()
            except Exception as e:
                print(f"[PLOTTER ERROR] Failed to update heatmap: {e}")

    def _get_color(self, obj_type):
        cmap = {
            "CAR": "red",
            "HUMAN": "blue",
            "BIKE": "green",
            "TRUCK": "orange",
            "BUS": "purple",
            "BICYCLE": "cyan",
            "UNKNOWN": "gray"
        }
        return cmap.get(obj_type.upper(), "black")
