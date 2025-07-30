import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class Live3DPlotter:
    def __init__(self):
        self.history = []
        self.fig = plt.figure(figsize=(7, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        plt.ion()
        plt.show()

    def update(self, targets):
        for t in targets:
            x, y, z = t.get("posX"), t.get("posY"), t.get("posZ")
            if x is not None and y is not None and z is not None:
                self.history.append((x, y, z))
        if len(self.history) > 1000:
            self.history = self.history[-1000:]

    def render(self):
        if not self.history:
            return
        arr = np.array(self.history)
        H, _ = np.histogramdd(arr, bins=(20, 20, 10), range=[[-2, 2], [0, 5], [-1, 1]])
        voxels = np.argwhere(H > 0)
        colors = H[H > 0].flatten()

        self.ax.clear()
        self.ax.scatter(voxels[:, 0], voxels[:, 1], voxels[:, 2], c=colors, cmap='hot', s=40)
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_zlabel("Z (m)")
        self.ax.set_title(f"3D Heatmap — {len(self.history)} pts")
        plt.pause(0.01)
