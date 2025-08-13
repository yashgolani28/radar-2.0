import numpy as np
import os
import csv

def fit_projection_matrix(csv_path="calibration/calibration_points.csv", output_path="calibration/camera_projection_matrix.npy"):
    radar_pts = []
    pixel_pts = []

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found.")

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            try:
                x, y, z, u, v = map(float, row)
                radar_pts.append([x])  # just X (azimuth)
                pixel_pts.append([u])  # just pixel X
            except:
                continue

    if len(radar_pts) < 5:
        raise ValueError("Not enough calibration points to fit model.")

    X = np.array(radar_pts)
    y = np.array(pixel_pts)

    # Add bias term (X_b = [x, 1])
    X_b = np.c_[X, np.ones_like(X)]
    theta = np.linalg.lstsq(X_b, y, rcond=None)[0]

    np.save(output_path, theta)
    print(f"[FIT] Saved projection matrix to {output_path}")
    print(f"[MODEL] Pixel X = {theta[0][0]:.2f} * Radar X + {theta[1][0]:.2f}")
    return theta
