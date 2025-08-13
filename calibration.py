# calibration.py

import os
import json
import numpy as np
from bounding_box import annotate_speeding_object, model, project_radar_to_pixel
import glob
import cv2

def run_calibration():
    snapshots_dir = "snapshots"
    snapshots = sorted(glob.glob(os.path.join(snapshots_dir, "speeding_*.jpg")))

    output_csv = "calibration/calibration_points.csv"
    os.makedirs("calibration", exist_ok=True)

    with open(output_csv, "w") as f:
        f.write("radar_x,radar_y,radar_z,pixel_x,pixel_y\n")

        for path in snapshots:
            basename = os.path.basename(path).replace(".jpg", ".json")
            json_path = os.path.join("calibration", basename)
            if not os.path.exists(json_path):
                print(f"[SKIP] Missing radar JSON for {path}")
                continue

            try:
                with open(json_path) as jf:
                    data = json.load(jf)
                    obj_x = float(data.get("x", 0.0))
                    obj_y = float(data.get("y", 0.0))
                    obj_z = float(data.get("z", 0.0))

                    print(f"[CALIB] Running object detection for {path}")
                    img = cv2.imread(path)
                    if img is None:
                        print(f"[SKIP] Failed to read {path}")
                        continue
                    h, w = img.shape[:2]

                    results = model(source=img, imgsz=640)[0]
                    pixel_from_azimuth, _ = project_radar_to_pixel(obj_x, obj_y, obj_z, w)

                    best_box = None
                    best_score = float("inf")

                    for box in results.boxes:
                        cls_id = int(box.cls)
                        class_name = model.names[cls_id].lower()
                        conf = float(box.conf[0])

                        if class_name not in {"person", "car", "truck", "bus", "bicycle", "motorbike"}:
                            continue
                        if conf < 0.3:
                            continue

                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        bbox_center_x = (x1 + x2) / 2
                        angle_score = abs(bbox_center_x - pixel_from_azimuth)

                        if angle_score < best_score:
                            best_score = angle_score
                            best_box = (x1, y1, x2, y2)

                    if best_box:
                        x1, y1, x2, y2 = best_box
                        bbox_center_x = int((x1 + x2) / 2)
                        bbox_center_y = int((y1 + y2) / 2)

                        f.write(f"{obj_x},{obj_y},{obj_z},{bbox_center_x},{bbox_center_y}\n")
                        print(f"[OK] Wrote calibration row for {path}")
                    else:
                        print(f"[WARN] No matching object for {path}")

            except Exception as e:
                print(f"[FAIL] {path}: {e}")
