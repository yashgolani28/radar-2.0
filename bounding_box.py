from ultralytics import YOLO
import cv2
import os
from threading import Thread
import numpy as np
import math
from logger import logger

PROJECTION_MATRIX_PATH = "calibration/camera_projection_matrix.npy"

try:
    model = YOLO("yolov8s.pt")
    print("[INFO] Custom model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    model = None

# Load projection matrix if available
projection_matrix = None
if os.path.exists(PROJECTION_MATRIX_PATH):
    try:
        projection_matrix = np.load(PROJECTION_MATRIX_PATH)
        print(f"[INFO] Loaded camera projection matrix from {PROJECTION_MATRIX_PATH}")
    except Exception as e:
        print(f"[WARN] Failed to load projection matrix: {e}")
        projection_matrix = None

def project_radar_to_pixel(x, y, z, image_width):
    try:
        coeffs = np.load("calibration/camera_projection_matrix.npy")
        pixel_x = coeffs[0] * x + coeffs[1]
        return float(pixel_x), 0.0
    except:
        return image_width / 2, 0.0

def estimate_distance_from_box(bbox, frame_height):
    x1, y1, x2, y2 = bbox
    box_height = y2 - y1

    if box_height <= 0 or not math.isfinite(box_height):
        return float('nan')  # Flag as invalid

    real_height_m = 1.7
    focal_length_px = 650

    distance = (real_height_m * focal_length_px) / box_height

    if not math.isfinite(distance) or distance > 100:
        return float('nan')
    return distance

def annotate_speeding_object(image_path, radar_distance, label=None, save_dir="snapshots",
                             min_confidence=0.55, obj_x=0.0, obj_y=1.0, obj_z=0.0):
    """
    Returns: (save_path, visual_distance, corrected_distance, bbox)
    - save_path: path to cropped image (or None)
    - visual_distance: distance estimated from YOLO box (meters) or None
    - corrected_distance: fused distance (meters)
    - bbox: (x1, y1, x2, y2) in the ORIGINAL image coordinates or None
    """
    print(f"[DEBUG] Annotating snapshot: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Failed to read image: {image_path}")
        return None, None, radar_distance, None

    results = model(source=img, imgsz=640)[0]
    h, w = img.shape[:2]
    best_box = None
    best_score = float("inf")

    pixel_from_azimuth, _ = project_radar_to_pixel(obj_x, obj_y, obj_z, w)
    print(f"[DEBUG] Radar azimuth projected to pixel: {pixel_from_azimuth}")

    allowed_classes = {"person", "car", "truck", "bus", "bicycle", "motorbike"}

    for box in results.boxes:
        cls_id = int(box.cls)
        class_name = model.names[cls_id].lower()
        conf = float(box.conf[0])
        if class_name not in allowed_classes:
            continue
        if conf < min_confidence:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if (x2 - x1) < 20 or (y2 - y1) < 20:
            continue

        center_x = (x1 + x2) / 2
        angle_score = abs(center_x - pixel_from_azimuth)
        est_dist = estimate_distance_from_box((x1, y1, x2, y2), h)
        if math.isnan(est_dist):
            continue

        if angle_score < best_score and angle_score <= 120:  # tighter gate
            best_score = angle_score
            best_box = (x1, y1, x2, y2)

    # Safer fallback: only among allowed classes and min_confidence
    if not best_box:
        filtered = [b for b in results.boxes
                    if model.names[int(b.cls)].lower() in allowed_classes
                    and float(b.conf[0]) >= min_confidence]
        if filtered:
            print("[FALLBACK] Using top-confidence allowed box")
            top = max(filtered, key=lambda b: float(b.conf[0]))
            best_box = tuple(map(int, top.xyxy[0]))

    if not best_box:
        print("[DEBUG] No valid bounding box matched.")
        return None, None, radar_distance, None

    x1, y1, x2, y2 = best_box
    cropped = img[y1:y2, x1:x2]
    if cropped.size == 0:
        print("[ERROR] Cropped image is empty.")
        return None, None, radar_distance, None

    # light vertical tidy
    ch = cropped.shape[0]
    mtop = int(0.08 * ch); mbot = int(0.08 * ch)
    cropped = cropped[mtop:ch - mbot, :]

    resized = cv2.resize(cropped, (384, 448), interpolation=cv2.INTER_AREA)
    save_name = f"cropped_{os.path.basename(image_path)}"
    save_path = os.path.join(save_dir, save_name)
    cv2.imwrite(save_path, resized)
    print(f"[DEBUG] Saved clean cropped image: {save_path}")

    visual_distance = estimate_distance_from_box((x1, y1, x2, y2), h)
    if not math.isfinite(visual_distance):
        visual_distance = None

    # simple fusion: trust radar unless the two are close
    corrected = radar_distance
    if visual_distance is not None and abs(visual_distance - radar_distance) <= 3.0:
        corrected = 0.6 * radar_distance + 0.4 * visual_distance

    return save_path, visual_distance, corrected, best_box

def annotate_async(image_path, radar_distance, label=None, save_dir="snapshots", min_confidence=0.55,
                   obj_x=0.0, obj_y=1.0, obj_z=0.0):
    def _run():
        try:
            annotate_speeding_object(
                image_path=image_path,
                radar_distance=radar_distance,
                label=label,
                save_dir=save_dir,
                min_confidence=min_confidence,
                obj_x=obj_x,
                obj_y=obj_y,
                obj_z=obj_z
            )
        except Exception as e:
            print(f"[ANNOTATION ASYNC FAIL] {e}")

    def low_priority():
        try:
            os.nice(10)
        except:
            pass
        _run()

    Thread(target=low_priority, daemon=True).start()
