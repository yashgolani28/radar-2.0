from ultralytics import YOLO
import cv2
import os
from threading import Thread
import numpy as np

try:
    model = YOLO("yolov8s.pt")
    print("[INFO] Custom model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    model = None

def estimate_distance_from_box(bbox, frame_height):
    x1, y1, x2, y2 = bbox
    box_height = y2 - y1
    if box_height <= 0:
        return 0.0
    real_height_m = 1.7
    focal_length_px = 650  

    distance = (real_height_m * focal_length_px) / box_height
    return min(distance, 20.0)

def annotate_speeding_object(image_path, radar_distance, label=None, save_dir="snapshots",
                             min_confidence=0.3, obj_x=0.0, obj_y=1.0):
    print(f"[DEBUG] Annotating snapshot: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Failed to read image: {image_path}")
        return None, None, None

    results = model(source=img, imgsz=640)[0]
    h, w = img.shape[:2]
    best_box = None
    best_score = float("inf")

    azimuth_deg = np.degrees(np.arctan2(obj_x, obj_y))
    pixel_from_azimuth = (w / 2) + (azimuth_deg / 45.0) * (w / 2)  # 90° FOV assumption

    for box in results.boxes:
        cls_id = int(box.cls)
        class_name = model.names[cls_id].lower()
        conf = float(box.conf[0])

        if class_name not in {"person", "car", "truck", "bus", "bicycle", "motorbike"}:
            continue
        if conf < min_confidence:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        if bbox_w < 20 or bbox_h < 20:
            continue

        bbox_center = (x1 + x2) / 2
        angle_score = abs(bbox_center - pixel_from_azimuth)

        if angle_score < best_score:
            best_score = angle_score
            best_box = (x1, y1, x2, y2, class_name, conf)

    if best_box:
        x1, y1, x2, y2, class_name, conf = best_box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label_text = f"{class_name.upper()} ({conf:.2f}) | R={radar_distance:.1f}m"
        if label:
            label_text += f" | {label}"
        cv2.putText(img, label_text, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        cv2.imwrite(image_path, img)
        print(f"[DEBUG] Annotated: {class_name}, angle_score={best_score:.2f}")
        return image_path, radar_distance, radar_distance

    print("[DEBUG] No valid bounding box matched.")
    return None, None, None

def annotate_async(image_path, radar_distance, label=None, save_dir="snapshots", min_confidence=0.5, obj_x=0.0, obj_y=1.0):
    def _run():
        try:
            annotate_speeding_object(
                image_path=image_path,
                radar_distance=radar_distance,
                label=label,
                save_dir=save_dir,
                min_confidence=min_confidence,
                obj_x=obj_x,
                obj_y=obj_y
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
