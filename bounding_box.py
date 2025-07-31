from ultralytics import YOLO
import cv2
import os
from threading import Thread
import numpy as np

try:
    model = YOLO("human_vehicel_weight.pt")
    print("[INFO] Custom model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    model = None

def estimate_distance_from_box(bbox, frame_width):
    x1, y1, x2, y2 = bbox
    box_width = x2 - x1
    if box_width <= 0:
        return 0.0
    ref_width_px = 150
    ref_distance_m = 3.0
    return (ref_width_px / box_width) * ref_distance_m

def annotate_speeding_object(image_path, radar_distance, label=None, save_dir="snapshots",
                             min_confidence=0.5, obj_x=0.0, obj_y=1.0):
    print(f"[DEBUG] Annotating snapshot: {image_path}")
    img = cv2.imread(image_path)

    if img is None:
        print(f"[ERROR] Failed to read image: {image_path}")
        return None, None, None

    results = model(source=img, imgsz=640)[0]
    h, w = img.shape[:2]
    visual_dist = 0.0
    best_box = None
    best_score = float("inf")

    allowed_classes = {"human", "person", "car", "truck", "bus", "bike", "vehicle"}

    # Radar azimuth in degrees (camera and radar aligned)
    azimuth_deg = np.degrees(np.arctan2(obj_x, obj_y))  # x is lateral, y is forward
    camera_fov_deg = 90
    image_center_x = w / 2
    pixel_from_azimuth = image_center_x + (azimuth_deg / (camera_fov_deg / 2)) * image_center_x

    for box in results.boxes:
        cls_id = int(box.cls)
        conf = float(box.conf[0])
        class_name = model.names[cls_id].lower()

        if conf < min_confidence or class_name not in allowed_classes:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        vis_dist = estimate_distance_from_box((x1, y1, x2, y2), frame_width=w)
        bbox_center_x = (x1 + x2) / 2
        score = abs(bbox_center_x - pixel_from_azimuth)

        if score < best_score:
            best_score = score
            best_box = {
                "bbox": (x1, y1, x2, y2),
                "class_name": class_name,
                "conf": conf,
                "vis_dist": vis_dist
            }

    if best_box:
        x1, y1, x2, y2 = best_box["bbox"]
        class_name = best_box["class_name"]
        conf = best_box["conf"]
        visual_dist = min(best_box["vis_dist"], 12.0)

        if (x2 - x1) > 0.4 * w:
            radar_distance = min(radar_distance, visual_dist)

        label_text = f"{class_name.upper()} ({conf:.2f}) | radar={radar_distance:.1f}m | visual={visual_dist:.1f}m"
        if label:
            label_text += f" | {label}"

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        print(f"[DEBUG] Annotated {class_name} @ radar={radar_distance:.1f}m, azimuth={azimuth_deg:.1f}°, pixel={pixel_from_azimuth:.0f}")
    else:
        print("[DEBUG] No relevant bounding box detected")

    cv2.imwrite(image_path, img)
    return image_path, visual_dist, radar_distance

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
