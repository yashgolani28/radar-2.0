from ultralytics import YOLO
import cv2
import os
from threading import Thread

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
        return 0
    ref_width_px = 150
    ref_distance_m = 3.0
    est_distance = (ref_width_px / box_width) * ref_distance_m
    return est_distance

def annotate_speeding_object(image_path, radar_distance, label=None, save_dir="snapshots", min_confidence=0.5):
    print(f"[DEBUG] Annotating snapshot: {image_path}")
    img = cv2.imread(image_path)

    if img is None:
        print(f"[ERROR] Failed to read image: {image_path}")
        return None, None, None

    results = model(source=img, imgsz=640)[0]
    h, w = img.shape[:2]
    best_box = None
    highest_confidence = 0.0

    allowed_classes = {"human", "person", "car", "truck", "bus", "bike", "vehicle"}
    detected_boxes = []

    for box in results.boxes:
        cls_id = int(box.cls)
        conf = float(box.conf[0])
        class_name = model.names[cls_id].lower()

        if conf < min_confidence:
            print(f"[DEBUG] Skipping low confidence: {class_name} ({conf:.2f})")
            continue

        if class_name not in allowed_classes:
            print(f"[DEBUG] Ignoring irrelevant class: {class_name}")
            continue

        if conf > highest_confidence:
            highest_confidence = conf
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            best_box = (x1, y1, x2, y2, model.names[cls_id], conf)
            detected_boxes.append({
                "bbox": [x1, y1, x2, y2],
                "class": class_name,
                "confidence": conf
            })

    visual_dist = 0.0
    if best_box:
        x1, y1, x2, y2, cls_name, conf = best_box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        visual_dist = min(estimate_distance_from_box((x1, y1, x2, y2), frame_width=w), 12.0)
        if (x2 - x1) > 0.4 * w:
            radar_distance = min(radar_distance, visual_dist)

        text = f"{cls_name.upper()} ({conf:.2f}) | radar={radar_distance:.1f}m | visual={visual_dist:.1f}m"
        if label:
            text += f" | {label}"
        cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        print(f"[DEBUG] Annotated bounding box for {cls_name} using radar distance {radar_distance:.1f} m")
    else:
        print("[DEBUG] No relevant bounding box detected")

    cv2.imwrite(image_path, img)
    return image_path, visual_dist, radar_distance

def annotate_async(image_path, radar_distance, label=None, save_dir="snapshots", min_confidence=0.5):
    def _run():
        try:
            annotate_speeding_object(
                image_path=image_path,
                radar_distance=radar_distance,
                label=label,
                save_dir=save_dir,
                min_confidence=min_confidence
            )
        except Exception as e:
            print(f"[ANNOTATION ASYNC FAIL] {e}")

    def low_priority():
        try: os.nice(10)
        except: pass
        _run()

    Thread(target=low_priority, daemon=True).start()

