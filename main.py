#!/usr/bin/env python3

import time
import os
import sys
import json
import signal
import atexit
from datetime import datetime
from collections import deque, defaultdict
from threading import Thread
import cv2
import numpy as np
import csv
import psycopg2
from iwr6843_interface import IWR6843Interface
from kalman_filter_tracking import ObjectTracker
from classify_objects import ObjectClassifier
from bounding_box import annotate_speeding_object
from camera import capture_snapshot
from config_utils import load_config
from radar_logger import IWR6843Logger
from logger import logger
from plotter import Live3DPlotter

config = load_config()
frame_buffer = deque(maxlen=6)
speeding_buffer = defaultdict(lambda: deque(maxlen=5))
last_snapshot_ids = {}
radar_csv_logger = IWR6843Logger()
violations_csv = "radar-logs/violations.csv"
os.makedirs("radar-logs", exist_ok=True)
os.makedirs("system-logs", exist_ok=True)

def to_native(val):
    if isinstance(val, (np.generic, np.ndarray)):
        return val.item()
    return val

def handle_exit(signum, frame):
    logger.info(f"Received signal {signum}, exiting...")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

def compute_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def log_violation_to_csv(obj, note="Snapshot Failed"):
    with open(violations_csv, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.fromtimestamp(obj["timestamp"]).strftime("%Y-%m-%d %H:%M:%S"),
            obj.get("sensor", "IWR6843ISK"),
            obj.get("object_id"),
            obj.get("type"),
            obj.get("confidence", 0.0),
            obj.get("speed_kmh", 0.0),
            obj.get("velocity", 0.0),
            obj.get("distance", 0.0),
            obj.get("radar_distance", 0.0),
            obj.get("visual_distance", 0.0),
            obj.get("direction", "unknown"),
            obj.get("signal_level", 0.0),
            obj.get("doppler_frequency", 0.0),
            obj.get("motion_state", "unknown"),
            note
        ])

def main():
    logger.info("[START] IWR6843 Radar Detection System")

    radar = IWR6843Interface()
    tracker = ObjectTracker(
        speed_limit_kmh=config.get("dynamic_speed_limits", {}).get("default", 3.0),
        speed_limits_map=config.get("dynamic_speed_limits", {})
    )
    classifier = ObjectClassifier()
    plotter = Live3DPlotter()

    COOLDOWN_SECONDS = config.get("cooldown_seconds", 0.1)

    try:
        while True:
            if os.path.exists("reload_flag.txt"):
                config.clear()
                config.update(load_config())
                tracker.speed_limits_map = config.get("dynamic_speed_limits", {})
                os.remove("reload_flag.txt")

            targets = radar.get_targets()
            if not targets:
                time.sleep(0.1)
                continue

            now = time.time()
            for obj in targets:
                obj["x"] = obj.get("posX", 0.0)
                obj["y"] = obj.get("posY", 0.0)
                obj["z"] = obj.get("posZ", 0.0)
                obj["range"] = obj.get("radar_distance", obj.get("distance", 0.0))

                if "object_id" not in obj:
                    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    radar_id = int(obj.get("id", 0))
                    obj_type = obj.get("type", "UNKNOWN").upper()
                    obj["object_id"] = f"{obj_type}_{now_str}_{radar_id}"

                obj.update({
                    "timestamp": now,
                    "sensor": "IWR6843ISK",
                    "radar_distance": obj.get("distance", 0.0),
                    "velocity": obj.get("velocity", 0.0),
                    "speed_kmh": obj.get("speed_kmh", 0.0),
                    "doppler_frequency": obj.get("doppler_frequency", 0.0),
                    "signal_level": obj.get("signal_level", 0.0),
                    "direction": obj.get("direction", "unknown"),
                    "motion_state": obj.get("motion_state", "unknown"),
                    "confidence": obj.get("confidence", 1.0),
                    "visual_distance": 0.0,
                    "snapshot_status": "PENDING"
                })

            classified = classifier.classify_objects(targets)
            for obj in classified:
                print("\n---------- Radar Object ----------")
                print(f"Object ID: {obj.get('object_id', 'N/A')}")
                print(f"Type: {obj.get('type', 'N/A')} | Confidence: {obj.get('confidence', 0.0):.2f}")
                print(f"Speed: {obj.get('speed_kmh', 0.0):.2f} km/h | Velocity: {obj.get('velocity', 0.0):.2f} m/s")
                print(f"Radar Distance: {obj.get('radar_distance', 0.0):.2f} m")
                print(f"Position: x={obj.get('posX', 0.0):.2f}, y={obj.get('posY', 0.0):.2f}, z={obj.get('posZ', 0.0):.2f}")
                print(f"Velocity Vector: vx={obj.get('velX', 0.0):.2f}, vy={obj.get('velY', 0.0):.2f}, vz={obj.get('velZ', 0.0):.2f}")
                print(f"Acceleration: ax={obj.get('accX', 0.0):.2f}, ay={obj.get('accY', 0.0):.2f}, az={obj.get('accZ', 0.0):.2f}")
                print(f"Azimuth: {obj.get('azimuth', 0.0):.2f}° | Elevation: {obj.get('elevation', 0.0):.2f}°")
                print(f"SNR: {obj.get('snr', 0.0):.1f} dB | Gain: {obj.get('gain', 0.0):.1f} dB")
                print(f"Doppler Frequency: {obj.get('doppler', 0.0):.2f} Hz | Signal Level: {obj.get('signal', 0.0):.1f}")
                print(f"Motion: {obj.get('motion_state', 'UNKNOWN')} | Direction: {obj.get('direction', 'UNKNOWN')}")

            logger.info(f"Classified {len(classified)} objects")
            radar_csv_logger.log_targets(classified)
            tracked = tracker.update_tracks(classified, frame_timestamp=now)
            logger.info(f"Tracking {len(tracked)} objects")
            plotter.update(tracked)
            plotter.render()

            for obj in tracked:
                print("\n--- Radar Object ---")
                print(f"ID: {obj.get('object_id', 'N/A')}")
                print(f"Type: {obj.get('type', 'N/A')} | Confidence: {obj.get('confidence', 0.0):.2f}")
                print(f"Speed: {obj.get('speed_kmh', 0.0):.2f} km/h | Velocity: {obj.get('velocity', 0.0):.2f} m/s")
                print(f"Distance: {obj.get('distance', 0.0):.2f} m | Radar Distance: {obj.get('radar_distance', 0.0):.2f} m")
                print(f"Position: x={obj.get('x', 0.0):.2f}, y={obj.get('y', 0.0):.2f}, z={obj.get('z', 0.0):.2f}")
                print(f"Velocity Vector: vx={obj.get('velX', 0.0):.2f}, vy={obj.get('velY', 0.0):.2f}, vz={obj.get('velZ', 0.0):.2f}")
                print(f"Acceleration: ax={obj.get('accX', 0.0):.2f}, ay={obj.get('accY', 0.0):.2f}, az={obj.get('accZ', 0.0):.2f}")
                print(f"Azimuth: {obj.get('azimuth', 0.0):.2f}° | Elevation: {obj.get('elevation', 0.0):.2f}°")
                print(f"SNR: {obj.get('snr', 0.0):.1f} dB | Gain: {obj.get('gain', 0.0):.1f} dB")
                print(f"Doppler Frequency: {obj.get('doppler_frequency', 0.0):.2f} Hz | Signal Level: {obj.get('signal_level', 0.0):.1f}")
                print(f"Motion State: {obj.get('motion_state', 'unknown')} | Direction: {obj.get('direction', 'unknown')}")

                obj_id = obj['object_id']
                speed_limit = tracker.get_limit_for(obj.get("type", "UNKNOWN"))
                speeding = obj["speed_kmh"] > speed_limit
                recent = speeding_buffer[obj_id]

                if speeding:
                    recent.append(now)
                else:
                    recent.clear()

                # Trigger only if object is speeding and recent buffer confirms persistence
                should_trigger = (
                    obj.get("motion_state") == "SPEEDING" and
                    (len(recent) >= 2 and now - recent[0] <= 2.0)
                )
                last_taken = last_snapshot_ids.get(obj_id, 0)

                if should_trigger and now - last_taken > COOLDOWN_SECONDS:
                    last_snapshot_ids[obj_id] = now
                    logger.info(f"Triggering snapshot for {obj_id} — {obj['type']} @ {obj['speed_kmh']:.1f} km/h")

                    cam = config.get("cameras", [{}])[config.get("selected_camera", 0)]
                    raw_path = capture_snapshot(cam.get("url"), cam.get("username"), cam.get("password"))
                    ann_path = None

                    if raw_path and os.path.exists(raw_path):
                        frame = cv2.imread(raw_path)
                        if frame is not None:
                            sharpness = compute_sharpness(frame)
                            logger.debug(f"Snapshot {raw_path} sharpness = {sharpness:.2f}")
                            frame_buffer.append({"image": frame, "path": raw_path, "sharpness": sharpness})

                    if frame_buffer:
                        best = max(frame_buffer, key=lambda x: x["sharpness"])
                        label = f"{obj['type']} | {obj['speed_kmh']:.1f} km/h"
                        ann_path, visual_dist, updated_radar = annotate_speeding_object(
                            image_path=best["path"],
                            radar_distance=obj['radar_distance'],
                            label=label,
                            obj_x=obj.get("x", 0.0),
                            obj_y=obj.get("y", 0.0)
                        )

                        if ann_path:
                            obj.update({
                                "visual_distance": visual_dist,
                                "radar_distance": updated_radar,
                                "snapshot_path": ann_path,
                                "snapshot_status": "SUCCESS"
                            })
                            logger.info(f"Annotated snapshot: {ann_path} | visual = {visual_dist:.2f}m, radar = {updated_radar:.2f}m")
                        else:
                            obj["snapshot_status"] = "FAILED"
                            logger.warning(f"Annotation failed for {obj_id}")

                    # Log to PostgreSQL regardless
                    try:
                        conn = psycopg2.connect(dbname="iwr6843_db", user="radar_user", password="securepass123", host="localhost")
                        cursor = conn.cursor()
                        cursor.execute("""
                            INSERT INTO radar_data (
                                timestamp, datetime, sensor, object_id, type, confidence, speed_kmh,
                                velocity, distance, radar_distance, visual_distance,
                                direction, signal_level, doppler_frequency, snapshot_path,
                                x, y, z, range, azimuth, elevation, motion_state, snapshot_status
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                                    %s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            to_native(obj['timestamp']),
                            datetime.fromtimestamp(to_native(obj['timestamp'])).strftime("%Y-%m-%d %H:%M:%S"),
                            to_native(obj.get('sensor', 'IWR6843ISK')),
                            to_native(obj.get('object_id')),
                            to_native(obj.get('type')),
                            to_native(obj.get('confidence', 1.0)),
                            to_native(obj.get('speed_kmh', 0.0)),
                            to_native(obj.get('velocity', 0.0)),
                            to_native(obj.get('distance', 0.0)),
                            to_native(obj.get('radar_distance', 0.0)),
                            to_native(obj.get('visual_distance', 0.0)),
                            to_native(obj.get("direction", "unknown")),
                            to_native(obj.get("signal_level", 0.0)),
                            to_native(obj.get("doppler_frequency", 0.0)),
                            to_native(obj.get("snapshot_path")),
                            to_native(obj.get("x", 0.0)),
                            to_native(obj.get("y", 0.0)),
                            to_native(obj.get("z", 0.0)),
                            to_native(obj.get("range", 0.0)),
                            to_native(obj.get("azimuth", 0.0)),
                            to_native(obj.get("elevation", 0.0)),
                            to_native(obj.get("motion_state", "unknown")),
                            to_native(obj.get("snapshot_status", "FAILED"))
                        ))
                        conn.commit()
                        conn.close()
                        logger.info(f"Logged violation: {obj_id} | Type: {obj['type']} | Speed: {obj['speed_kmh']:.2f} km/h | Status: {obj['snapshot_status']}")
                    except Exception as e:
                        logger.error(f"Database insert failed for {obj_id}: {e}")

                    if obj["snapshot_status"] != "SUCCESS":
                        log_violation_to_csv(obj)
                        logger.warning(f"Logged fallback to CSV for {obj_id} due to snapshot failure")

            time.sleep(0.05)

    except KeyboardInterrupt:
        logger.info("Interrupted. Exiting radar loop.")

if __name__ == "__main__":
    main()
