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
import matplotlib.pyplot as plt
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
acceleration_cache = defaultdict(lambda: deque(maxlen=5))  
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

            frame = radar.get_targets()
            targets = frame.get("trackData", [])
            heatmap = frame.get("range_doppler_heatmap", [])

            if not targets:
                time.sleep(0.1)
                continue

            plotter.update_heatmap(heatmap)

            now = time.time()
            for obj in targets:
                obj["x"] = obj.get("posX", 0.0)
                obj["y"] = obj.get("posY", 0.0)
                obj["z"] = obj.get("posZ", 0.0)
                obj["range"] = obj.get("distance", obj.get("distance", 0.0))

                if "object_id" not in obj:
                    date_str = datetime.now().strftime("%Y%m%d")
                    radar_id = obj.get("id")
                    try:
                        radar_id = int(radar_id)
                    except:
                        radar_id = 0
                    obj_type = obj.get("type", "UNKNOWN").upper()
                    obj["object_id"] = f"{obj_type}_{date_str}_{radar_id}"

                obj.update({
                    "timestamp": now,
                    "sensor": "IWR6843ISK",
                    "distance": obj.get("distance", 0.0),
                    "velocity": obj.get("velocity", 0.0),
                    "speed_kmh": obj.get("speed_kmh", 0.0),
                    "doppler_frequency": obj.get("doppler_frequency", 0.0),
                    "signal_level": obj.get("signal_level", 0.0),
                    "direction": obj.get("direction", "unknown"),
                    "motion_state": obj.get("motion_state", "unknown"),
                    "confidence": obj.get("confidence", 1.0),
                    "snapshot_status": "PENDING"
                })

            classified = classifier.classify_objects(targets)
            for obj in classified:
                print("\n---------- Radar Object ----------")
                print(f"Object ID: {obj.get('object_id', 'N/A')}")
                print(f"Type: {obj.get('type', 'N/A')} | Confidence: {obj.get('confidence', 0.0):.2f}")
                print(f"Speed: {obj.get('speed_kmh', 0.0):.2f} km/h | Velocity: {obj.get('velocity', 0.0):.2f} m/s")
                print(f"Distance: {obj.get('distance', 0.0):.2f} m")
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
                print(f"Distance: {obj.get('distance', 0.0):.2f} m")
                print(f"Position: x={obj.get('x', 0.0):.2f}, y={obj.get('y', 0.0):.2f}, z={obj.get('z', 0.0):.2f}")
                print(f"Velocity Vector: vx={obj.get('velX', 0.0):.2f}, vy={obj.get('velY', 0.0):.2f}, vz={obj.get('velZ', 0.0):.2f}")
                print(f"Acceleration: ax={obj.get('accX', 0.0):.2f}, ay={obj.get('accY', 0.0):.2f}, az={obj.get('accZ', 0.0):.2f}")
                print(f"Azimuth: {obj.get('azimuth', 0.0):.2f}° | Elevation: {obj.get('elevation', 0.0):.2f}°")
                print(f"SNR: {obj.get('snr', 0.0):.1f} dB | Gain: {obj.get('gain', 0.0):.1f} dB")
                print(f"Doppler Frequency: {obj.get('doppler_frequency', 0.0):.2f} Hz | Signal Level: {obj.get('signal_level', 0.0):.1f}")
                print(f"Motion State: {obj.get('motion_state', 'unknown')} | Direction: {obj.get('direction', 'unknown')}")

                obj_id = obj['object_id']
                # Cache acceleration magnitude
                acc_mag = np.linalg.norm([
                    obj.get("accX", 0.0),
                    obj.get("accY", 0.0),
                    obj.get("accZ", 0.0)
                ])
                acceleration_cache[obj_id].append(acc_mag)

                speed_limit = tracker.get_limit_for(obj.get("type", "UNKNOWN"))
                speeding = obj["speed_kmh"] > speed_limit
                acc_buffer = acceleration_cache[obj_id]
                acc_threshold = config.get("acceleration_threshold", 2.0)
                acc_required_frames = config.get("min_acc_violation_frames", 3)

                # Only evaluate if enough samples exist
                acceleration_violating = (
                    len(acc_buffer) >= acc_required_frames and
                    sum(1 for a in acc_buffer if a > acc_threshold) >= acc_required_frames
                )
                recent = speeding_buffer[obj_id]

                if speeding:
                    recent.append(now)
                else:
                    recent.clear()

                # Trigger only if object is speeding and recent buffer confirms persistence
                should_trigger = (
                    (
                        obj.get("motion_state") == "SPEEDING" and
                        (len(recent) >= 2 and now - recent[0] <= 2.0)
                    )
                    or acceleration_violating
                )
                reason = "speeding" if speeding else "acceleration"
                logger.info(f"Trigger reason: {reason} for {obj_id}")
                last_taken = last_snapshot_ids.get(obj_id, 0)

                if should_trigger and now - last_taken > COOLDOWN_SECONDS:
                    last_snapshot_ids[obj_id] = now
                    logger.info(f"Triggering snapshot for {obj_id} — {obj['type']} @ {obj['speed_kmh']:.1f} km/h")

                    cam = config.get("cameras", [{}])[config.get("selected_camera", 0)]
                    raw_path = capture_snapshot(
                        camera_url=cam.get("snapshot_url"),
                        username=cam.get("username"),
                        password=cam.get("password")
                    )
                    ann_path = None

                    if raw_path and os.path.exists(raw_path):
                        frame = cv2.imread(raw_path)
                        if frame is not None:
                            sharpness = compute_sharpness(frame)
                            logger.debug(f"Snapshot {raw_path} sharpness = {sharpness:.2f}")
                            frame_buffer.append({"image": frame, "path": raw_path, "sharpness": sharpness})

                    if frame_buffer:
                        best = max(frame_buffer, key=lambda x: x["sharpness"])
                        label = f"{obj['type']} | {obj['speed_kmh']:.1f} km/h | {obj['distance']:.1f} m"

                        # Copy image for annotation
                        frame_copy_path = best["path"]
                        temp_annotated_path = os.path.join("snapshots", f"temp_{obj_id}.jpg")
                        cv2.imwrite(temp_annotated_path, cv2.imread(frame_copy_path))
                        logger.info(f"[ANNOTATION] Trying to annotate: {frame_copy_path} | Label: {label}")

                        ann_path, _, _ = annotate_speeding_object(
                            image_path=temp_annotated_path,
                            radar_distance=obj['distance'],
                            label=label,
                            obj_x=obj.get("x", 0.0),
                            obj_y=obj.get("y", 0.0)
                        )

                        if ann_path:
                            obj.update({
                                "snapshot_path": ann_path,
                                "snapshot_status": "SUCCESS"
                            })
                            logger.info(f"Annotated snapshot for {obj_id}: {ann_path} | Distance = {obj['distance']:.2f}m | Speed = {obj['speed_kmh']:.1f}km/h ")
                        else:
                            obj["snapshot_status"] = "FAILED"
                            logger.warning(f"[ANNOTATION FAILED] Returned None for {obj_id}")

                    # Log to PostgreSQL regardless
                    try:
                        conn = psycopg2.connect(dbname="iwr6843_db", user="radar_user", password="securepass123", host="localhost")
                        cursor = conn.cursor()
                        cursor.execute("""
                            INSERT INTO radar_data (
                                timestamp, datetime, sensor, object_id, type, confidence, speed_kmh,
                                velocity, distance, direction, signal_level, doppler_frequency, snapshot_path,
                                x, y, z, range, azimuth, elevation, motion_state, snapshot_status,
                                velx, vely, velz, snr, noise,
                                accx, accy, accz,
                                range_profile, noise_profile
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s,
                            %s, %s, %s,
                            %s, %s)
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
                            to_native(obj.get("snapshot_status", "FAILED")),
                            to_native(obj.get("velX", 0.0)),
                            to_native(obj.get("velY", 0.0)),
                            to_native(obj.get("velZ", 0.0)),
                            to_native(obj.get("snr", 0.0)),
                            to_native(obj.get("noise", 0.0)),
                            to_native(obj.get("accX", 0.0)),
                            to_native(obj.get("accY", 0.0)),
                            to_native(obj.get("accZ", 0.0)),
                            list(map(float, frame.get("range_profile", []))),
                            list(map(float, frame.get("noise_profile", [])))
                        ))
                        conn.commit()
                        conn.close()
                        logger.info(f"Logged violation: {obj_id} | Type: {obj['type']} | Speed: {obj['speed_kmh']:.2f} km/h | Status: {obj['snapshot_status']}")
                    except Exception as e:
                        logger.error(f"Database insert failed for {obj_id}: {e}")

                    if obj["snapshot_status"] != "SUCCESS":
                        log_violation_to_csv(obj)
                        logger.warning(f"Logged fallback to CSV for {obj_id} due to snapshot failure")

                        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                        fallback_path = os.path.join("snapshots", f"heatmap_{obj_id}_{timestamp_str}.jpg")

                        try:
                            reshaped = None
                            if plotter.heatmap_data is not None and plotter.heatmap_data.size > 0:
                                for rows in [32, 64, 128]:
                                    if plotter.heatmap_data.size % rows == 0:
                                        try:
                                            reshaped = plotter.heatmap_data.reshape((rows, -1))
                                            break
                                        except Exception:
                                            continue

                            if reshaped is not None:
                                vmin = np.percentile(reshaped, 1)
                                vmax = np.percentile(reshaped, 99)
                                fig, ax = plt.subplots(figsize=(6, 4))
                                im = ax.imshow(reshaped, cmap='hot', interpolation='nearest', origin='lower',
                                            aspect='auto', vmin=vmin, vmax=vmax)
                                ax.set_title("Range-Doppler Heatmap")
                                fig.colorbar(im, ax=ax, label="Intensity")
                                fig.tight_layout()
                                fig.savefig(fallback_path)
                                plt.close(fig)
                                logger.info(f"Saved fallback heatmap snapshot for {obj_id} → {fallback_path}")
                            else:
                                logger.warning(f"Skipped fallback heatmap save — reshaping failed for object {obj_id}")
                        except Exception as e:
                            logger.warning(f"Failed to save fallback heatmap for {obj_id}: {e}")

            time.sleep(0.05)

    except KeyboardInterrupt:
        logger.info("Interrupted. Exiting radar loop.")

if __name__ == "__main__":
    main()

def start_main_loop():
    t = Thread(target=main, daemon=True)
    t.start()
