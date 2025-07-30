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
_last_reload_time = 0
radar = None
radar_csv_logger = IWR6843Logger()

def handle_exit(signum, frame):
    logger.info(f"Received signal {signum}, exiting...")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

def compute_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def main():
    global radar, config
    logger.info("[START] IWR6843 Radar Detection System")

    cli_port = config["iwr_ports"]["cli"]
    data_port = config["iwr_ports"]["data"]
    cfg_path = config["iwr_ports"]["cfg_path"]

    radar = IWR6843Interface(cli=cli_port, data=data_port, cfg_path=cfg_path)

    tracker = ObjectTracker(
        speed_limit_kmh=config.get("dynamic_speed_limits", {}).get("default", 3.0),
        speed_limits_map=config.get("dynamic_speed_limits", {})
    )
    classifier = ObjectClassifier()
    plotter = Live3DPlotter()

    last_snapshot_ids = {}
    COOLDOWN_SECONDS = config.get("cooldown_seconds", 0.1)

    try:
        while True:
            if os.path.exists("reload_flag.txt"):
                config = load_config()
                tracker.speed_limits_map = config.get("dynamic_speed_limits", {})
                os.remove("reload_flag.txt")

            targets = radar.get_targets()
            if not targets:
                time.sleep(0.2)
                continue

            now = time.time()
            for obj in targets:
                obj.update({
                    "timestamp": now,
                    "sensor": "IWR6843ISK",
                    "radar_distance": obj.get("range") or 0.0,
                    "velocity": obj.get("vRel", 0.0),
                    "speed_kmh": abs(obj.get("vRel", 0.0)) * 3.6,
                    "doppler_frequency": obj.get("vRel", 0.0) / 0.0625,
                    "signal_level": obj.get("snr", 0.0),
                    "direction": obj.get("motionState", "unknown"),
                    "confidence": 1.0
                })

            classified = classifier.classify_objects(targets)
            radar_csv_logger.log_targets(classified)
            tracked = tracker.update_tracks(classified, yolo_detections=[], frame_timestamp=now)
            plotter.update(classified)
            plotter.render()

            for obj in tracked:
                obj_id = obj['object_id']
                speed_limit = tracker.get_limit_for(obj.get("type", "UNKNOWN"))
                speeding = obj["speed_kmh"] > speed_limit
                recent = speeding_buffer[obj_id]

                if speeding:
                    recent.append(now)
                else:
                    recent.clear()

                should_trigger = speeding or (len(recent) >= 2 and now - recent[0] <= 2.0)
                last_taken = last_snapshot_ids.get(obj_id, 0)

                if should_trigger and now - last_taken > COOLDOWN_SECONDS:
                    last_snapshot_ids[obj_id] = now

                    cam = config.get("cameras", [{}])[config.get("selected_camera", 0)]
                    raw_path = capture_snapshot(cam.get("url"), cam.get("username"), cam.get("password"))
                    if raw_path and os.path.exists(raw_path):
                        frame = cv2.imread(raw_path)
                        if frame is not None:
                            sharpness = compute_sharpness(frame)
                            frame_buffer.append({"image": frame, "path": raw_path, "sharpness": sharpness})

                    if frame_buffer:
                        best = max(frame_buffer, key=lambda x: x["sharpness"])
                        label = f"{obj['type']} | {obj['speed_kmh']:.1f} km/h"
                        ann_path, visual_dist, updated_radar = annotate_speeding_object(
                            best["path"], obj['radar_distance'], label)

                        if ann_path:
                            obj["visual_distance"] = visual_dist
                            obj["radar_distance"] = updated_radar
                            conn = psycopg2.connect(dbname="radar_iwr6843", user="postgres",
                                                    password="1234", host="localhost")
                            cursor = conn.cursor()
                            cursor.execute("""
                                INSERT INTO radar_data (
                                    timestamp, datetime, sensor, object_id, type, confidence, speed_kmh,
                                    velocity, distance, radar_distance, visual_distance,
                                    direction, signal_level, doppler_frequency, snapshot_path,
                                    x, y, z, range, azimuth, elevation, motion_state
                                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                                        %s, %s, %s, %s, %s, %s, %s)
                            """, (
                                float(obj['timestamp']),
                                datetime.fromtimestamp(obj['timestamp']).strftime("%Y-%m-%d %H:%M:%S"),
                                obj['sensor'],
                                obj['object_id'],
                                obj['type'],
                                float(obj['confidence']),
                                float(obj['speed_kmh']),
                                float(obj['velocity']),
                                float(obj['radar_distance']),
                                float(obj['radar_distance']),
                                float(obj['visual_distance']),
                                obj['direction'],
                                float(obj['signal_level']),
                                float(obj['doppler_frequency']),
                                ann_path,
                                obj.get("x", 0.0),
                                obj.get("y", 0.0),
                                obj.get("z", 0.0),
                                obj.get("range", 0.0),
                                obj.get("azimuth", 0.0),
                                obj.get("elevation", 0.0),
                                obj.get("motionState", "unknown")
                            ))
                            conn.commit()
                            conn.close()
            time.sleep(0.05)

    except KeyboardInterrupt:
        logger.info("Interrupted. Exiting radar loop.")

if __name__ == "__main__":
    main()
