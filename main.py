#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IWR6843ISK Radar Detection Core — Async Pipeline
- Main loop: parse radar → classify → track → enqueue work only
- Workers: violation (snapshot + annotate), DB insert, heatmap saver
- Detailed logging throughout + periodic health metrics
"""

import os
import sys
import time
import json
import csv
import atexit
import signal
import argparse
import traceback
from datetime import datetime
from collections import deque, defaultdict
from threading import Thread, Lock, Event
from queue import Queue, Empty

import numpy as np
import cv2
import matplotlib.pyplot as plt
import psycopg2
from sklearn.cluster import DBSCAN

# Local modules
from iwr6843_interface import IWR6843Interface
from kalman_filter_tracking import ObjectTracker
from classify_objects import ObjectClassifier
from bounding_box import annotate_speeding_object
from camera import capture_snapshot
from config_utils import load_config
from radar_logger import IWR6843Logger
from logger import logger
from plotter import Live3DPlotter

# ──────────────────────────────────────────────────────────────────────────────
# Config / Globals
# ──────────────────────────────────────────────────────────────────────────────
config = load_config()

# Buffers / Queues
frame_buffer = deque(maxlen=6)                         # local frame cache (sharpness-based)
speeding_buffer = defaultdict(lambda: deque(maxlen=5)) # persistence buffer
acceleration_cache = defaultdict(lambda: deque(maxlen=5))
last_snapshot_ids = {}

# Async job queues
_violation_q = Queue(maxsize=64)   # snapshot + annotation jobs
_db_q        = Queue(maxsize=256)  # DB write jobs
_heatmap_q   = Queue(maxsize=8)    # heatmap save/update jobs (coalesced)

# Metrics
_METRICS = {
    "enq_violation": 0, "done_violation": 0, "drop_violation": 0,
    "enq_db": 0,        "done_db": 0,        "drop_db": 0,
    "enq_heatmap": 0,   "done_heatmap": 0,   "drop_heatmap": 0,
    "frames": 0,        "classified": 0,     "tracked": 0
}
_METRICS_LOCK = Lock()

# Paths
violations_csv = "radar-logs/violations.csv"
os.makedirs("radar-logs", exist_ok=True)
os.makedirs("system-logs", exist_ok=True)
os.makedirs("snapshots", exist_ok=True)

# CSV logger
radar_csv_logger = IWR6843Logger()

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────
def to_native(val):
    if isinstance(val, (np.generic, np.ndarray)):
        try:
            return val.item()
        except Exception:
            return float(val)
    return val


_LAST_PROGRESS_TS = time.monotonic()

def get_targets_with_timeout(radar, timeout_s=0.8):
    """
    Call radar.get_targets() in a tiny worker thread and wait up to timeout_s.
    Returns a dict frame, or None if timed out (i.e., likely stalled read).
    """
    result = {}
    done = Event()

    def _worker():
        nonlocal result
        try:
            result = radar.get_targets()
        except Exception as e:
            logger.warning(f"[RADAR] get_targets error: {e}")
            result = {}
        finally:
            done.set()

    t = Thread(target=_worker, daemon=True)
    t.start()
    finished = done.wait(timeout_s)
    return result if finished else None


def _maybe_hard_exit_if_stalled(seconds_without_progress, limit_s=15):
    if seconds_without_progress >= limit_s:
        logger.error(f"[WATCHDOG] main loop stalled for {seconds_without_progress:.1f}s → exiting for restart")
        os._exit(21)  # let systemd restart us

def handle_exit(signum, frame):
    logger.info(f"[EXIT] Received signal {signum}, shutting down…")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

def compute_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def roi_motion_score(prev_img, curr_img, bbox):
    """Mean absolute pixel diff inside bbox; static patches are usually <~2–3."""
    if bbox is None:
        return 0.0
    x1, y1, x2, y2 = [int(max(0, v)) for v in bbox]
    h, w = curr_img.shape[:2]
    x1 = min(x1, w-1); x2 = min(x2, w); y1 = min(y1, h-1); y2 = min(y2, h)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return 0.0
    p = cv2.cvtColor(prev_img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    c = cv2.cvtColor(curr_img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    p = cv2.GaussianBlur(p, (5,5), 0)
    c = cv2.GaussianBlur(c, (5,5), 0)
    return float(cv2.absdiff(p, c).mean())

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

def _extract_xyz_doppler(points):
    """
    Accepts either list[dict] (x,y,z,doppler,snr,noise) or list[list] with columns
    [x,y,z,doppler,(snr),(noise),(trackIdx)]. Returns np arrays: X(N,3), D(N,), S(N,), N(N,)
    """
    if not points:
        return np.empty((0, 3), np.float32), np.empty((0,), np.float32), np.empty((0,), np.float32), np.empty((0,), np.float32)
    first = points[0]
    if isinstance(first, dict):
        X = np.array([[p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)] for p in points], np.float32)
        D = np.array([p.get("doppler", 0.0) for p in points], np.float32)
        S = np.array([p.get("snr", 0.0) for p in points], np.float32)
        N = np.array([p.get("noise", 0.0) for p in points], np.float32)
        return X, D, S, N
    A = np.array(points, np.float32)
    X = A[:, :3]
    D = A[:, 3] if A.shape[1] > 3 else np.zeros((A.shape[0],), np.float32)
    S = A[:, 4] if A.shape[1] > 4 else np.zeros((A.shape[0],), np.float32)
    N = A[:, 5] if A.shape[1] > 5 else np.zeros((A.shape[0],), np.float32)
    return X, D, S, N

def derive_targets_from_pointcloud(points):
    """
    Lightweight DBSCAN-based grouping when trackData is missing.
    Produces 'track-like' dicts expected by the pipeline.
    """
    XYZ, D, S, N = _extract_xyz_doppler(points)
    if XYZ.shape[0] == 0:
        return []
    labels = DBSCAN(eps=0.5, min_samples=6).fit_predict(XYZ)
    targets = []
    cid = 1
    for k in set(labels):
        if k == -1:
            continue
        mask = labels == k
        C = XYZ[mask]
        pos = C.mean(axis=0)
        # Convert average Doppler [Hz] to radial velocity [m/s], λ≈5mm at 60GHz → v≈f*λ/2
        v_rad = float(D[mask].mean()) * 0.005 / 2.0
        targets.append({
            # Do NOT emit 'id' here; it resets each frame and confuses tracking
            "pc_id": cid, "source": "pointcloud",
            "posX": float(pos[0]), "posY": float(pos[1]), "posZ": float(pos[2]),
            "velX": 0.0, "velY": v_rad, "velZ": 0.0,
            "snr": float(np.median(S[mask])) if S.size else 0.0,
            "noise": float(np.median(N[mask])) if N.size else 0.0,
            "speed_kmh": abs(v_rad) * 3.6,
            "distance": float(np.linalg.norm(pos)),
        })
        cid += 1
    return targets

# ──────────────────────────────────────────────────────────────────────────────
# Workers
# ──────────────────────────────────────────────────────────────────────────────
def _violation_worker():
    """
    Consumes violation jobs:
      1) Capture 3 quick snapshots → choose sharpest
      2) Annotate with YOLO
      3) Enqueue DB job (with range/noise profiles and optional heatmap)
    """
    while True:
        job = _violation_q.get()
        try:
            obj = job["obj"]
            cam = job["camera"]
            meta = job.get("frame_meta", {})
            heatmap = job.get("heatmap", None)

            oid = obj.get("object_id", "UNKNOWN")
            logger.info(f"[VIOLATION] Worker start for {oid} | {obj.get('type')} @ {obj.get('speed_kmh',0.0):.1f} km/h")

            # Burst capture: take up to 3 frames and pick the sharpest (thread-local, isolated)
            candidates = []
            for i in range(3):
                try:
                    path = capture_snapshot(
                        camera_url=cam.get("url"),
                        username=cam.get("username"),
                        password=cam.get("password"),
                        timeout=5
                    )
                    if path and os.path.exists(path):
                        img = cv2.imread(path)
                        if img is not None:
                            sharp = compute_sharpness(img)
                            candidates.append((sharp, path))
                            logger.debug(f"[CAMERA] {oid} snap#{i+1} sharpness={sharp:.2f} → {path}")
                except Exception as e:
                    logger.error(f"[CAMERA] {oid} snapshot error: {e}")
                time.sleep(0.05)

            ann_path = None
            if candidates:
                candidates.sort(key=lambda t: t[0], reverse=True)
                best_sharp, best_path = candidates[0]
                label = (f"{obj.get('type','UNKNOWN')} | {obj.get('speed_kmh',0.0):.1f} km/h | "
                         f"{obj.get('distance',0.0):.1f} m | "
                         f"Az:{obj.get('azimuth',0.0):.1f}° El:{obj.get('elevation',0.0):.1f}°")
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                tmp_copy = os.path.join("snapshots", f"temp_{oid}_{ts}.jpg")
                try:
                    best_img = cv2.imread(best_path)
                    cv2.imwrite(tmp_copy, best_img)
                    logger.info(f"[ANNOTATION] {oid} best sharpness={best_sharp:.2f}; annotating {tmp_copy}")
                    ann_path, visual_distance, corrected_distance, bbox = annotate_speeding_object(
                        image_path=tmp_copy,
                        radar_distance=max(0.1, min(float(obj.get("distance", 0.0)), 100.0)),
                        label=label,
                        obj_x=float(obj.get("x", 0.0)),
                        obj_y=float(obj.get("y", 0.0)),
                        obj_z=float(obj.get("z", 0.0))
                    )
                    obj["visual_distance"] = float(visual_distance or 0.0)
                    obj["distance"] = float(corrected_distance or obj.get("distance", 0.0))
                    obj["snapshot_status"] = "valid" if ann_path else "failed"
                except Exception as e:
                    logger.exception(f"[ANNOTATION] {oid} failed: {e}")
                    obj["snapshot_status"] = "failed"
                # Vision motion gate: reject static boxes (bags/signs)
                try:
                    if ann_path and bbox:
                        # use second-best as 'previous' if available
                        prev_path = candidates[1][1] if len(candidates) > 1 else None
                        if prev_path and os.path.exists(prev_path):
                            prev_img = cv2.imread(prev_path)
                            curr_img = cv2.imread(best_path)
                            thr = float(config.get("vision_motion_threshold", 3.0))
                            score = roi_motion_score(prev_img, curr_img, bbox)
                            logger.debug(f"[MOTION GATE] ROI mean diff = {score:.2f} (thr={thr})")
                            if score < thr:
                                logger.info("[MOTION GATE] Static ROI — dropping annotation.")
                                try:
                                    os.remove(ann_path)
                                except Exception:
                                    pass
                                ann_path = None
                                obj["snapshot_status"] = "failed"
                except Exception as e:
                    logger.debug(f"[MOTION GATE] skipped: {e}")

                # If we have a valid crop, give it a final, stable name for the gallery/DB
                if ann_path:
                    final_path = os.path.join("snapshots", f"violation_{oid}_{ts}.jpg")
                    try:
                        os.rename(ann_path, final_path)
                        ann_path = final_path
                    except Exception as e:
                        logger.debug(f"[RENAME] {ann_path} -> {final_path} failed: {e}")

                # Housekeeping: delete unused raw candidates and the temp working copy
                try:
                    for _, p in candidates:
                        if os.path.exists(p) and p != best_path:
                            os.remove(p)
                    if os.path.exists(tmp_copy):
                        os.remove(tmp_copy)
                except Exception as e:
                    logger.debug(f"[CLEANUP] skipped: {e}")
            else:
                logger.warning(f"[CAMERA] {oid} no valid snapshots captured")
                obj["snapshot_status"] = "failed"

            # Ensure velX/velY/velZ exist for DB logging
            for k in ("velX", "velY", "velZ"):
                obj.setdefault(k, float(obj.get(k, 0.0)))

            obj["snapshot_path"] = ann_path
            db_job = {
                "obj": obj,
                "range_profile": list(map(float, meta.get("range_profile", []))),
                "noise_profile": list(map(float, meta.get("noise_profile", []))),
                "heatmap": heatmap  # optional, for fallback image if DB insert fails
            }
            try:
                _db_q.put_nowait(db_job)
                with _METRICS_LOCK:
                    _METRICS["enq_db"] += 1
                logger.info(f"[QUEUE] → DB job enqueued for {oid}")
            except Exception:
                with _METRICS_LOCK:
                    _METRICS["drop_db"] += 1
                logger.error(f"[QUEUE] DB queue full; dropping job for {oid}")

        except Exception as e:
            logger.exception(f"[VIOLATION] Worker error: {e}")
        finally:
            with _METRICS_LOCK:
                _METRICS["done_violation"] += 1
            _violation_q.task_done()

def _db_worker():
    """
    Consumes DB jobs and writes to PostgreSQL.
    On failure: logs CSV fallback and tries to save a heatmap snapshot if provided.
    """
    while True:
        job = _db_q.get()
        try:
            obj = job["obj"]
            oid = obj.get("object_id", "UNKNOWN")
            logger.debug(f"[DB] Writing record for {oid}")

            with psycopg2.connect(dbname="iwr6843_db", user="radar_user",
                                  password="securepass123", host="localhost") as conn:
                cur = conn.cursor()
                cur.execute("""
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
                    job.get("range_profile", []),
                    job.get("noise_profile", [])
                ))
                conn.commit()
                logger.info(f"[DB] OK → {oid} | {obj.get('type')} | {obj.get('speed_kmh',0.0):.1f} km/h | {obj.get('snapshot_status')}")
        except Exception as e:
            # Fallbacks
            obj = job.get("obj", {})
            oid = obj.get("object_id", "UNKNOWN")
            logger.error(f"[DB] Insert failed for {oid}: {e}")
            try:
                log_violation_to_csv(obj)
                logger.warning(f"[DB] Fallback CSV logged for {oid}")
            except Exception as e2:
                logger.error(f"[DB] CSV fallback failed for {oid}: {e2}")

            # Save a fallback heatmap image if available
            try:
                H = job.get("heatmap", None)
                if H is not None:
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    fallback_path = os.path.join("snapshots", f"heatmap_{oid}_{timestamp_str}.jpg")
                    arr = None
                    # Try a few plausible row counts
                    flat = np.array(H).ravel()
                    for rows in (32, 64, 128):
                        if flat.size % rows == 0:
                            try:
                                arr = flat.reshape((rows, -1))
                                break
                            except Exception:
                                pass
                    if arr is not None and arr.size > 0:
                        vmin = np.percentile(arr, 1)
                        vmax = np.percentile(arr, 99)
                        fig, ax = plt.subplots(figsize=(6, 4))
                        im = ax.imshow(arr, cmap='hot', interpolation='nearest', origin='lower',
                                       aspect='auto', vmin=vmin, vmax=vmax)
                        ax.set_title("Range/Doppler Heatmap")
                        fig.colorbar(im, ax=ax, label="Intensity")
                        fig.tight_layout()
                        fig.savefig(fallback_path)
                        plt.close(fig)
                        logger.info(f"[HEATMAP] Fallback saved for {oid} → {fallback_path}")
                    else:
                        logger.warning(f"[HEATMAP] Fallback skipped; reshape failed for {oid}")
            except Exception as e3:
                logger.warning(f"[HEATMAP] Fallback error for {oid}: {e3}")
        finally:
            with _METRICS_LOCK:
                _METRICS["done_db"] += 1
            _db_q.task_done()

def _heatmap_worker(plotter: Live3DPlotter):
    """
    Coalesces heatmap updates — keeps only the newest payload.
    Payload can be any array-like; plotter.update_heatmap handles storage/writes.
    """
    pending = None
    while True:
        try:
            pending = _heatmap_q.get(timeout=0.5)
            # Drain to keep only the latest
            while True:
                pending = _heatmap_q.get_nowait()
        except Empty:
            pass
        except Exception as e:
            logger.warning(f"[HEATMAP] Worker queue error: {e}")

        if pending is not None:
            try:
                plotter.update_heatmap(pending)  # plotter will handle saving/logging
                with _METRICS_LOCK:
                    _METRICS["done_heatmap"] += 1
            except Exception as e:
                logger.warning(f"[HEATMAP] Update failed: {e}")
            finally:
                pending = None

def _metrics_worker():
    """
    Periodically prints queue sizes & throughput metrics.
    """
    while True:
        time.sleep(5)
        try:
            with _METRICS_LOCK:
                m = dict(_METRICS)
            logger.info(
                "[HEALTH] Qsizes: vio=%d db=%d hmap=%d | "
                "enq(vio/db/hmap)=%d/%d/%d | done(vio/db/hmap)=%d/%d/%d | drops(vio/db/hmap)=%d/%d/%d | "
                "frames=%d classified=%d tracked=%d",
                _violation_q.qsize(), _db_q.qsize(), _heatmap_q.qsize(),
                m["enq_violation"], m["enq_db"], m["enq_heatmap"],
                m["done_violation"], m["done_db"], m["done_heatmap"],
                m["drop_violation"], m["drop_db"], m["drop_heatmap"],
                m["frames"], m["classified"], m["tracked"]
            )
        except Exception:
            logger.warning("[HEALTH] Metrics print failed:\n" + traceback.format_exc())

class _NoPlotter:
    def update(self, *_a, **_k): pass
    def update_heatmap(self, *_a, **_k): pass

def _start_workers(plotter):
    # Violation pipeline workers (parallelize if needed)
    for _ in range(2):
        Thread(target=_violation_worker, daemon=True).start()
    # DB + Heatmap + Metrics
    Thread(target=_db_worker, daemon=True).start()
    Thread(target=_heatmap_worker, args=(plotter,), daemon=True).start()
    Thread(target=_metrics_worker, daemon=True).start()
    logger.info("[ASYNC] Workers started: 2x violation, 1x db, 1x heatmap, 1x metrics")

@atexit.register
def _dump_metrics_on_exit():
    with _METRICS_LOCK:
        summary = json.dumps(_METRICS, indent=2)
    logger.info(f"[EXIT] Final metrics:\n{summary}")

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    logger.info("[START] IWR6843 Radar Detection System (async enabled)")

    radar = IWR6843Interface()
    tracker = ObjectTracker(
        speed_limit_kmh=config.get("dynamic_speed_limits", {}).get("default", 3.0),
        speed_limits_map=config.get("dynamic_speed_limits", {})
    )
    classifier = ObjectClassifier()
    plotter = Live3DPlotter() if config.get("enable_plotter", False) else _NoPlotter()

    _start_workers(plotter)

    COOLDOWN_SECONDS = config.get("cooldown_seconds", 0.1)
    stall_count = 0
    RADAR_TIMEOUT_S = float(config.get("radar_get_timeout", 0.8))
    STALL_REINIT_AFTER = int(config.get("radar_stall_retries", 3))
    WATCHDOG_EXIT_AFTER = float(config.get("watchdog_hard_exit_after", 0))

    try:
        while True:
            # Hot-reload config marker
            if os.path.exists("reload_flag.txt"):
                logger.info("[CONFIG] Reload flag detected → reloading config")
                config.clear()
                config.update(load_config())
                tracker.speed_limits_map = config.get("dynamic_speed_limits", {})
                os.remove("reload_flag.txt")

            frame = get_targets_with_timeout(radar, timeout_s=RADAR_TIMEOUT_S)
            if frame is None:
                stall_count += 1
                # optional hard-exit if you want systemd to restart after long stalls
                if WATCHDOG_EXIT_AFTER > 0 and (stall_count * RADAR_TIMEOUT_S) >= WATCHDOG_EXIT_AFTER:
                    logger.error(f"[WATCHDOG] stalled for ~{stall_count * RADAR_TIMEOUT_S:.1f}s → exiting")
                    os._exit(21)
                if stall_count >= STALL_REINIT_AFTER:
                    logger.warning(f"[RADAR] get_targets timed out {stall_count}× → reinitializing interface")
                    try:
                        radar = IWR6843Interface()
                    except Exception as e:
                        logger.error(f"[RADAR] reinit failed: {e}")
                    stall_count = 0
                time.sleep(0.02)
                continue
            else:
                stall_count = 0
            with _METRICS_LOCK:
                _METRICS["frames"] += 1

            # Heatmap (enqueue only)
            heatmap = None
            for key in ("range_doppler_heatmap", "range_azimuth_heatmap", "azimuth_elevation_heatmap"):
                h = frame.get(key)
                if h is None:
                    continue
                try:
                    if isinstance(h, np.ndarray) and h.size > 0:
                        heatmap = h
                        break
                    if isinstance(h, (list, tuple)) and len(h) > 0:
                        heatmap = h
                        break
                    if isinstance(h, (bytes, bytearray, memoryview)) and len(h) > 0:
                        heatmap = h
                        break
                except Exception:
                    continue

            if heatmap is not None:
                try:
                    _heatmap_q.put_nowait(heatmap)
                    with _METRICS_LOCK:
                        _METRICS["enq_heatmap"] += 1
                except Exception:
                    with _METRICS_LOCK:
                        _METRICS["drop_heatmap"] += 1
                    pass  # best-effort only

            # Targets
            targets = frame.get("trackData", []) or []
            if not targets:
                pts = frame.get("pointCloud") or frame.get("point_cloud")
                pts_nonempty = (isinstance(pts, np.ndarray) and pts.size > 0) or (isinstance(pts, (list, tuple)) and len(pts) > 0)
                if pts_nonempty:
                    try:
                        targets = derive_targets_from_pointcloud(pts)
                        logger.debug(f"[POINTCLOUD] Derived {len(targets)} provisional targets")
                    except Exception as e:
                        logger.warning(f"[POINTCLOUD] Derivation failed: {e}")
                        time.sleep(0.05)
                        continue
                else:
                    time.sleep(0.05)
                    continue

            now = time.time()
            # Normalize and enrich
            for obj in targets:
                obj["x"] = float(obj.get("x", obj.get("posX", 0.0)))
                obj["y"] = float(obj.get("y", obj.get("posY", 0.0)))
                obj["z"] = float(obj.get("z", obj.get("posZ", 0.0)))
                obj["range"] = float(obj.get("distance", 0.0))

                # Angles
                if "azimuth" not in obj or "elevation" not in obj:
                    x, y, z = obj["x"], obj["y"], obj["z"]
                    obj["azimuth"]   = round(np.degrees(np.arctan2(x, y)), 2)
                    obj["elevation"] = round(np.degrees(np.arctan2(z, np.hypot(x, y))), 2)

                # RF fields
                if "doppler_frequency" not in obj and "doppler" in obj:
                    obj["doppler_frequency"] = float(obj.get("doppler", 0.0))
                obj["doppler_frequency"] = float(obj.get("doppler_frequency", 0.0))

                if "signal_level" not in obj and "signal" in obj:
                    obj["signal_level"] = float(obj.get("signal", 0.0))
                obj["signal_level"] = float(obj.get("signal_level", 0.0))

                # Velocity vector
                vx = float(obj.get("velX", obj.get("vx", 0.0)))
                vy = float(obj.get("velY", obj.get("vy", 0.0)))
                vz = float(obj.get("velZ", obj.get("vz", 0.0)))
                obj["velocity_vector"] = [vx, vy, vz]

                # Do not pass object_id from radar. Let the tracker assign persistent TRK IDs.
                obj.pop("object_id", None)

                # Source/TID labeling
                obj["source"] = obj.get("source", "")
                if obj["source"] != "pointcloud" and "id" in obj:
                    try:
                        obj["source_id"] = f"TID_{int(obj['id'])}"
                    except Exception:
                        obj["source_id"] = f"TID_{str(obj['id'])}"
                elif obj["source"] == "pointcloud" and "pc_id" in obj:
                    obj["source_id"] = f"PC_{int(obj['pc_id'])}"

                # Common fields
                obj.update({
                    "timestamp": now,
                    "sensor": "IWR6843ISK",
                    "distance": float(obj.get("distance", obj.get("range", 0.0))),
                    "velocity": float(obj.get("velocity", np.linalg.norm([vx, vy, vz]))),
                    "speed_kmh": float(obj.get("speed_kmh", np.linalg.norm([vx, vy, vz]) * 3.6)),
                    "direction": obj.get("direction", "unknown"),
                    "motion_state": obj.get("motion_state", "unknown"),
                    "confidence": float(obj.get("confidence", 1.0)),
                    "snapshot_status": "PENDING",
                })

            # Classify
            classifier = classifier or ObjectClassifier()
            classified = classifier.classify_objects(targets)
            with _METRICS_LOCK:
                _METRICS["classified"] += len(classified)
            logger.info(f"[CLASSIFY] {len(classified)} objects")

            # Human-readable prints (pre-track)
            for obj in classified:
                vv = obj.get('velocity_vector', [obj.get('velX', 0.0), obj.get('velY', 0.0), obj.get('velZ', 0.0)])
                acc = obj.get('acceleration', [obj.get('accX', 0.0), obj.get('accY', 0.0), obj.get('accZ', 0.0)])
                df = obj.get('doppler_frequency', obj.get('doppler', 0.0))
                sl = obj.get('signal_level', obj.get('signal', 0.0))
                print("\n---------- Radar Object (pre-track) ----------")
                if 'source_id' in obj: print(f"Source ID: {obj.get('source_id')}")
                elif 'id' in obj:      print(f"Source ID: {obj.get('id')}")
                print(f"Type: {obj.get('type','N/A')} | Conf: {obj.get('confidence',0.0):.2f}")
                print(f"Speed: {obj.get('speed_kmh',0.0):.2f} km/h | Vel: {obj.get('velocity',0.0):.2f} m/s")
                print(f"Dist: {obj.get('distance',0.0):.2f} m | Pos: x={obj.get('x',0.0):.2f} y={obj.get('y',0.0):.2f} z={obj.get('z',0.0):.2f}")
                print(f"VelVec: vx={vv[0]:.2f} vy={vv[1]:.2f} vz={vv[2]:.2f} | Acc: ax={acc[0]:.2f} ay={acc[1]:.2f} az={acc[2]:.2f}")
                print(f"Az:{obj.get('azimuth',0.0):.2f}° El:{obj.get('elevation',0.0):.2f}° | SNR:{obj.get('snr',0.0):.1f}dB")
                print(f"Doppler:{df:.2f}Hz | Signal:{sl:.1f} | Motion:{obj.get('motion_state','UNKNOWN')} Dir:{obj.get('direction','UNKNOWN')}")

            # Track
            norm = []
            for det in classified:
                d = dict(det)
                d['x'] = float(det.get('x',  det.get('posX', 0.0)))
                d['y'] = float(det.get('y',  det.get('posY', 0.0)))
                d['z'] = float(det.get('z',  det.get('posZ', 0.0)))
                vv = det.get('velocity_vector', [det.get('velX', 0.0), det.get('velY', 0.0), det.get('velZ', 0.0)])
                d['initial_velocity'] = [float(v) for v in vv]
                if 'doppler_frequency' not in d and 'doppler' in d:
                    d['doppler_frequency'] = float(det.get('doppler', 0.0))
                if 'signal_level' not in d and 'signal' in d:
                    d['signal_level'] = float(det.get('signal', 0.0))
                # Carry source/source_id; tracker will assign persistent object_id
                d['source'] = det.get('source', d.get('source', ''))
                if d['source'] != 'pointcloud' and 'id' in det:
                    try:
                        d['source_id'] = f"TID_{int(det['id'])}"
                    except Exception:
                        d['source_id'] = f"TID_{str(det['id'])}"
                elif d['source'] == 'pointcloud' and 'pc_id' in det:
                    d['source_id'] = f"PC_{int(det['pc_id'])}"
                d.pop('object_id', None)
                norm.append(d)

            tracked = tracker.update_tracks(norm, current_time=now)
            with _METRICS_LOCK:
                _METRICS["tracked"] += len(tracked)
            logger.info(f"[TRACK] {len(tracked)} active tracks")
            try:
                plotter.update(tracked)
            except Exception as e:
                logger.debug(f"[PLOTTER] update skipped: {e}")

            # Violation logic → enqueue only
            for obj in tracked:
                oid = obj['object_id']
                vv = obj.get('velocity_vector') or [obj.get('velX', 0.0), obj.get('velY', 0.0), obj.get('velZ', 0.0)]
                if 'velocity' not in obj or obj.get('velocity', 0.0) == 0.0:
                    obj['velocity'] = float(np.linalg.norm(vv))
                if 'speed_kmh' not in obj or obj.get('speed_kmh', 0.0) == 0.0:
                    obj['speed_kmh'] = obj['velocity'] * 3.6

                # Acceleration cache
                acc_mag = np.linalg.norm([obj.get("accX", 0.0), obj.get("accY", 0.0), obj.get("accZ", 0.0)])
                acceleration_cache[oid].append(acc_mag)

                speed_limit = tracker.get_limit_for(obj.get("type", "UNKNOWN"))
                speeding = obj["speed_kmh"] > speed_limit
                recent = speeding_buffer[oid]

                if speeding:
                    recent.append(now)
                else:
                    recent.clear()

                acc_buffer = acceleration_cache[oid]
                acc_threshold = config.get("acceleration_threshold", 2.0)
                acc_required_frames = config.get("min_acc_violation_frames", 3)
                acceleration_violating = (
                    len(acc_buffer) >= acc_required_frames and
                    sum(1 for a in acc_buffer if a > acc_threshold) >= acc_required_frames
                )

                should_trigger = (
                    (obj.get("motion_state") == "SPEEDING" and (len(recent) >= 2 and now - recent[0] <= 2.0))
                    or acceleration_violating
                )
                reason = "speeding" if speeding else ("acceleration" if acceleration_violating else "none")
                logger.info(f"[TRIGGER] {oid} reason={reason} v={obj['speed_kmh']:.1f} limit={speed_limit:.1f}")

                last_taken = last_snapshot_ids.get(oid, 0)
                if should_trigger and now - last_taken > COOLDOWN_SECONDS:
                    last_snapshot_ids[oid] = now
                    logger.info(f"[QUEUE] Enqueue violation for {oid} — {obj['type']} @ {obj['speed_kmh']:.1f} km/h")

                    # Camera config (supports multi-camera config structure)
                    cam_cfg = {}
                    cams = config.get("cameras")
                    if isinstance(cams, list) and cams:
                        idx = config.get("selected_camera", 0)
                        cam_cfg = cams[min(max(idx, 0), len(cams)-1)]
                    elif isinstance(cams, dict):
                        cam_cfg = cams
                    camera_payload = {
                        "url": cam_cfg.get("snapshot_url") or cam_cfg.get("url"),
                        "username": cam_cfg.get("username"),
                        "password": cam_cfg.get("password"),
                    }

                    # Prepare job payload
                    job_obj = dict(obj)  # shallow copy
                    frame_meta = {
                        "range_profile": frame.get("range_profile", []),
                        "noise_profile": frame.get("noise_profile", []),
                    }
                    job_heatmap = heatmap  # cheap reference; worker handles any heavy ops

                    try:
                        _violation_q.put_nowait({
                            "obj": job_obj,
                            "camera": camera_payload,
                            "frame_meta": frame_meta,
                            "heatmap": job_heatmap
                        })
                        with _METRICS_LOCK:
                            _METRICS["enq_violation"] += 1
                    except Exception:
                        with _METRICS_LOCK:
                            _METRICS["drop_violation"] += 1
                        logger.warning(f"[QUEUE] Violation queue full; dropping job for {oid}")

            time.sleep(0.02)  # keep the loop snappy

    except KeyboardInterrupt:
        logger.info("[END] Interrupted by user.")
    except Exception:
        logger.error("[FATAL] Unhandled exception in main loop:\n" + traceback.format_exc())

# ──────────────────────────────────────────────────────────────────────────────
# CLI Entrypoint
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", action="store_true", help="Force snapshots for calibration")
    args = parser.parse_args()
    RADAR_TIMEOUT_S = float(config.get("radar_get_timeout", 0.8))
    if args.calibrate:
        logger.info("[CALIBRATION MODE] Running snapshot capture loop")
        radar = IWR6843Interface()
        ObjectTracker()  # not used but keeps parity
        try:
            while True:
                frame = get_targets_with_timeout(radar, timeout_s=RADAR_TIMEOUT_S)
                if frame is None:
                    time.sleep(0.05)
                    continue

                targets = frame.get("trackData", []) or []
                if not targets:
                    time.sleep(0.1)
                    continue

                for obj in targets:
                    obj.update({
                        "x": obj.get("posX", 0.0),
                        "y": obj.get("posY", 0.0),
                        "z": obj.get("posZ", 0.0),
                    })

                    cams = config.get("cameras")
                    if isinstance(cams, list) and cams:
                        idx = config.get("selected_camera", 0)
                        cam = cams[min(max(idx, 0), len(cams)-1)]
                    elif isinstance(cams, dict):
                        cam = cams
                    else:
                        cam = {}

                    raw_path = capture_snapshot(
                        camera_url=cam.get("snapshot_url") or cam.get("url"),
                        username=cam.get("username"),
                        password=cam.get("password")
                    )
                    if raw_path:
                        _ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        temp_name = f"temp_{obj.get('object_id','UNKNOWN')}_{_ts}.jpg"
                        temp_path = os.path.join("snapshots", temp_name)
                        cv2.imwrite(temp_path, cv2.imread(raw_path))
                        logger.info(f"[CALIBRATION] Saved snapshot → {temp_path}")

                        json_path = temp_path.replace(".jpg", ".json")
                        with open(json_path, "w") as jf:
                            json.dump({
                                "x": float(obj.get("x", 0.0)),
                                "y": float(obj.get("y", 0.0)),
                                "z": float(obj.get("z", 0.0))
                            }, jf, indent=2)
                        logger.info(f"[CALIBRATION] Saved radar JSON → {json_path}")
                time.sleep(2)
        except KeyboardInterrupt:
            logger.info("[CALIBRATION MODE] Exit requested by user.")
    else:
        main()

def start_main_loop():
    t = Thread(target=main, daemon=True)
    t.start()
