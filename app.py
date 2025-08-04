# System + CV
import os
import requests
import time
import json
import uuid
import cv2
import base64
import zipfile
import shutil
import psutil
import sys
import csv
import logging
import threading
import subprocess
import traceback
import numpy as np
import threading
from datetime import datetime, timedelta
from collections import deque, Counter
from io import BytesIO
import io
from contextlib import contextmanager

from flask import (
    Flask, render_template, request, redirect, url_for, send_file,
    send_from_directory, session, jsonify, flash, abort, Response, stream_with_context
)
from flask_login import (
    LoginManager, login_user, login_required, logout_user,
    UserMixin, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_socketio import SocketIO
from flask_cors import CORS
from requests.auth import HTTPDigestAuth

# External models + tools
import lightgbm as lgb
import psycopg2
import psycopg2.extras
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import joblib
import secrets
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Internal modules
from config_utils import load_config
from logger import logger
from iwr6843_interface import IWR6843Interface, check_radar_connection
from classify_objects import ObjectClassifier
from kalman_filter_tracking import ObjectTracker
from camera import capture_snapshot
from bounding_box import annotate_speeding_object
from report import generate_pdf_report
from train_lightbgm import fetch_training_data 
from plotter import Live3DPlotter
from main import main as main, start_main_loop

# Flask app init
app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins="*")
CORS(app)
config = load_config()
log_lock = threading.Lock()
logger = logging.getLogger("RadarLogger")

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

if not logger.hasHandlers():
    logger.addHandler(handler)
# Camera setup
selected = config.get("selected_camera", 0)
cam = config.get("cameras", [{}])[selected] if isinstance(config.get("cameras"), list) else {}
camera_url = cam.get("url")
camera_auth = HTTPDigestAuth(cam.get("username"), cam.get("password")) if cam.get("username") else None
camera_frame = None
camera_lock = threading.Lock()
last_frame = None
camera_capture = None
camera_enabled = cam.get("enabled", True)
plotter = Live3DPlotter()
last_heatmap = None
heatmap_lock = threading.Lock()

# Folders
SNAPSHOT_FOLDER = "snapshots"
BACKUP_FOLDER = "backups"
CONFIG_FILE = "app_config.json" 

# --- User Management ---
class User(UserMixin):
    def __init__(self, id_, username, password_hash, role="viewer"):
        self.id = id_
        self.username = username
        self.password_hash = password_hash
        self.role = role

    def get_id(self):
        return str(self.id)

    @property
    def is_authenticated(self):
        return True

def safe_log(level, msg):
    with log_lock:
        if level == "info":
            logger.info(msg)
        elif level == "error":
            logger.error(msg)
            
def get_user_by_id(user_id):
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute("SELECT id, username, password_hash, role FROM users WHERE id = %s", (user_id,))
        row = cursor.fetchone()
        return User(*row) if row else None


def get_user_by_username(username):
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute("SELECT id, username, password_hash, role FROM users WHERE username = %s", (username,))
        row = cursor.fetchone()
        return User(*row) if row else None


@contextmanager
def get_db_connection():
    conn = None
    max_retries = 3
    retry_delay = 0.2

    for attempt in range(max_retries):
        try:
            conn = psycopg2.connect(
                dbname="iwr6843_db",
                user="radar_user",
                password="securepass123",
                host="localhost"
            )
            break
        except psycopg2.OperationalError as e:
            logger.warning(f"[DB RETRY] Attempt {attempt + 1} failed: {e}")
            time.sleep(retry_delay)
            retry_delay *= 2
        except Exception as e:
            logger.error(f"[DB ERROR] Unexpected: {e}")
            raise

    try:
        yield conn
    finally:
        if conn:
            conn.close()

def save_model_metadata(version, features, accuracy, labels):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO model_metadata (version, features, accuracy, labels)
            VALUES (%s, %s, %s, %s)
        """, (version, features, accuracy, labels))
        conn.commit()

def get_latest_model_metadata():
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute("SELECT * FROM model_metadata ORDER BY trained_at DESC LIMIT 1")
        return cursor.fetchone()

def save_cameras_to_db(cameras, selected_idx):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM cameras")
        for i, cam in enumerate(cameras):
            cursor.execute("""
                INSERT INTO cameras (url, username, password, is_active, stream_type)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                cam.get("url"),
                cam.get("username"),
                cam.get("password"),
                i == selected_idx,
                cam.get("stream_type", "mjpeg")
            ))
        conn.commit()


def load_cameras_from_db():
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute("SELECT url, username, password, is_active, stream_type FROM cameras ORDER BY id")
        rows = cursor.fetchall()
        cameras = []
        selected = 0
        for i, row in enumerate(rows):
            cameras.append({                                                                                                                                                
                "url": row["url"],
                "username": row["username"],
                "password": row["password"],
                "stream_type": row["stream_type"] or "mjpeg",
                "enabled": row["is_active"]  # <-- Add this!
            })
            if row["is_active"]:
                selected = i
        return cameras, selected

def update_user_activity(user_id):
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO user_activity (user_id, last_activity)
                VALUES (%s, %s)
                ON CONFLICT (user_id)
                DO UPDATE SET last_activity = EXCLUDED.last_activity
            """, (user_id, datetime.utcnow()))
            conn.commit()
    except Exception as e:
        logger.error(f"[USER ACTIVITY] Update failed: {e}")


def get_active_users(minutes=30):
    try:
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            cursor.execute("""
                SELECT u.id, u.username, u.role, ua.last_activity
                FROM users u
                JOIN user_activity ua ON u.id = ua.user_id
                WHERE ua.last_activity >= %s
                ORDER BY ua.last_activity DESC
            """, (cutoff,))
            return cursor.fetchall()
    except Exception as e:
        logger.error(f"[USER ACTIVITY] Query failed: {e}")
        return []


def clean_inactive_sessions():
    try:
        cutoff = datetime.utcnow() - timedelta(hours=24)
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM user_activity WHERE last_activity < %s", (cutoff,))
            conn.commit()
    except Exception as e:
        logger.error(f"[SESSION CLEANUP] Failed: {e}")

def is_admin():
    return current_user.is_authenticated and getattr(current_user, "role", None) == "admin"

def save_config(config):
    """Save updated config to app_config.json"""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"[CONFIG SAVE ERROR] {e}")
        return False

def apply_pagination(total_items, page=1, limit=100):
    total_pages = max((total_items + limit - 1) // limit, 1)
    current_page = max(min(page, total_pages), 1)
    offset = (current_page - 1) * limit
    return offset, limit, total_pages, current_page

def ensure_directories():
    """Ensure required folders exist"""
    for directory in [SNAPSHOT_FOLDER, BACKUP_FOLDER]:
        os.makedirs(directory, exist_ok=True)

def validate_snapshots():
    """Remove missing snapshot paths from database"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            cursor.execute("SELECT id, snapshot_path FROM radar_data WHERE snapshot_path IS NOT NULL")
            rows = cursor.fetchall()

            invalid_count = 0
            for row in rows:
                path = row['snapshot_path']
                if not os.path.exists(path):
                    cursor.execute("UPDATE radar_data SET snapshot_path = NULL WHERE id = %s", (row['id'],))
                    invalid_count += 1
            conn.commit()
            logger.info(f"[SNAPSHOT VALIDATOR] Removed {invalid_count} broken paths")
            return invalid_count
    except Exception as e:
        logger.error(f"[SNAPSHOT VALIDATOR ERROR] {e}")
        return 0

def save_model_metadata(accuracy, method):
    """Store model training results"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO model_info (accuracy, method, updated_at) VALUES (%s, %s, %s)",
                           (accuracy, method, datetime.now()))
            conn.commit()
    except Exception as e:
        logger.error(f"[MODEL INFO SAVE ERROR] {e}")

def get_model_metadata():
    """Return latest two model metadata entries with change calculation"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            cursor.execute("SELECT * FROM model_info ORDER BY updated_at DESC LIMIT 2")
            rows = cursor.fetchall()
            if not rows:
                return None
            latest = rows[0]
            prev = rows[1] if len(rows) > 1 else None
            change = (latest['accuracy'] - prev['accuracy']) if prev else None
            return {
                "accuracy": latest['accuracy'],
                "updated_at": latest['updated_at'].strftime("%Y-%m-%d %H:%M:%S"),
                "method": latest['method'],
                "change": round(change, 2) if change is not None else None
            }
    except Exception as e:
        logger.error(f"[MODEL INFO LOAD ERROR] {e}")
        return None
    
def create_app():
    """Application factory pattern"""
    app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), "templates"))
    app.secret_key = os.getenv("SECRET_KEY", secrets.token_hex(32))
    app.permanent_session_lifetime = timedelta(minutes=30)

    global radar
    # Radar system pipeline
    radar = IWR6843Interface()
    def load_classifier():
        return ObjectClassifier()

    classifier = load_classifier()
    tracker = ObjectTracker(speed_limits_map=config.get("dynamic_speed_limits", {}))

    # Flask-Login
    login_manager = LoginManager()
    login_manager.login_view = "login"
    login_manager.login_message = "Please log in to access this page."
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id):
        return get_user_by_id(user_id)

    @app.context_processor
    def inject_globals():
        return {"now": datetime.now(), "is_admin": is_admin()}

    @app.errorhandler(404)
    def not_found(error):
        return render_template('errors.html', message="Page not found", error_code=404), 404

    @app.errorhandler(500)
    def internal_error(error):
        return render_template('errors.html', message="Internal server error", error_code=500), 500

    @app.errorhandler(413)
    def file_too_large(error):
        return render_template('errors.html', message="File too large", error_code=413), 413

    @app.errorhandler(Exception)
    def unhandled_exception(error):
        logger.error(f"[ERROR] Unhandled exception: {error}")
        logger.error(traceback.format_exc())
        return render_template('errors.html', message="Internal error", error_code=500), 500
    
    @app.route("/heatmap_feed")
    @login_required
    def heatmap_feed():
        def generate_heatmap_frame():
            with plotter.lock:
                fig = plotter.heatmap_fig or plotter.fig
                canvas = FigureCanvas(fig)
                buf = io.BytesIO()
                canvas.print_jpeg(buf)
                return buf.getvalue()

        def generate():
            while True:
                try:
                    frame = generate_heatmap_frame()
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
                    time.sleep(1.0)
                except Exception as e:
                    logger.error(f"[HEATMAP_FEED] Frame error: {e}")
                    time.sleep(1.0)

        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route("/camera_feed")
    @login_required
    def camera_feed():
        def generate_heatmap_frame():
            with plotter.lock:
                fig = plotter.heatmap_fig or plotter.fig  
                canvas = FigureCanvas(fig)
                buf = io.BytesIO()
                canvas.print_jpeg(buf)
                return buf.getvalue()

        def generate():
            try:
                config["cameras"], config["selected_camera"] = load_cameras_from_db()
                cam = config["cameras"][config["selected_camera"]] if config["cameras"] else {}

                camera_enabled = cam.get("enabled", True)
                stream_type = cam.get("stream_type", "snapshot").lower()
                username = cam.get("username")
                password = cam.get("password")
                auth = HTTPDigestAuth(username, password) if username and password else None

                # Determine stream URL
                url = cam.get("url")
                if not url:
                    ip = cam.get("ip")
                    if stream_type == "snapshot":
                        url = f"http://{ip}/axis-cgi/jpg/image.cgi"
                    elif stream_type == "mjpeg":
                        url = f"http://{ip}/mjpg/video.mjpg"
                    elif stream_type == "rtsp":
                        url = f"rtsp://{ip}/axis-media/media.amp"

                if not camera_enabled or not url:
                    logger.warning("[CAMERA_FEED] Falling back to 3D heatmap (camera disabled or missing)")
                    def heatmap_stream():
                        while True:
                            frame = generate_heatmap_frame()
                            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
                            time.sleep(1.0)

                    return Response(heatmap_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

                if stream_type == "rtsp":
                    # RTSP stream to MJPEG via FFmpeg
                    auth_str = f"{username}:{password}@" if username and password else ""
                    rtsp_url = url if "@" in url else url.replace("rtsp://", f"rtsp://{auth_str}")

                    ffmpeg_cmd = [
                        "ffmpeg", "-rtsp_transport", "tcp",
                        "-user_agent", "Mozilla/5.0",
                        "-i", rtsp_url,
                        "-f", "mjpeg", "-qscale:v", "2", "-r", "5", "-"
                    ]
                    process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                    buffer = b""
                    try:
                        while True:
                            chunk = process.stdout.read(4096)
                            if not chunk:
                                break
                            buffer += chunk
                            start = buffer.find(b'\xff\xd8')
                            end = buffer.find(b'\xff\xd9')
                            if start != -1 and end != -1 and end > start:
                                frame = buffer[start:end+2]
                                buffer = buffer[end+2:]
                                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
                    finally:
                        process.kill()

                elif stream_type == "snapshot":
                    while True:
                        try:
                            response = requests.get(url, auth=auth, timeout=5)
                            if response.status_code == 200 and response.content.startswith(b'\xff\xd8'):
                                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + response.content + b"\r\n")
                            else:
                                logger.warning(f"[CAMERA_FEED] Snapshot failed, status={response.status_code}")
                        except Exception as e:
                            logger.error(f"[CAMERA_FEED] Snapshot error: {e}")
                        time.sleep(0.5)

                else:
                    # MJPEG stream fallback
                    try:
                        with requests.get(url, auth=auth, stream=True, timeout=10) as r:
                            if r.status_code != 200:
                                logger.error(f"[CAMERA_FEED] MJPEG stream returned {r.status_code}")
                                return
                            buffer = b""
                            for chunk in r.iter_content(chunk_size=4096):
                                buffer += chunk
                                start = buffer.find(b'\xff\xd8')
                                end = buffer.find(b'\xff\xd9')
                                if start != -1 and end != -1 and end > start:
                                    frame = buffer[start:end+2]
                                    buffer = buffer[end+2:]
                                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
                    except Exception as e:
                        logger.error(f"[CAMERA_FEED] MJPEG stream error: {e}")
            except Exception as e:
                logger.exception(f"[CAMERA_FEED] Fatal error: {e}")
                time.sleep(2)

        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')  

    @app.route("/toggle_camera", methods=["POST"])
    @login_required
    def toggle_camera():
        if not is_admin():
            return jsonify({"error": "Unauthorized"}), 403

        try:
            config = load_config()
            selected = config.get("selected_camera", 0)
            cameras = config.get("cameras", [])
            if not cameras or selected >= len(cameras):
                return jsonify({"error": "Invalid camera config"}), 400

            new_status = not cameras[selected].get("enabled", True)
            cameras[selected]["enabled"] = new_status
            save_config(config)
            save_cameras_to_db(cameras, selected)

            return jsonify({
                "status": "ok",
                "enabled": cameras[selected]["enabled"]
            })
        except Exception as e:
            logger.error(f"[TOGGLE CAMERA ERROR] {e}")
            return jsonify({"error": "Internal error"}), 500                                                                                                               
    
    @app.route("/api/reload_config", methods=["POST"])
    @login_required
    def reload_config():
        if not is_admin():
            return jsonify({"error": "Unauthorized"}), 403

        try:
            # Write to a file flag or message queue
            with open("reload_flag.txt", "w") as f:
                f.write(str(time.time()))
            return jsonify({"status": "ok", "message": "Config reload requested."})
        except Exception as e:
            logger.error(f"[RELOAD API ERROR] {e}")
            return jsonify({"error": "Internal error"}), 500

    @app.route("/manual_snapshot", methods=["POST"])
    @login_required
    def manual_snapshot():
        if not is_admin():
            return jsonify({"error": "Unauthorized"}), 403

        try:
            now = time.time()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            config = load_config()
            selected = config.get("selected_camera", 0)
            cam = config.get("cameras", [{}])[selected] if isinstance(config.get("cameras"), list) else {}
            radar_frame = radar.get_latest_frame() or {}
            objects = radar_frame.get("objects", [])

            snapshot_url = cam.get("snapshot_url", cam.get("url"))
            username = cam.get("username")
            password = cam.get("password")
            auth = HTTPDigestAuth(username, password) if username and password else None
            response = requests.get(snapshot_url, auth=auth, timeout=5)

            if response.status_code != 200 or not response.content.startswith(b'\xff\xd8'):
                return jsonify({"error": "Snapshot capture failed"}), 500

            snapshot_path = os.path.join(SNAPSHOT_FOLDER, f"manual_{timestamp}.jpg")
            with open(snapshot_path, "wb") as f:
                f.write(response.content)

            label = f"MANUAL | {datetime.now().strftime('%H:%M:%S')}"
            conf_thresh = config.get("annotation_conf_threshold", 0.5)
            annotated_path, distance = annotate_speeding_object(
                image_path=snapshot_path,
                radar_distance=0.0,  # legacy param, ignored internally
                label=label,
                min_confidence=conf_thresh
            )

            if not annotated_path:
                return jsonify({"error": "Annotation failed"}), 500

            with get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                if not objects:
                    return jsonify({"error": "No radar object found"}), 400

                # Pick most relevant object
                radar_obj = objects[0]  # Or use your scoring logic

                cursor.execute("""
                    INSERT INTO radar_data (
                        timestamp, datetime, sensor, object_id, type, confidence, speed_kmh,
                        velocity, distance, direction, signal_level, doppler_frequency, snapshot_path,
                        x, y, z, range, azimuth, elevation, motion_state, snapshot_status,
                        velx, vely, velz, snr, noise,
                        reviewed, flagged, range_profile, noise_profile,
                        accx, accy, accz, snapshot_type
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s,
                            %s, %s, %s, %s)
                """, (
                    now,
                    datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S"),
                    "Manual",
                    f"manual_{uuid.uuid4().hex[:6]}",
                    radar_obj.get("type", "UNKNOWN"),
                    radar_obj.get("confidence", 0.0),
                    radar_obj.get("speed_kmh", 0.0),
                    radar_obj.get("velocity", 0.0),
                    radar_obj.get("distance", 0.0),
                    radar_obj.get("direction", "manual"),
                    radar_obj.get("signal_level", 0.0),
                    radar_obj.get("doppler_frequency", 0.0),
                    annotated_path,
                    radar_obj.get("x", 0.0),
                    radar_obj.get("y", 0.0),
                    radar_obj.get("z", 0.0),
                    radar_obj.get("range", 0.0),
                    radar_obj.get("azimuth", 0.0),
                    radar_obj.get("elevation", 0.0),
                    radar_obj.get("motion_state", "STATIC"),
                    "valid",
                    radar_obj.get("velx", 0.0),
                    radar_obj.get("vely", 0.0),
                    radar_obj.get("velz", 0.0),
                    radar_obj.get("snr", 0.0),
                    radar_obj.get("noise", 0.0),
                    0, 0,
                    radar_obj.get("range_profile", []),
                    radar_obj.get("noise_profile", []),
                    radar_obj.get("accx", 0.0),
                    radar_obj.get("accy", 0.0),
                    radar_obj.get("accz", 0.0),
                    "manual"
                ))
                conn.commit()

            return jsonify({"status": "ok", "message": "Snapshot captured successfully."})

        except Exception as e:
            logger.error(f"[MANUAL SNAPSHOT ERROR] {e}")
            return jsonify({"error": "Internal error"}), 500

    @app.route("/logs")
    @login_required
    def view_logs():
        try:
            log_path = os.path.join("system-logs", "radar.log")
            logs = []
            if os.path.isfile(log_path):
                with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                    logs = f.readlines()[-1000:]
            else:
                logs = ["[INFO] Log file not found: system-logs/radar.log"]

            return render_template("logs.html", logs=logs)

        except Exception as e:
            err_msg = f"Exception in /logs: {e}\n{traceback.format_exc()}"
            return f"<pre style='color:red;'>{err_msg}</pre>", 500


    @app.route("/api/logs")
    @login_required
    def api_logs():
        try:
            log_path = os.path.join("system-logs", "radar.log")
            offset = int(request.args.get("offset", 0))
            limit = int(request.args.get("limit", 100))
            max_lines = offset + limit

            if not os.path.exists(log_path):
                return jsonify({"logs": [], "has_more": False})

            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                all_lines = deque(f, maxlen=max_lines)

            logs = list(all_lines)
            paginated = logs[-limit:]

            return jsonify({
                "logs": [line.strip() for line in paginated],
                "has_more": len(all_lines) >= max_lines
            })

        except Exception as e:
            logger.exception(f"[LOGS API ERROR] {e}")
            return jsonify({"error": "Internal server error"}), 500


    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            username = request.form["username"].strip()
            password = request.form["password"]
            user = get_user_by_username(username)
            if user and check_password_hash(user.password_hash, password):
                login_user(user)
                return redirect(url_for("index"))
            return render_template("login.html", error="Invalid credentials")
        return render_template("login.html")


    @app.route("/logout", methods=["POST"])
    @login_required
    def logout():
        logout_user()
        flash("You have been logged out successfully", "success")
        return redirect(url_for("login"))
    
    @app.route("/")
    @login_required
    def index():
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

                # Get recent detections 
                cursor.execute("""
                    SELECT datetime, type, speed_kmh, distance, direction, motion_state, snapshot_type,
                           snapshot_path, object_id, confidence
                    FROM radar_data 
                    WHERE snapshot_path IS NOT NULL
                    ORDER BY datetime DESC
                    LIMIT 10
                """)
                rows = cursor.fetchall()

                # Get summary statistics
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN LOWER(COALESCE(type, '')) LIKE '%human%' OR LOWER(COALESCE(type, '')) LIKE '%person%' THEN 1 ELSE 0 END) as humans,
                        SUM(CASE WHEN LOWER(COALESCE(type, '')) LIKE '%vehicle%' OR LOWER(COALESCE(type, '')) LIKE '%car%' OR LOWER(COALESCE(type, '')) LIKE '%truck%' OR LOWER(COALESCE(type, '')) LIKE '%bike%' THEN 1 ELSE 0 END) as vehicles,
                        SUM(CASE WHEN snapshot_type = 'manual' THEN 1 ELSE 0 END) as manual_snaps,
                        AVG(CASE WHEN speed_kmh IS NOT NULL AND speed_kmh >= 0 THEN speed_kmh END) as avg_speed,
                        MAX(speed_kmh) as top_speed,
                        MAX(datetime) as last_detection
                    FROM radar_data
                    WHERE snapshot_path IS NOT NULL AND snapshot_path != '';
                """)
                stats = cursor.fetchone()

                total, humans, vehicles, avg_speed, last_detection = (
                    stats['total'] if stats else 0,
                    stats['humans'] if stats else 0,
                    stats['vehicles'] if stats else 0,
                    stats['avg_speed'] if stats else 0,
                    stats['last_detection'] if stats else None
                )

                # Build snapshot card data
                snapshots = []
                for r in rows:
                    snapshot_data = {
                        "datetime": r['datetime'] or "N/A",
                        "type": r['type'] or "UNKNOWN",
                        "speed": round(float(r['speed_kmh']) if r['speed_kmh'] is not None else 0, 2),
                        "distance": round(float(r['distance']) if r['distance'] is not None else 0, 2),
                        "direction": r['direction'] or "N/A",
                        "image": os.path.basename(r['snapshot_path']) if r['snapshot_path'] else None,
                        "object_id": r['object_id'] or "N/A",
                        "confidence": round(float(r['confidence']) if r['confidence'] is not None else 0, 2),
                        "snapshot_type": r.get('snapshot_type') or "auto"
                    }
                    try:
                        label = f"{snapshot_data['type']} | {snapshot_data['speed']} km/h | {snapshot_data['distance']} m | {snapshot_data['direction']}"
                    except Exception as e:
                        logger.warning(f"Snapshot label formatting failed: {e}")
                        label = "UNKNOWN"
                    snapshot_data["label"] = label
                    snapshots.append(snapshot_data)

            # Load logs
            log_path = os.path.join("system-logs", "radar.log")
            logs = []
            if os.path.isfile(log_path):
                with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                    logs = f.readlines()[-15:]
            else:
                logs = ["[INFO] Log file not found: system-logs/radar.log"]

            # Read Pi temperature
            try:
                with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                    temp_c = int(f.read()) / 1000.0
            except Exception:
                temp_c = None

            # Build summary
            summary = {
                "total": total or 0,
                "humans": humans or 0,
                "vehicles": vehicles or 0,
                "average_speed": round(avg_speed, 2) if avg_speed else 0,
                "last_detection": (
                    last_detection.strftime("%Y-%m-%d %H:%M:%S")
                    if isinstance(last_detection, datetime) else str(last_detection or "N/A")
                ),
                "logs": logs,
                "pi_temperature": round(temp_c, 1) if temp_c is not None else "N/A",
                "cpu_load": round(os.getloadavg()[0], 2)
            }

            return render_template("index.html", snapshots=snapshots, summary=summary, config=load_config())

        except Exception as e:
            logger.error(f"[INDEX ROUTE ERROR] {e}")
            flash("Error loading dashboard data", "error")
            summary = {
                "total": 0,
                "humans": 0,
                "vehicles": 0,
                "average_speed": 0,
                "last_detection": "N/A",
                "logs": [],
                "pi_temperature": "N/A",
                "cpu_load": 0.0
            }
            return render_template("index.html", snapshots=[], summary=summary, config=load_config())
        
    @app.route("/api/status")
    @login_required
    def api_status():
        return jsonify({
            "status": "ok",
            "timestamp": datetime.now().isoformat()
        })

    @app.route("/api/charts")
    def api_charts():
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

                try:
                    days = int(request.args.get("days", 30))
                except ValueError:
                    days = 30

                date_filter = (
                    "DATE(datetime::TIMESTAMP) = CURRENT_DATE" if days <= 0 else
                    "CAST(datetime AS TIMESTAMP) >= CURRENT_DATE - INTERVAL %s DAY"
                )
                date_param = () if days <= 0 else (str(days),)

                # --- Speed histogram ---
                cursor.execute(f"""
                    SELECT speed_kmh 
                    FROM radar_data 
                    WHERE speed_kmh IS NOT NULL AND speed_kmh BETWEEN 0 AND 200
                    AND {date_filter}
                """, date_param)
                speeds = [float(row["speed_kmh"]) for row in cursor.fetchall() if row["speed_kmh"] is not None]

                speed_bins = list(range(0, 101, 10))
                speed_labels = [f"{i}-{i+9}" for i in speed_bins[:-1]] + ["100+"]
                speed_counts = [0] * len(speed_labels)

                for speed in speeds:
                    index = int(speed // 10) if speed < 100 else -1
                    speed_counts[index] += 1

                # Define expected labels
                direction_labels = ["Approaching", "Departing", "Stationary", "Left", "Right", "Unknown"]

                # Normalize actual DB values to known labels
                cursor.execute(f"""
                    SELECT direction 
                    FROM radar_data 
                    WHERE direction IS NOT NULL AND TRIM(direction) != ''
                    AND {date_filter}
                """, date_param)

                raw_directions = [
                    str(row["direction"]).strip().lower()
                    for row in cursor.fetchall()
                    if row["direction"] is not None
                ]
                normalized = []
                for d in raw_directions:
                    if d in ["towards", "approaching"]:
                        normalized.append("Approaching")
                    elif d in ["away", "departing"]:
                        normalized.append("Departing")
                    elif d in ["static", "stationary"]:
                        normalized.append("Stationary")
                    elif d in ["left"]:
                        normalized.append("Left")
                    elif d in ["right"]:
                        normalized.append("Right")
                    else:
                        normalized.append("Unknown")

                direction_data = [normalized.count(label) for label in direction_labels]

                # --- Violations per hour ---
                cursor.execute(f"""
                    SELECT TO_CHAR(datetime::TIMESTAMP, 'HH24') as hour, COUNT(*) as count
                    FROM radar_data 
                    WHERE speed_kmh > 0 AND {date_filter}
                    GROUP BY hour
                    ORDER BY hour
                """, date_param)
                hourly_rows = cursor.fetchall()
                hour_labels = [f"{int(r['hour']):02d}:00" for r in hourly_rows]
                hour_data = [r['count'] for r in hourly_rows]

                return jsonify({
                    "speed_histogram": {
                        "labels": speed_labels,
                        "data": speed_counts
                    },
                    "direction_breakdown": {
                        "labels": direction_labels,
                        "data": direction_data
                    },
                    "violations_per_hour": {
                        "labels": hour_labels,
                        "data": hour_data
                    }
                })

        except psycopg2.Error as e:
            logger.exception("[API CHARTS] PostgreSQL error")
            return jsonify({"error": "Database error"}), 500
        except Exception as e:
            logger.exception("[API CHARTS] Unhandled error")
            return jsonify({"error": "Internal server error"}), 500
    
    @app.route("/gallery")
    @login_required
    def gallery():
        obj_type = request.args.get("type", "").upper()
        min_speed = float(request.args.get("min_speed") or 0)
        max_speed = float(request.args.get("max_speed") or 999)
        direction = request.args.get("direction", "").lower()
        motion_state = request.args.get("motion_state", "").lower()
        snapshot_type = request.args.get("snapshot_type", "").lower()
        object_id = request.args.get("object_id", "")
        start_date = request.args.get("start_date", "")
        end_date = request.args.get("end_date", "")
        min_confidence = float(request.args.get("min_confidence") or 0)
        max_confidence = float(request.args.get("max_confidence") or 1)
        reviewed_only = request.args.get("reviewed_only") == "1"
        flagged_only = request.args.get("flagged_only") == "1"
        unannotated_only = request.args.get("unannotated_only") == "1"
        download = request.args.get("download", "0") == "1"
        page = int(request.args.get("page", 1))
        limit = int(request.args.get("limit", 100))

        try:
            with get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

                # 1. Build Filter Query
                filters = ["snapshot_path IS NOT NULL"]
                params = []

                if min_speed > 0 or max_speed < 999:
                    filters.append("speed_kmh BETWEEN %s AND %s")
                    params.extend([min_speed, max_speed])
                if obj_type:
                    filters.append("UPPER(COALESCE(type, '')) LIKE %s")
                    params.append(f"%{obj_type}%")
                if direction:
                    filters.append("LOWER(COALESCE(direction, '')) LIKE %s")
                    params.append(f"%{direction}%")
                if motion_state:
                    filters.append("LOWER(COALESCE(motion_state, '')) LIKE %s")
                    params.append(f"%{motion_state}%")
                if snapshot_type:
                    filters.append("LOWER(COALESCE(snapshot_type, '')) = %s")
                    params.append(snapshot_type)
                if object_id:
                    filters.append("COALESCE(object_id, '') LIKE %s")
                    params.append(f"%{object_id}%")
                if start_date:
                    filters.append("DATE(COALESCE(datetime::TIMESTAMP, to_timestamp(timestamp))) >= %s")
                    params.append(start_date)
                if end_date:
                    filters.append("DATE(COALESCE(datetime::TIMESTAMP, to_timestamp(timestamp))) <= %s")
                    params.append(end_date)
                if min_confidence > 0 or max_confidence < 1:
                    filters.append("confidence BETWEEN %s AND %s")
                    params.extend([min_confidence, max_confidence])
                if reviewed_only:
                    filters.append("reviewed = 1")
                if flagged_only:
                    filters.append("flagged = 1")
                if unannotated_only:
                    filters.append("reviewed = 0 AND flagged = 0")

                where_clause = " AND ".join(filters)

                # 2. Count total
                cursor.execute(f"SELECT COUNT(*) FROM radar_data WHERE {where_clause}", params)
                total_items = cursor.fetchone()[0]

                # 3. Pagination
                offset, _, total_pages, current_page = apply_pagination(total_items, page, limit)

                # 4. Query Data
                query = f"""
                    SELECT datetime, type, speed_kmh, distance, direction, motion_state,
                        snapshot_path, object_id, confidence, reviewed, flagged,
                        snapshot_status, snapshot_type
                    FROM radar_data
                    WHERE {where_clause}
                    ORDER BY COALESCE(datetime::TIMESTAMP, to_timestamp(timestamp)) DESC
                    LIMIT %s OFFSET %s
                """
                cursor.execute(query, params + [limit, offset])
                rows = cursor.fetchall()

                # 5. Format Results
                snapshots = []
                for r in rows:
                    snapshot_data = {
                        "filename": os.path.basename(r["snapshot_path"]) if r["snapshot_path"] else "no_image.jpg",
                        "datetime": r["datetime"] or "N/A",
                        "type": r["type"] or "UNKNOWN",
                        "speed": round(float(r["speed_kmh"] or 0), 2),
                        "distance": round(float(r["distance"] or 0), 2),
                        "direction": r["direction"] or "N/A",
                        "motion_state": r["motion_state"] or "N/A",
                        "object_id": r["object_id"] or "N/A",
                        "confidence": round(float(r["confidence"] or 0), 2),
                        "reviewed": bool(r["reviewed"]),
                        "flagged": bool(r["flagged"]),
                        "snapshot_status": r["snapshot_status"] or "valid",
                        "snapshot_type": r["snapshot_type"] or "auto",
                        "path": r["snapshot_path"] if r["snapshot_path"] and os.path.exists(r["snapshot_path"]) else None
                    }
                    snapshots.append(snapshot_data)

                # 6. Download ZIP
                if download and snapshots:
                    buffer = BytesIO()
                    with zipfile.ZipFile(buffer, 'w') as zipf:
                        for snap in snapshots:
                            path = snap.get("path")
                            if path and os.path.isfile(path):
                                try:
                                    zipf.write(path, arcname=snap["filename"])
                                except Exception as e:
                                    logger.warning(f"Failed to add {snap['filename']} to zip: {e}")
                    if not zipf.namelist():
                        return jsonify({"error": "No valid files"}), 400
                    buffer.seek(0)
                    return send_file(buffer, mimetype='application/zip', as_attachment=True,
                                    download_name="filtered_snapshots.zip")

                # 7. Render page
                return render_template("gallery.html",
                                    snapshots=snapshots,
                                    current_page=current_page,
                                    total_pages=total_pages)

        except Exception as e:
            logger.error(f"[GALLERY ERROR] {e}")
            logger.error(traceback.format_exc())
            flash("Error loading gallery data", "error")
            logger.info(f"[GALLERY DEBUG] Rendering {len(snapshots)} snapshots, current_page={current_page}, total_pages={total_pages}")
            for i, s in enumerate(snapshots[:3]):
                logger.info(f"[SNAPSHOT {i}] {s}")

            return render_template("gallery.html", snapshots=snapshots or [], current_page=current_page or 1, total_pages=total_pages or 1)
        
    @app.route("/mark_snapshot", methods=["POST"])
    @login_required
    def mark_snapshot():
        try:
            data = request.get_json()
            snapshot = data.get("snapshot")
            action = data.get("action") 

            if not snapshot or action not in ("reviewed", "flagged"):
                return jsonify({"error": "Invalid input"}), 400

            with get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                cursor.execute(f"SELECT {action} FROM radar_data WHERE snapshot_path LIKE %s", (f"%{snapshot}",))
                current = cursor.fetchone()
                new_value = 0 if current and current[action] == 1 else 1

                cursor.execute(f"""
                    UPDATE radar_data SET {action} = %s 
                    WHERE snapshot_path LIKE %s
                """, (new_value, f"%{snapshot}",))
                conn.commit()

            return jsonify({"status": "updated", "new_value": new_value})
        
        except Exception as e:
            logger.error(f"Error marking snapshot: {e}")
            return jsonify({"error": "Internal server error"}), 500
    
    @app.route("/snapshots/<filename>")
    @login_required
    def serve_snapshot(filename):
        try:
            return send_from_directory(SNAPSHOT_FOLDER, filename)
        except FileNotFoundError:
            return "File not found", 404

    @app.route("/export_pdf")
    @login_required
    def export_pdf():
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                cursor.execute("""
                    SELECT datetime, sensor, object_id, type, confidence, speed_kmh, velocity, distance,
                        direction, motion_state, signal_level, doppler_frequency, reviewed, flagged, snapshot_path, snapshot_type
                    FROM radar_data
                    WHERE snapshot_path IS NOT NULL
                    ORDER BY datetime DESC
                    LIMIT 100
                """)
                rows = cursor.fetchall()

            data = [dict(row) for row in rows]
            speeds = [float(d["speed_kmh"]) for d in data if d.get("speed_kmh") is not None]
            types = [d["type"].upper() for d in data if d.get("type")]
            directions = [d["direction"].lower() for d in data if d.get("direction")]
            motion = [d["motion_state"].lower() for d in data if d.get("motion_state")]
            snapshot_types = [d["snapshot_type"] for d in data if d.get("snapshot_type")]

            summary = {
                "total_records": len(data),
                "manual_snapshots": snapshot_types.count("manual"),
                "auto_snapshots": snapshot_types.count("auto"),
                "avg_speed": round(sum(speeds)/len(speeds), 2) if speeds else 0.0,
                "top_speed": max(speeds) if speeds else 0.0,
                "lowest_speed": min(speeds) if speeds else 0.0,
                "most_detected_object": Counter(types).most_common(1)[0][0] if types else "N/A",
                "approaching_count": sum(1 for d in directions if d == "approaching"),
                "stationary_count": sum(1 for d in directions if d == "stationary"),
                "departing_count": sum(1 for d in directions if d == "departing"),
                "last_detection": data[0].get("datetime") if data else "N/A",
                "speed_limits": load_config().get("dynamic_speed_limits", {})
            }

            try:
                response = requests.get("http://127.0.0.1:5000/api/charts")
                charts = response.json() if response.ok else {}
            except Exception as e:
                logger.warning(f"[CHART FETCH ERROR] {e}")
                charts = {}

            logo_path = "/home/pi/radar/static/essi_logo.jpeg"
            filename = f"radar_full_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            filepath = os.path.join("backups", filename)
            os.makedirs("backups", exist_ok=True)

            generate_pdf_report(filepath, data=data, summary=summary, logo_path=logo_path, charts=charts)
            return send_file(filepath, as_attachment=True)

        except Exception as e:
            logger.error(f"[EXPORT_PDF_ERROR] {e}")
            return str(e), 500

    @app.route("/export_filtered_pdf")
    @login_required
    def export_filtered_pdf():
        try:
            params = request.args.to_dict()
            filters = params.copy()

            query = """
                SELECT datetime, sensor, object_id, type, confidence, speed_kmh, velocity, distance,
                    direction, signal_level, doppler_frequency, reviewed, flagged, snapshot_path, snapshot_type
                FROM radar_data
                WHERE snapshot_path IS NOT NULL
            """
            sql_params = []

            # Filter: Object Type
            if 'type' in params and params['type']:
                query += " AND UPPER(COALESCE(type, '')) LIKE %s"
                sql_params.append(f"%{params['type'].upper()}%")

            # Filter: Speed Range
            if 'min_speed' in params:
                try:
                    query += " AND speed_kmh >= %s"
                    sql_params.append(float(params['min_speed']))
                except ValueError:
                    pass
            if 'max_speed' in params:
                try:
                    query += " AND speed_kmh <= %s"
                    sql_params.append(float(params['max_speed']))
                except ValueError:
                    pass

            # Filter: Direction
            if 'direction' in params and params['direction']:
                query += " AND LOWER(COALESCE(direction, '')) = %s"
                sql_params.append(params['direction'].lower())

            # Filter: Object ID
            if 'object_id' in params and params['object_id']:
                query += " AND COALESCE(object_id, '') LIKE %s"
                sql_params.append(f"%{params['object_id']}%")

            # Filter: Dates
            if 'start_date' in params and params['start_date']:
                query += " AND DATE(COALESCE(datetime::TIMESTAMP, to_timestamp(timestamp))) >= %s"
                sql_params.append(params['start_date'])
            if 'end_date' in params and params['end_date']:
                query += " AND DATE(COALESCE(datetime::TIMESTAMP, to_timestamp(timestamp))) <= %s"
                sql_params.append(params['end_date'])

            # Filter: Confidence
            if 'min_confidence' in params:
                try:
                    query += " AND confidence >= %s"
                    sql_params.append(float(params['min_confidence']))
                except ValueError:
                    pass
            if 'max_confidence' in params:
                try:
                    query += " AND confidence <= %s"
                    sql_params.append(float(params['max_confidence']))
                except ValueError:
                    pass

            # Snapshot type filter
            if 'snapshot_type' in params and params['snapshot_type'] in ['manual', 'auto']:
                query += " AND snapshot_type = %s"
                sql_params.append(params['snapshot_type'])

            # Flags
            if params.get("reviewed_only") == "1":
                query += " AND reviewed = 1"
            if params.get("flagged_only") == "1":
                query += " AND flagged = 1"
            if params.get("unannotated_only") == "1":
                query += " AND reviewed = 0 AND flagged = 0"

            query += " ORDER BY datetime DESC LIMIT 1000"

            with get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                cursor.execute(query, sql_params)
                rows = cursor.fetchall()

            data = [dict(row) for row in rows]
            speeds = [float(d.get("speed_kmh") or 0) for d in data if d.get("speed_kmh") is not None]
            types = [d["type"].upper() for d in data if d.get("type")]
            directions = [d["direction"].lower() for d in data if d.get("direction")]
            snapshot_types = [d["snapshot_type"] for d in data if d.get("snapshot_type")]

            summary = {
                "total_records": len(data),
                "manual_snapshots": snapshot_types.count("manual"),
                "auto_snapshots": snapshot_types.count("auto"),
                "avg_speed": round(sum(speeds) / len(speeds), 2) if speeds else 0.0,
                "top_speed": max(speeds) if speeds else 0.0,
                "lowest_speed": min(speeds) if speeds else 0.0,
                "most_detected_object": Counter(types).most_common(1)[0][0] if types else "N/A",
                "approaching_count": sum(1 for d in directions if d == "approaching"),
                "stationary_count": sum(1 for d in directions if d == "stationary"),
                "departing_count": sum(1 for d in directions if d == "departing"),
                "last_detection": data[0].get("datetime") if data else "N/A",
                "speed_limits": load_config().get("dynamic_speed_limits", {})
            }

            charts = {}
            try:
                resp = requests.get("http://127.0.0.1:5000/api/charts?days=0")
                if resp.ok:
                    charts = resp.json()
            except Exception as e:
                logger.warning(f"[Chart Fetch Error] {e}")

            logo_path = "/home/pi/radar/static/essi_logo.jpeg"
            filename = f"radar_filtered_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            filepath = os.path.join("backups", filename)
            os.makedirs("backups", exist_ok=True)

            generate_pdf_report(filepath, data=data, summary=summary, filters=filters, logo_path=logo_path, charts=charts)
            return send_file(filepath, as_attachment=True)

        except Exception as e:
            logger.error(f"[EXPORT_FILTERED_PDF_ERROR] {e}")
            return str(e), 500
    
    @app.route("/retrain_model", methods=["POST"])
    @login_required
    def retrain_model():
        if not is_admin():
            return jsonify({"error": "Unauthorized"}), 403

        try:
            logger.info("[MODEL] Retraining LightGBM model from DB...")
            result = subprocess.run(["python3", "train_lightbgm.py"], capture_output=True, text=True)

            if result.returncode == 0:
                logger.info("[MODEL] Retraining completed successfully.")
                acc = None
                for line in result.stdout.splitlines():
                    if "ACCURACY:" in line:
                        try:
                            acc = float(line.strip().split("ACCURACY:")[1].strip())
                            save_model_metadata(acc, "retrain")
                            break
                        except Exception as e:
                            logger.warning(f"Could not parse accuracy: {e}")
                return jsonify({"status": "ok", "message": "Model retrained successfully."})
            else:
                logger.error(f"[MODEL] Retrain failed: {result.stderr}")
                return jsonify({"error": "Retraining failed", "details": result.stderr}), 500

        except Exception as e:
            logger.exception("[RETRAIN ERROR]")
            return jsonify({"error": "Internal server error."}), 500


    @app.route("/upload_model", methods=["POST"])
    @login_required
    def upload_model():
        if not is_admin():
            return jsonify({"error": "Unauthorized"}), 403

        if "model_file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["model_file"]

        if not file.filename.endswith(".pkl"):
            return jsonify({"error": "File must be a .pkl"}), 400

        try:
            from tempfile import NamedTemporaryFile
            import joblib

            with NamedTemporaryFile(delete=False) as tmp:
                file.save(tmp.name)
                loaded = joblib.load(tmp.name)

            if not isinstance(loaded, tuple) or len(loaded) != 2:
                return jsonify({"error": "Model format invalid. Expected (model, scaler) tuple."}), 400

            model, scaler = loaded

            if not isinstance(model, lgb.LGBMClassifier) or not isinstance(scaler, StandardScaler):
                return jsonify({"error": "Incorrect model or scaler type."}), 400

            joblib.dump((model, scaler), "radar_lightgbm_model.pkl")
            save_model_metadata(-1, "upload")  # score unavailable; use -1 as placeholder
            return jsonify({"status": "ok", "message": "Model uploaded and validated successfully."})

        except Exception as e:
            logger.exception("[MODEL UPLOAD ERROR]")
            return jsonify({"error": "Upload failed", "details": str(e)}), 500
    
    @app.route("/control", methods=["GET", "POST"])
    @login_required
    def control():
        if not is_admin():
            flash("Admin access required", "error")
            return redirect(url_for("index"))
        
        message = None
        config = load_config()
        try:
            cameras, selected = load_cameras_from_db()
            for cam in cameras:
                if "stream_type" not in cam:
                    cam["stream_type"] = "mjpeg"  # default fallback
            config["cameras"] = cameras
            config["selected_camera"] = selected
        except Exception as e:
            logger.warning(f"Could not load cameras from DB: {e}")
            config["cameras"] = []
            config["selected_camera"] = 0
        snapshot = None
        
        if request.method == "POST":
            action = request.form.get("action")
            
            try:
                if action == "clear_db":
                    with get_db_connection() as conn:
                        conn.execute("DELETE FROM radar_data")
                        conn.commit()
                    message = "All radar data cleared successfully."
                    
                elif action == "backup_db":
                    try:
                        backup_name = f"radar_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sql"
                        backup_path = os.path.join(BACKUP_FOLDER, backup_name)
                        os.makedirs(BACKUP_FOLDER, exist_ok=True)

                        result = subprocess.run(
                            ["pg_dump", "-U", "radar_user", "-h", "localhost", "-d", "iwr6843_db", "-f", backup_path],
                            env={**os.environ, "PGPASSWORD": "securepass123"},
                            check=True
                        )

                        return send_file(backup_path, as_attachment=True, download_name=backup_name)

                    except Exception as e:
                        logger.error(f"[BACKUP ERROR] {e}")
                        message = f"Database backup failed: {str(e)}"
                    
                elif action == "restore_db":
                    if 'backup_file' in request.files:
                        file = request.files['backup_file']
                        if file and file.filename.endswith('.sql'):
                            filename = secure_filename(file.filename)
                            temp_path = os.path.join(BACKUP_FOLDER, f"temp_{filename}")
                            file.save(temp_path)

                            try:
                                subprocess.run(
                                    ["psql", "-U", "radar_user", "-h", "localhost", "-d", "iwr6843_db", "-f", temp_path],
                                    env={**os.environ, "PGPASSWORD": "securepass123"},
                                    check=True
                                )
                                os.remove(temp_path)
                                message = "Database restored successfully."
                            except subprocess.CalledProcessError as e:
                                message = f"Restore failed: {e}"
                                logger.error(f"[RESTORE ERROR] {e}")
                        else:
                            message = "Please upload a valid .sql file."
                            
                elif action == "cleanup_snapshots":
                    retention_days = int(request.form.get("retention_days", config.get("retention_days", 30)))
                    cutoff = datetime.now() - timedelta(days=retention_days)
                    
                    with get_db_connection() as conn:
                        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                        cursor.execute("SELECT snapshot_path FROM radar_data WHERE COALESCE(datetime::TIMESTAMP, to_timestamp(timestamp)) < %s", 
                                     (cutoff.strftime("%Y-%m-%d %H:%M:%S"),))
                        old_paths = [row['snapshot_path'] for row in cursor.fetchall() if row['snapshot_path']]
                        
                        deleted_count = 0
                        for path in old_paths:
                            if os.path.exists(path):
                                try:
                                    os.remove(path)
                                    deleted_count += 1
                                except Exception as e:
                                    logger.warning(f"Failed to delete {path}: {e}")
                        
                        cursor.execute("DELETE FROM radar_data WHERE COALESCE(datetime::TIMESTAMP, to_timestamp(timestamp)) < %s", 
                                     (cutoff.strftime("%Y-%m-%d %H:%M:%S"),))
                        deleted_records = cursor.rowcount
                        conn.commit()
                    
                    message = f"Cleaned up {deleted_count} snapshots and {deleted_records} records."

                elif action == "test_radar":
                    try:
                        result_1 = check_radar_connection(port="/dev/serial/by-id/usb-Silicon_Labs_CP2105_Dual_USB_to_UART_Bridge_Controller_0151A9D6-if00-port0", baudrate=115200)
                        result_2 = check_radar_connection(port="/dev/serial/by-id/usb-Silicon_Labs_CP2105_Dual_USB_to_UART_Bridge_Controller_0151A9D6-if01-port0", baudrate=921600)

                        if result_1 and result_2:
                            message = "Radar test successful. Both UART ports are active."
                            result_1.close()
                            result_2.close()
                        elif result_1 or result_2:
                            message = "Partial radar test passed. One UART port is responding."
                            if result_1: result_1.close()
                            if result_2: result_2.close()
                        else:
                            message = "Radar test failed. Both UART ports unresponsive."

                    except Exception as e:
                        logger.error(f"[RADAR TEST] {e}")
                        message = f"Radar test error: {e}"
                    
                elif action == "test_camera":
                    try:
                        selected = config.get("selected_camera", 0)
                        cam = config.get("cameras", [{}])[selected] if isinstance(config.get("cameras"), list) else {}
                        url = cam.get("url", "")
                        username = cam.get("username", "")
                        password = cam.get("password", "")
                        stream_type = cam.get("stream_type", "mjpeg")
                        auth = HTTPDigestAuth(username, password) if username and password else None

                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        test_filename = f"test_{timestamp}.jpg"
                        test_path = os.path.join(SNAPSHOT_FOLDER, test_filename)
                        response = None

                        if stream_type == "snapshot":
                            response = requests.get(url, auth=auth, timeout=5)
                            if response.status_code == 200 and response.content.startswith(b'\xff\xd8'):
                                with open(test_path, "wb") as f:
                                    f.write(response.content)

                        elif stream_type == "mjpeg":
                            r = requests.get(url, auth=auth, stream=True, timeout=5)
                            buffer = b""
                            for chunk in r.iter_content(1024):
                                buffer += chunk
                                start = buffer.find(b'\xff\xd8')
                                end = buffer.find(b'\xff\xd9')
                                if start != -1 and end != -1 and end > start:
                                    frame = buffer[start:end+2]
                                    with open(test_path, "wb") as f:
                                        f.write(frame)
                                    break
                            r.close()

                        elif stream_type == "rtsp":
                            result = subprocess.run([
                                "ffmpeg", "-y", "-rtsp_transport", "tcp",
                                "-i", url,
                                "-vframes", "1", "-q:v", "2", test_path
                            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=8)

                        if os.path.exists(test_path) and os.path.getsize(test_path) > 1024:
                            snapshot = os.path.basename(test_path)
                            message = "Camera test successful. Snapshot captured."
                        else:
                            message = "Camera test failed — no image returned."

                    except Exception as e:
                        logger.error(f"[CAMERA TEST] {e}")
                        message = f"Camera test error: {e}"
                    
                elif action == "update_config":
                    config["cooldown_seconds"] = float(request.form.get("cooldown_seconds", 0.5))
                    config["retention_days"] = int(request.form.get("retention_days", 30))
                    config["selected_camera"] = int(request.form.get("selected_camera", 0))
                    config["annotation_conf_threshold"] = float(request.form.get("annotation_conf_threshold", 0.5))
                    config["label_format"] = request.form.get("label_format", "{type} | {speed:.1f} km/h")

                    # Parse all camera fields
                    cameras = []
                    i = 0
                    while True:
                        cam_url = request.form.get(f"camera_url_{i}")
                        if not cam_url:
                            break
                        cam_username = request.form.get(f"camera_username_{i}", "")
                        cam_password = request.form.get(f"camera_password_{i}", "")
                        cam_type = request.form.get(f"camera_stream_type_{i}", "mjpeg")
                        cameras.append({
                            "url": cam_url.strip(),
                            "username": cam_username.strip(),
                            "password": cam_password.strip(),
                            "stream_type": cam_type.strip()
                        })
                        i += 1

                    if cameras:
                        config["cameras"] = cameras
                        save_cameras_to_db(cameras, config.get("selected_camera", 0))

                    # Dynamic speed limits
                    updated_limits = {}
                    for key in config.get("dynamic_speed_limits", {}).keys():
                        form_key = f"speed_limit_{key}"
                        val = request.form.get(form_key)
                        if val:
                            try:
                                updated_limits[key] = float(val)
                            except ValueError:
                                pass  # retain old

                    if updated_limits:
                        config["dynamic_speed_limits"] = updated_limits

                    if save_config(config):
                        message = "Configuration updated successfully."
                    else:
                        message = "Failed to save configuration."
                        
                elif action == "validate_snapshots":
                    invalid_count = validate_snapshots()
                    message = f"Snapshot validation complete. {invalid_count} invalid paths cleaned."
                        
            except Exception as e:
                logger.error(f"Control action error: {e}")
                message = f"Action failed: {str(e)}"
        
        # Get system stats
        try:
            disk_usage = psutil.disk_usage('/')
            disk_free = disk_usage.free / (1024**3)
        except Exception:
            disk_free = 0

        try:
            with get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                cursor.execute("SELECT COUNT(*) FROM radar_data")
                total_records = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM radar_data WHERE snapshot_path IS NOT NULL")
                snapshot_records = cursor.fetchone()[0]
        except Exception:
            total_records = 0
            snapshot_records = 0

        try:
            radar_ok = isinstance(radar.get_targets(), list)
        except Exception as e:
            logger.warning(f"[RADAR TEST] Interface error: {e}")
            radar_ok = False
        except Exception:
            radar_ok = False

        cams = config.get("cameras", [])
        selected = config.get("selected_camera", 0)
        cam = cams[selected] if cams and selected < len(cams) else {}
        stream_type = cam.get("stream_type", "mjpeg")
        camera_ok = False

        should_check_camera = action != "test_camera" if request.method == "POST" else True

        if should_check_camera:
            try:
                url = cam.get("url", "")
                username = cam.get("username", "")
                password = cam.get("password", "")
                if stream_type == "rtsp":
                    if url.startswith("rtsp://") and "@" not in url and username and password:
                        url = url.replace("rtsp://", f"rtsp://{username}:{password}@")

                    logger.info(f"[CONTROL CAMERA TEST] RTSP URL: {url}")
                    result = subprocess.run(
                        ["ffmpeg", "-rtsp_transport", "tcp", "-i", url, "-t", "1", "-f", "null", "-"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=5
                    )
                    camera_ok = (result.returncode == 0)

                elif stream_type == "mjpeg":
                    r = requests.get(url, auth=HTTPDigestAuth(username, password), stream=True, timeout=5)
                    if r.status_code == 200:
                        buffer = b""
                        for chunk in r.iter_content(1024):
                            buffer += chunk
                            if b'\xff\xd8' in buffer and b'\xff\xd9' in buffer:
                                camera_ok = True
                                break
                    r.close()

                elif stream_type == "snapshot":
                    r = requests.get(url, auth=HTTPDigestAuth(username, password), timeout=5)
                    if r.status_code == 200 and r.content.startswith(b'\xff\xd8'):
                        camera_ok = True

                logger.info(f"[CONTROL CAMERA TEST RESULT] camera_ok = {camera_ok}")

            except Exception as e:
                logger.warning(f"[CONTROL CAMERA TEST] Unexpected failure: {e}")

        try:
            return render_template("control.html",
                message=message,
                config=config,
                disk_free=round(disk_free, 2),
                total_records=total_records,
                snapshot_records=snapshot_records,
                snapshot=snapshot,
                radar_status=radar_ok,
                camera_status=camera_ok,
                model_info=get_model_metadata()
            )
        except Exception as e:
            import traceback
            logger.error(f"[CONTROL PAGE ERROR] {e}\n{traceback.format_exc()}")
            return f"<pre>{traceback.format_exc()}</pre>", 500
    
    @app.route("/users", methods=["GET", "POST"])
    @login_required
    def users():
        if request.method == "POST":
            action = request.form.get("action")
            try:
                with get_db_connection() as conn:
                    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

                    if action == "add_user":
                        if not is_admin():
                            flash("Admin access required", "error")
                            return redirect(url_for("users"))

                        username = request.form.get("username", "").strip()
                        password = request.form.get("password", "").strip()
                        role = request.form.get("role", "viewer")

                        if not username or not password:
                            flash("Username and password are required", "error")
                        elif len(password) < 6:
                            flash("Password must be at least 6 characters", "error")
                        elif role not in ["admin", "viewer"]:
                            flash("Invalid role", "error")
                        else:
                            cursor.execute("""
                                INSERT INTO users (username, password_hash, role) 
                                VALUES (%s, %s, %s)
                            """, (username, generate_password_hash(password), role))
                            conn.commit()
                            flash(f"User '{username}' added successfully.", "success")

                    elif action == "change_password":
                        current_password = request.form.get("current_password", "")
                        new_password = request.form.get("new_password", "")
                        confirm_password = request.form.get("confirm_password", "")
                        user = get_user_by_id(current_user.id)

                        if not all([current_password, new_password, confirm_password]):
                            flash("All password fields are required", "error")
                        elif new_password != confirm_password:
                            flash("New passwords do not match", "error")
                        elif len(new_password) < 6:
                            flash("New password must be at least 6 characters", "error")
                        elif not user or not check_password_hash(user.password_hash, current_password):
                            flash("Current password is incorrect", "error")
                        else:
                            cursor.execute("""
                                UPDATE users SET password_hash = %s WHERE id = %s
                            """, (generate_password_hash(new_password), user.id))
                            conn.commit()
                            flash("Password changed successfully", "success")

            except psycopg2.IntegrityError:
                flash("Username already exists.", "error")
            except Exception as e:
                logger.error(f"[USER MANAGEMENT ERROR] {e}")
                flash("An error occurred while managing users.", "error")

            return redirect(url_for("users"))

        # --- GET Request: Load user list ---
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                cursor.execute("""
                    SELECT u.id, u.username, u.role, u.created_at, ua.last_activity
                    FROM users u
                    LEFT JOIN user_activity ua ON u.id = ua.user_id
                    ORDER BY u.username
                """)
                users_data = cursor.fetchall()

            active_user_ids = set()
            cutoff = datetime.now() - timedelta(minutes=30)

            for user in users_data:
                ts = user["last_activity"]
                try:
                    if isinstance(ts, str):
                        ts = datetime.fromisoformat(ts)
                    if isinstance(ts, datetime) and ts >= cutoff:
                        active_user_ids.add(user["id"])
                except Exception as e:
                    logger.warning(f"[USER TIME PARSE FAIL] {user['username']}: {e}")

            users_list = []
            for user in users_data:
                u = dict(user)
                u["is_active"] = u["id"] in active_user_ids
                users_list.append(u)

            logger.info(f"Loaded {len(users_list)} users, {len(active_user_ids)} active.")

        except Exception as e:
            logger.error(f"[USERS LOAD ERROR] {e}")
            flash("Error loading user list", "error")
            return render_template("users.html", users=[])

        return render_template("users.html", users=users_list)
    
    @app.route('/delete_user/<int:user_id>', methods=['POST'])
    @login_required
    def delete_user(user_id):
        if not current_user.is_authenticated:
            if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                return jsonify({'success': False, 'error': 'Session expired'}), 401
            return redirect(url_for('login'))

        if current_user.role != 'admin':
            return jsonify({'success': False, 'error': 'Unauthorized'}), 403

        if current_user.id == user_id:
            return jsonify({'success': False, 'error': 'You cannot delete your own account'}), 400

        try:
            with get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
                user = cursor.fetchone()

                if not user:
                    return jsonify({'success': False, 'error': 'User not found'}), 404

                cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
                cursor.execute("DELETE FROM user_activity WHERE user_id = %s", (user_id,))
                conn.commit()

            logger.info(f"[DELETE USER] Deleted user ID {user_id}")
            return jsonify({'success': True}), 200

        except Exception as e:
            logger.exception(f"[DELETE USER ERROR] {e}")
            return jsonify({'success': False, 'error': 'Internal server error'}), 500

    @app.route("/change_password", methods=["POST"])
    @login_required
    def change_password():
        try:
            current_password = request.form.get("current_password", "").strip()
            new_password = request.form.get("new_password", "").strip()
            confirm_password = request.form.get("confirm_password", "").strip()

            if not all([current_password, new_password, confirm_password]):
                flash("All fields are required", "error")
                return redirect(url_for("users"))

            if new_password != confirm_password:
                flash("New passwords do not match", "error")
                return redirect(url_for("users"))

            if len(new_password) < 6:
                flash("New password must be at least 6 characters", "error")
                return redirect(url_for("users"))

            user = get_user_by_id(current_user.id)
            if not user or not check_password_hash(user.password_hash, current_password):
                flash("Current password is incorrect", "error")
                return redirect(url_for("users"))

            with get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                cursor.execute("""
                    UPDATE users SET password_hash = %s WHERE id = %s
                """, (generate_password_hash(new_password), user.id))
                conn.commit()

            flash("Password changed successfully", "success")
            logger.info(f"[PASSWORD CHANGE] User {current_user.username} changed password.")
            return redirect(url_for("users"))

        except Exception as e:
            logger.error(f"[PASSWORD CHANGE ERROR] {e}")
            flash("Error changing password", "error")
            return redirect(url_for("users"))
    
    @app.route("/api/active_users")
    @login_required
    def api_active_users():
        try:
            active_users = get_active_users(minutes=30)  
            return jsonify({
                "active_count": len(active_users),
                "active_users": [
                    {
                        "username": user['username'],
                        "role": user['role'],
                        "last_activity": user['last_activity']
                    } for user in active_users
                ]
            })
        except Exception as e:
            logger.error(f"[API ACTIVE USERS ERROR] {e}")
            return jsonify({"error": "Internal server error"}), 500
        
    @app.before_request
    def before_request():
        if current_user.is_authenticated:
            try:
                update_user_activity(current_user.id)
            except Exception as e:
                logger.warning(f"[USER ACTIVITY TRACKING ERROR] {e}")

    return app

ensure_directories()
flask_app = create_app()
