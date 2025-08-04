import time
import uuid
import numpy as np
from collections import deque
from datetime import datetime

class KalmanFilter3D:
    def __init__(self, initial_position, initial_velocity=None):
        self.state = np.zeros((6, 1))  # [x, y, z, vx, vy, vz]
        self.state[:3, 0] = initial_position
        if initial_velocity:
            self.state[3:, 0] = initial_velocity

        self.P = np.eye(6)
        self.Q = np.eye(6) * 0.05
        self.R = np.eye(3) * 0.1  # Measurement noise: we only observe position
        self.H = np.hstack((np.eye(3), np.zeros((3, 3))))
        self.last_update = time.time()

    def update(self, position_measurement):
        now = time.time()
        dt = max(now - self.last_update, 1e-3)
        self.last_update = now

        A = np.eye(6)
        for i in range(3):
            A[i, i+3] = dt  # Position += velocity * dt

        self.state = A @ self.state
        self.P = A @ self.P @ A.T + self.Q

        z = np.array(position_measurement).reshape((3, 1))
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state += K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

    def get_state(self):
        pos = self.state[:3, 0]
        vel = self.state[3:, 0]
        return pos, vel

class ObjectTracker:
    def __init__(self, speed_limit_kmh=1, speed_limits_map=None):
        self.trackers = {}
        self.history = {}
        self.speed_limit_kmh = speed_limit_kmh
        self.speed_limits_map = speed_limits_map or {"default": speed_limit_kmh}
        self.max_age = 4  # seconds
        self.match_tolerance = 0.75  # meters in 3D
        self.logged_ids = set()  # Keep track of already logged speeding detections

    def get_limit_for(self, obj_type):
        return self.speed_limits_map.get(obj_type.upper(), self.speed_limits_map.get("default", self.speed_limit_kmh))

    def _generate_id(self, pos):
        rounded = tuple(round(x, 1) for x in pos)
        key = f"{rounded}"
        if key not in self.history:
            ts = datetime.now().strftime("%Y%m%d%H%M%S")
            self.history[key] = f"obj_{ts}_{uuid.uuid4().hex[:4]}"
        return self.history[key]

    def _distance(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def calculate_score(self, obj):
        speed_score = min(obj.get('speed_kmh', 0) / 3, 1.5)
        conf_score = obj.get('confidence', 0.0)
        signal_score = obj.get('signal_level', 0.0) / 80
        return speed_score + (conf_score * 2.0) + signal_score

    def update_tracks(self, detections, yolo_detections=None, frame_timestamp=None):
        current_time = time.time()
        updated_objects = []

        self.trackers = {
            oid: trk for oid, trk in self.trackers.items()
            if current_time - trk.last_update < self.max_age
        }

        for det in detections:
            if not all(k in det for k in ('x', 'y', 'z')):
                continue

            position = [det['x'], det['y'], det['z']]
            matched_id = None
            min_dist = float('inf')

            for oid, tracker in self.trackers.items():
                pred_pos, _ = tracker.get_state()
                dist = self._distance(pred_pos, position)
                if dist < self.match_tolerance and dist < min_dist:
                    matched_id = oid
                    min_dist = dist

            matched_id = det.get("object_id")

            if not matched_id:
                obj_type = det.get("type", "UNKNOWN").upper()
                date_str = datetime.fromtimestamp(current_time).strftime("%Y%m%d")
                radar_id = det.get("id")
                try:
                    radar_id = int(radar_id)
                except:
                    radar_id = len(self.trackers) + 1
                matched_id = f"{obj_type}_{date_str}_{radar_id}"
                det["object_id"] = matched_id  # assign it back

            # Always ensure the tracker is created before accessing
            if matched_id not in self.trackers:
                self.trackers[matched_id] = KalmanFilter3D(position)

            tracker = self.trackers[matched_id]
            tracker.update(position)
            pos, vel = tracker.get_state()
            speed_kmh = np.linalg.norm(vel) * 3.6
            det['object_id'] = matched_id

            snapshot_status = "PENDING"
            direction = det.get("direction", "unknown")
            motion_state = det.get("motion_state", "unknown")
            distance = np.linalg.norm(pos)

            det.update({
                'object_id': matched_id,
                'x': pos[0], 'y': pos[1], 'z': pos[2],
                'vx': vel[0], 'vy': vel[1], 'vz': vel[2],
                'distance': distance,
                'speed_kmh': speed_kmh,
                'timestamp': current_time,
                'snapshot_path': None,
                'snapshot_status': snapshot_status,
                'direction': direction,
                'motion_state': motion_state
            })

            det['score'] = self.calculate_score(det)
            updated_objects.append(det)

        return updated_objects
