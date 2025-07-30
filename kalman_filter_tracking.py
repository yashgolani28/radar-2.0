import time
import uuid
import numpy as np
from collections import deque
from datetime import datetime

class KalmanFilter:
    def __init__(self, initial_position, initial_velocity=0.0):
        self.state = np.array([[initial_position], [initial_velocity]])  # [position, velocity]
        self.P = np.eye(2)  # Covariance matrix
        self.Q = np.array([[0.08, 0], [0, 0.08]])
        self.R = np.array([[0.02]])  # Measurement noise
        self.H = np.array([[1, 0]])  # Measurement function
        self.last_update = time.time()

    def update(self, position_measurement, velocity_measurement=None):
        now = time.time()
        dt = max(now - self.last_update, 1e-3)
        self.last_update = now

        A = np.array([[1, dt], [0, 1]])
        self.state = A @ self.state
        self.P = A @ self.P @ A.T + self.Q

        z = np.array([[position_measurement]])
        H = self.H
        R = self.R

        if velocity_measurement is not None:
            z = np.array([[position_measurement], [velocity_measurement]])
            H = np.array([[1, 0], [0, 1]])
            R = np.array([[0.05, 0], [0, 0.02]])

        y = z - H @ self.state
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.state += K @ y
        self.P = (np.eye(2) - K @ H) @ self.P

    def get_state(self):
        return self.state[0, 0], self.state[1, 0]  # position, velocity

class ObjectTracker:
    def __init__(self, speed_limit_kmh=1, speed_limits_map=None):
        self.trackers = {}
        self.history = {}
        self.speed_limit_kmh = speed_limit_kmh
        self.speed_limits_map = speed_limits_map or {"default": speed_limit_kmh}
        self.max_age = 4  # seconds
        self.match_tolerance = 0.35
        self.visual_tracks = {}
        self.max_distance_px = 50
        self.max_velocity_diff = 5.0

    def get_limit_for(self, obj_type):
        return self.speed_limits_map.get(obj_type.upper(), self.speed_limits_map.get("default", self.speed_limit_kmh))

    def _generate_id(self, radar_distance, velocity=0.0):
        rounded_dist = round(radar_distance, 1) if radar_distance is not None else 0.0
        rounded_vel = round(velocity, 1) if velocity is not None else 0.0
        key = f"{rounded_dist}_{rounded_vel}"

        if key not in self.history:
            obj_id = f"obj_{rounded_dist:.1f}_{rounded_vel:.1f}_{uuid.uuid4().hex[:4]}"
            self.history[key] = obj_id

        return self.history[key]

    def _euclidean(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def calculate_score(self, obj):
        speed_score = min(obj.get('speed_kmh', 0) / 3, 1.5)
        conf_score = obj.get('confidence', 0.0)
        signal_score = obj.get('signal_level', 0.0) / 80
        distance_weight = 1.0 if obj.get('radar_distance', 0) < 10 else 0.5
        return speed_score + (conf_score * 2.0) + (signal_score * distance_weight)

    def update_tracks(self, detections, yolo_detections=None, frame_timestamp=None):
        current_time = time.time()
        updated_objects = []

        self.trackers = {
            oid: tracker for oid, tracker in self.trackers.items()
            if current_time - tracker.last_update < self.max_age
        }

        if yolo_detections and frame_timestamp:
            expired_ids = [tid for tid, trk in self.visual_tracks.items() if frame_timestamp - trk['last_seen'] > self.max_age]
            for tid in expired_ids:
                del self.visual_tracks[tid]

            for det in yolo_detections:
                box = det['bbox']
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2
                best_match = None
                best_score = float('inf')

                for tid, trk in self.visual_tracks.items():
                    dist_px = self._euclidean(trk['center'], (cx, cy))
                    vdiff = abs(trk['velocity'] - detections[0]['velocity'])
                    if dist_px < self.max_distance_px and vdiff < self.max_velocity_diff:
                        score = dist_px + vdiff * 10
                        if score < best_score:
                            best_match = tid
                            best_score = score

                if best_match:
                    trk = self.visual_tracks[best_match]
                    trk.update({
                        'bbox': box,
                        'center': (cx, cy),
                        'last_seen': frame_timestamp,
                        'velocity': detections[0]['velocity'],
                        'radar_distance': detections[0]['radar_distance'],
                        'speed_kmh': abs(detections[0]['velocity']) * 3.6,
                        'class': det['class'],
                        'confidence': det['confidence']
                    })
                    trk['score'] = self.calculate_score(trk)
                    updated_objects.append({**trk, 'object_id': best_match, 'timestamp': frame_timestamp})
                else:
                    new_id = f"faux_{uuid.uuid4().hex[:6]}"
                    self.visual_tracks[new_id] = {
                        'bbox': box,
                        'center': (cx, cy),
                        'last_seen': frame_timestamp,
                        'velocity': detections[0]['velocity'],
                        'radar_distance': detections[0]['radar_distance'],
                        'speed_kmh': abs(detections[0]['velocity']) * 3.6,
                        'class': det['class'],
                        'confidence': det['confidence']
                    }
                    self.visual_tracks[new_id]['score'] = self.calculate_score(self.visual_tracks[new_id])
                    updated_objects.append({**self.visual_tracks[new_id], 'object_id': new_id, 'timestamp': frame_timestamp})

            return updated_objects

        for det in detections:
            if det.get('speed_kmh', 0) > 40:
                self.match_tolerance = 0.5
            else:
                self.match_tolerance = 0.25

            matched_id = None
            for oid, tracker in self.trackers.items():
                pred_x, pred_v = tracker.get_state()
                age = current_time - tracker.last_update

                if abs(pred_x - det['radar_distance']) < self.match_tolerance:
                    if age > 2.0 or abs(pred_v - det['velocity']) > 1.0:
                        continue
                    matched_id = oid
                    break

            if not matched_id:
                matched_id = self._generate_id(det['radar_distance'], det.get('velocity', 0.0))
                initial_pos = det.get('radar_distance', 0.0) or 0.0
                initial_vel = det.get('velocity', 0.0) or 0.0

                self.trackers[matched_id] = KalmanFilter(
                    initial_position=initial_pos,
                    initial_velocity=initial_vel
                )
                self.history[matched_id] = deque(maxlen=5)

            tracker = self.trackers[matched_id]

            velocity_measured = det.get('velocity', None)
            if velocity_measured is not None and abs(velocity_measured) > 0.01:
                prev_vel = tracker.get_state()[1]
                fused_vel = 0.6 * velocity_measured + 0.4 * prev_vel
                tracker.update(det['radar_distance'], fused_vel)
            else:
                tracker.update(det['radar_distance'])

            est_pos, est_vel = tracker.get_state()
            speed_kmh = abs(est_vel) * 3.6

            self.history[matched_id].append((current_time, est_pos))

            if len(self.history[matched_id]) < 2:
                continue

            det.update({
                'object_id': matched_id,
                'radar_distance': est_pos,
                'velocity': est_vel,
                'speed_kmh': speed_kmh,
                'timestamp': current_time
            })

            det['score'] = self.calculate_score(det)
            updated_objects.append(det)

        return updated_objects
