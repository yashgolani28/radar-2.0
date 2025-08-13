import numpy as np
from datetime import datetime
from math import inf

# --- Radar constants (computed from config if available) ---
_C_MPS = 299_792_458.0
try:
    from config_utils import load_config
    _cfg = load_config()
    _fc_ghz = float(_cfg.get("radar", {}).get("center_frequency_ghz", 60.0))
except Exception:
    _fc_ghz = 60.0  # fallback for IWR6843 (~60 GHz)
_LAMBDA_M = _C_MPS / (_fc_ghz * 1e9)


class KalmanFilter3D:
    """
    Constant-velocity Kalman filter in 3D for position and velocity.
    State: [x y z vx vy vz]^T
    """
    def __init__(self, init_pos, dt=0.1, initial_velocity=None):
        self.state = np.zeros((6, 1), dtype=float)
        self.state[0:3, 0] = np.array(init_pos, dtype=float)

        # If provided, seed the velocity so speed doesn't collapse to ~0
        if initial_velocity is not None and len(initial_velocity) == 3:
            try:
                vx, vy, vz = [float(v) for v in initial_velocity]
                self.state[3, 0] = vx
                self.state[4, 0] = vy
                self.state[5, 0] = vz
            except Exception:
                pass

        # Default timestep; can be overridden per update/predict
        self.dt = float(dt)

        # State transition (will be rebuilt on-the-fly when dt changes)
        self._build_F(self.dt)

        # Observe only position
        self.H = np.zeros((3, 6), dtype=float)
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = 1.0

        # Process/measurement noise (tuned conservatively for radar)
        self.Q_base = 0.2
        self.R = np.eye(3, dtype=float) * 0.5

        # Covariance
        self.P = np.eye(6, dtype=float) * 10.0

    def _build_F(self, dt):
        self.F = np.eye(6, dtype=float)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

    def predict(self, dt=None):
        if dt is None:
            dt = self.dt
        else:
            dt = float(dt)
        if dt <= 0:
            dt = self.dt

        # Update transition for current dt
        self._build_F(dt)

        # Process noise scaled with dt
        q = self.Q_base
        G = np.array([[0.5*dt**2], [0.5*dt**2], [0.5*dt**2], [dt], [dt], [dt]], dtype=float)
        Q = (G @ G.T) * q

        # Predict
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + Q

    def update(self, meas_pos, dt=None):
        """
        meas_pos: iterable of (x, y, z)
        """
        if dt is not None:
            # Predict with the dt from last update time
            self.predict(dt=dt)
        else:
            self.predict(dt=self.dt)

        z = np.array(meas_pos, dtype=float).reshape(3, 1)

        # Innovation
        y = z - (self.H @ self.state)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update
        self.state = self.state + K @ y
        I = np.eye(6, dtype=float)
        self.P = (I - K @ self.H) @ self.P

    def get_state(self):
        pos = self.state[0:3, 0]
        vel = self.state[3:6, 0]
        return pos, vel


class ObjectTracker:
    def __init__(self, speed_limit_kmh=1, speed_limits_map=None):
        self.trackers = {}          # object_id (TRK_#####) -> KalmanFilter3D
        self.last_time = {}         # object_id -> last update timestamp (float)
        self.speed_limit_kmh = float(speed_limit_kmh)
        self.speed_limits_map = speed_limits_map or {"default": self.speed_limit_kmh}
        self.max_age = 4.0          # seconds (for pruning)
        self.match_tolerance = 0.75 # meters in 3D for association
        self.logged_ids = set()     # speeding already logged
        self.seq = 0                # persistent tracker id counter (TRK_#####)

        # --- duplicate suppression (emit-gating) ---
        self.min_emit_pos = 0.06    # m: minimal positional change to emit
        self.min_emit_vel = 0.12    # m/s: minimal velocity change to emit
        self.min_emit_dt  = 0.12    # s: minimal time delta to emit
        self.last_emitted = {}      # oid -> (t, pos(np.array 3,), vel(np.array 3,))

        # --- innovation gating ---
        self.innovation_gate_pos = 1.0  # m: reject update if position innovation exceeds AND...
        self.innovation_gate_vel = 2.0  # m/s: ...velocity innovation exceeds

        # --- raw TID -> persistent TRK mapping (short-lived) ---
        self.tid_map  = {}          # "TID_4" -> "TRK_00012"
        self.tid_seen = {}          # "TID_4" -> last_ts
        self.tid_timeout = 2.0      # expire mapping if not seen recently

    def get_limit_for(self, obj_type):
        if obj_type is None:
            return self.speed_limits_map.get("default", self.speed_limit_kmh)
        return self.speed_limits_map.get(str(obj_type).upper(), self.speed_limits_map.get("default", self.speed_limit_kmh))

    def _distance(self, a, b):
        return float(np.linalg.norm(np.array(a, dtype=float) - np.array(b, dtype=float)))

    def _prune_old_tracks(self, now_ts):
        to_delete = []
        for oid, tlast in self.last_time.items():
            if now_ts - tlast > self.max_age:
                to_delete.append(oid)
        for oid in to_delete:
            self.trackers.pop(oid, None)
            self.last_time.pop(oid, None)

        # prune stale TID links
        stale = []
        for tid, ts in list(self.tid_seen.items()):
            if now_ts - ts > self.tid_timeout or self.tid_map.get(tid) not in self.trackers:
                stale.append(tid)
        for tid in stale:
            self.tid_map.pop(tid, None)
            self.tid_seen.pop(tid, None)

    def _extract_position(self, det):
        """
        Attempt to get (x,y,z). If missing, try to derive from distance & azimuth/elevation (radians).
        """
        if all(k in det for k in ("x", "y", "z")):
            return float(det["x"]), float(det["y"]), float(det["z"])

        # Fallbacks (best-effort)
        r = float(det.get("distance", 0.0))
        az = float(det.get("azimuth", 0.0))
        el = float(det.get("elevation", 0.0))
        x = r * np.cos(el) * np.cos(az)
        y = r * np.cos(el) * np.sin(az)
        z = r * np.sin(el)
        return float(x), float(y), float(z)

    def calculate_score(self, det):
        """
        Lightweight score combining speed, signal, and (optional) classifier confidence.
        """
        speed = float(det.get("speed_kmh", 0.0))
        sig = float(det.get("signal_level", 0.0))
        conf = float(det.get("confidence", 0.5))
        limit = max(self.get_limit_for(det.get("type", "default")), 1e-3)
        speed_ratio = min(speed / limit, 3.0)
        # Weighted sum; tweak if needed
        return float(0.6 * speed_ratio + 0.3 * conf + 0.1 * np.tanh(sig / 10.0))

    def update_tracks(self, detections, current_time=None):
        """
        Update trackers with a list of radar detections.
        Each detection is a dict expected to contain at least:
        - type (str), source/source_id (for TID), x,y,z or distance/azimuth/elevation, snr, gain
        Returns a list of enriched detection dicts with smoothed state & metadata.
        """
        if current_time is None:
            current_time = datetime.now().timestamp()

        updated_objects = []

        for det in detections:
            position = self._extract_position(det)

            # ---- Stable key preference: use radar TID only for trackData, not pointcloud groupings
            matched_id = None
            source = str(det.get("source", "")).lower()
            raw_id = det.get("source_id") or det.get("id")
            if source == "pointcloud":
                raw_id = None
            if isinstance(raw_id, (int, float)):
                stable_key = f"TID_{int(raw_id)}"
            elif isinstance(raw_id, str) and raw_id.startswith("TID_"):
                stable_key = raw_id
            else:
                stable_key = None

            # If we already know which TRK this TID belongs to, use it directly
            if stable_key and stable_key in self.tid_map and self.tid_map[stable_key] in self.trackers:
                matched_id = self.tid_map[stable_key]

            # Otherwise, try nearest neighbor association
            if matched_id is None:
                min_dist = inf
                for oid, kf in self.trackers.items():
                    last_t = self.last_time.get(oid, current_time)
                    dt = max(0.0, float(current_time - last_t))
                    pos_pred, _ = self._predict_peek(kf, dt)
                    dist = self._distance(pos_pred, position)
                    if dist < self.match_tolerance and dist < min_dist:
                        matched_id = oid
                        min_dist = dist

            # Measurement velocity vector (if available)
            meas_v = det.get("initial_velocity")
            if not meas_v:
                meas_v = [
                    det.get("vx", det.get("velX", 0.0)),
                    det.get("vy", det.get("velY", 0.0)),
                    det.get("vz", det.get("velZ", 0.0)),
                ]
            try:
                meas_v = [float(meas_v[0]), float(meas_v[1]), float(meas_v[2])]
            except Exception:
                meas_v = [0.0, 0.0, 0.0]

            # Create a new persistent tracker if none matched
            if not matched_id:
                self.seq += 1
                matched_id = f"TRK_{self.seq:05d}"
                self.trackers[matched_id] = KalmanFilter3D(position, initial_velocity=meas_v)
                self.last_time[matched_id] = float(current_time)

            # Link/refresh TID→TRK mapping if we have a true TID
            if stable_key:
                self.tid_map[stable_key] = matched_id
                self.tid_seen[stable_key] = float(current_time)

            # Innovation gating vs. current prediction
            kf = self.trackers[matched_id]
            last_t = self.last_time.get(matched_id, current_time)
            dt = max(0.0, float(current_time - last_t))
            pos_pred, vel_pred = self._predict_peek(kf, dt)
            try:
                v_meas = np.array(meas_v, dtype=float)
            except Exception:
                v_meas = np.zeros(3, dtype=float)

            innov_pos = float(np.linalg.norm(np.array(position, dtype=float) - pos_pred))
            innov_vel = float(np.linalg.norm(v_meas - vel_pred))
            if (innov_pos > self.innovation_gate_pos) and (innov_vel > self.innovation_gate_vel):
                # Reject this measurement: advance the filter but do not correct, and skip emission
                kf.predict(dt=dt)
                self.last_time[matched_id] = float(current_time)
                continue
            else:
                # Accept correction
                kf.update(position, dt=dt)
                self.last_time[matched_id] = float(current_time)

            # Fused state (nudge towards measured velocity to avoid decay)
            pos, vel = kf.get_state()
            try:
                m = np.array(meas_v, dtype=float)
                v = np.array(vel, dtype=float)
                alpha = 0.6
                fused = (1.0 - alpha) * v + alpha * m
                kf.state[3:, 0] = fused
                vel = fused
            except Exception:
                pass

            vel_mag = float(np.linalg.norm(vel))
            speed_kmh = float(vel_mag * 3.6)

            # LOS radial velocity (toward radar negative if range decreasing)
            rng = float(np.linalg.norm(pos))
            if rng > 1e-6:
                los = pos / rng
                v_radial = float(np.dot(vel, los))
            else:
                v_radial = 0.0

            # Doppler frequency, signed
            doppler = float((2.0 * v_radial) / _LAMBDA_M)

            snr = float(det.get("snr", 0.0))
            gain = float(det.get("gain", 1.0))
            signal_level = float(gain * snr) if (gain and snr) else 0.0

            # --- Direction fallback from vy if not provided ---
            direction = det.get('direction', det.get('Direction', None))
            if direction in (None, "unknown"):
                vy = float(vel[1]) if hasattr(vel, "__len__") else float(det.get("velY", 0.0))
                if vy < -0.05:
                    direction = "approaching"
                elif vy > 0.05:
                    direction = "departing"
                else:
                    direction = "stationary"

            det['object_id'] = matched_id  # persistent, UI/DB should use this
            if stable_key:
                det['source_id'] = stable_key  # carry raw radar id separately

            det.update({
                'x': float(pos[0]), 'y': float(pos[1]), 'z': float(pos[2]),
                'vx': float(vel[0]), 'vy': float(vel[1]), 'vz': float(vel[2]),
                'velX': float(vel[0]), 'velY': float(vel[1]), 'velZ': float(vel[2]),
                'velocity_vector': [float(vel[0]), float(vel[1]), float(vel[2])],
                'velocity': float(vel_mag),
                'radial_velocity': float(v_radial),
                'speed_kmh': float(speed_kmh),
                'doppler_frequency': float(doppler),
                'signal_level': float(signal_level),
                'distance': float(rng),
                'timestamp': float(current_time),
                'snapshot_path': det.get('snapshot_path', None),
                'snapshot_status': det.get('snapshot_status', "PENDING"),
                'direction': direction,
                'motion_state': det.get('motion_state', det.get('MotionState', "unknown")),
            })

            det['score'] = self.calculate_score(det)

            # --- Emission debouncing: only output if meaningfully changed ---
            last = self.last_emitted.get(matched_id)
            pos_vec = np.array([det['x'], det['y'], det['z']], dtype=float)
            vel_vec = np.array([det['vx'], det['vy'], det['vz']], dtype=float)
            emit = True
            if last:
                lt, lp, lv = last
                dt_emit = float(current_time - lt)
                if (dt_emit < self.min_emit_dt and
                    float(np.linalg.norm(pos_vec - lp)) < self.min_emit_pos and
                    float(np.linalg.norm(vel_vec - lv)) < self.min_emit_vel):
                    emit = False
            if emit:
                self.last_emitted[matched_id] = (float(current_time), pos_vec, vel_vec)
                updated_objects.append(det)

        self._prune_old_tracks(float(current_time))
        return updated_objects

    @staticmethod
    def _predict_peek(kf: KalmanFilter3D, dt: float):
        """
        Predict without mutating the real filter (for association).
        """
        # Copy minimal state
        state = kf.state.copy()
        P = kf.P.copy()

        # Build F & Q like in kf.predict(dt)
        F = np.eye(6, dtype=float)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        q = kf.Q_base
        G = np.array([[0.5*dt**2], [0.5*dt**2], [0.5*dt**2], [dt], [dt], [dt]], dtype=float)
        Q = (G @ G.T) * q

        state_pred = F @ state
        P_pred = F @ P @ F.T + Q
        pos_pred = state_pred[0:3, 0]
        vel_pred = state_pred[3:6, 0]
        return pos_pred, vel_pred
