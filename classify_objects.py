import time
import numpy as np
import os
import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import defaultdict, deque
import warnings
from config_utils import load_config

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

class ObjectClassifier:
    def __init__(self, model_path="radar_lightgbm_model.pkl"):
        self.model_path = model_path
        self.model, self.scaler = self._load_or_create_model()
        self.object_cache = defaultdict(lambda: deque(maxlen=10))
        self.feature_buffer = deque(maxlen=1000)
        self.config = load_config()
        self.history_cache = {}

    def _load_or_create_model(self):
        if os.path.exists(self.model_path):
            return joblib.load(self.model_path)

        training_data = np.random.uniform(0, 1, size=(10, 13))
        labels = np.array(['HUMAN', 'VEHICLE', 'BICYCLE', 'UNKNOWN', 'CAR', 'TRUCK', 'BUS', 'BIKE', 'HUMAN', 'VEHICLE'])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(training_data)

        model = lgb.LGBMClassifier(
            n_estimators=50,
            max_depth=8,
            learning_rate=0.1,
            objective='multiclass'
        )
        model.fit(X_scaled, labels)

        joblib.dump((model, scaler), self.model_path)
        return model, scaler

    def _extract_features(self, obj):
        # Compute safe stats
        def safe_stats(arr):
            if not arr:
                return [0.0, 0.0]
            try:
                arr = [float(x) for x in arr if x is not None]
                return [np.mean(arr), np.std(arr)]
            except:
                return [0.0, 0.0]

        range_feats = safe_stats(obj.get("range_profile", []))
        noise_feats = safe_stats(obj.get("noise_profile", []))

        return [
            obj.get("speed_kmh", 0.0),
            obj.get("radar_distance", obj.get("distance", 0.0)),
            abs(obj.get("velocity", 0.0)),
            obj.get("signal_level", 0.0),
            obj.get("doppler_frequency", 0.0),
            obj.get("posX", 0.0),
            obj.get("posY", 0.0),
            obj.get("posZ", 0.0),
            obj.get("velX", 0.0),
            obj.get("velY", 0.0),
            obj.get("velZ", 0.0),
            obj.get("snr", 0.0),
            obj.get("noise", 0.0),
            obj.get("accX", 0.0),
            obj.get("accY", 0.0),
            obj.get("accZ", 0.0),
            *range_feats,
            *noise_feats
        ]

    def classify_objects(self, objects):
        classified = []

        for obj in objects:
            features = self._extract_features(obj)

            df = pd.DataFrame([features], columns=[
                "speed_kmh", "radar_distance", "velocity", "signal_level", "doppler_frequency",
                "x", "y", "z", "velX", "velY", "velZ", "snr", "noise",
                "accX", "accY", "accZ",
                "range_mean", "range_std", "noise_mean", "noise_std"
            ])
            features_scaled = self.scaler.transform(df)
            probabilities = self.model.predict_proba(features_scaled)[0]
            classes = self.model.classes_

            top_idx = np.argmax(probabilities)
            obj_type = classes[top_idx]
            confidence = probabilities[top_idx]

            radar_distance = obj.get("radar_distance", 0.0)
            speed_kmh = obj.get("speed_kmh", 0.0)

            if obj_type == 'VEHICLE' and speed_kmh < 6 and radar_distance < 10:
                obj_type = "HUMAN"
                confidence = min(confidence, 0.8)

            obj.update({
                'type': obj_type,
                'confidence': round(confidence, 3),
                'raw_probabilities': dict(zip(classes, probabilities.round(3)))
            })

            # Add a composite score (confidence × SNR)
            obj["score"] = round(obj["confidence"] * obj.get("snr", 1.0), 2)

            # Motion state and direction
            limits = self.config.get("dynamic_speed_limits", {})
            default_limit = limits.get("default", 5.0)
            speed_limit = limits.get(obj_type.upper(), default_limit)

            vel_x = obj.get("velX", 0.0)
            vel_y = obj.get("velY", 0.0)

            if speed_kmh >= speed_limit:
                obj["motion_state"] = "SPEEDING"
            elif speed_kmh > 0.3:
                obj["motion_state"] = "MOVING"
            else:
                obj["motion_state"] = "STATIONARY"

            if abs(vel_y) < 0.05 and abs(vel_x) < 0.05:
                obj["direction"] = "STATIC"
            elif abs(vel_y) >= abs(vel_x):
                obj["direction"] = "AWAY" if vel_y > 0 else "TOWARDS"
            else:
                obj["direction"] = "LEFT" if vel_x > 0 else "RIGHT"

            classified.append(obj)

        return classified
