import time
import numpy as np
import os
import joblib
import lightgbm as lgb
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from collections import defaultdict, deque
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

class ObjectClassifier:
    def __init__(self, model_path="radar_lightgbm_model.pkl"):
        self.model_path = model_path
        self.model, self.scaler = self._load_or_create_model()
        self.object_cache = defaultdict(lambda: deque(maxlen=10))
        self.feature_buffer = deque(maxlen=1000)
        self.history_cache = {}

    def _load_or_create_model(self):
        if os.path.exists(self.model_path):
            return joblib.load(self.model_path)

        # Fallback model (dummy training if no model found)
        training_data = np.array([
            [0.0, 0.5, 0.0, 25, 0.1],
            [2.5, 1.2, 0.7, 35, 0.8],
            [5.8, 2.1, 1.6, 42, 1.5],
            [0.0, 15.0, 0.0, 85, 0.0],
            [25.0, 20.0, 6.9, 90, 15.2],
            [45.0, 35.0, 12.5, 88, 28.5],
            [75.0, 50.0, 20.8, 82, 45.8],
            [95.0, 80.0, 26.4, 78, 58.2],
            [3.2, 0.8, 0.9, 38, 1.2],
            [35.0, 25.0, 9.7, 87, 22.1],
            [12.0, 8.0, 3.3, 65, 8.5],
            [0.5, 0.8, 0.1, 30, 0.3],
        ])
        labels = np.array([
            'HUMAN', 'HUMAN', 'HUMAN', 'VEHICLE', 'VEHICLE',
            'VEHICLE', 'VEHICLE', 'VEHICLE', 'HUMAN', 'VEHICLE', 'BICYCLE', 'UNKNOWN'
        ])

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

    def classify_objects(self, objects):
        current_time = time.time()
        classified = []

        for obj in objects:
            features = self._extract_features(obj)
            obj_id = self._generate_id(obj)  
            feature_names = ["speed_kmh", "radar_distance", "velocity", "signal_level", "doppler_frequency"]
            df = pd.DataFrame([features], columns=feature_names)
            features_scaled = self.scaler.transform(df)
            probabilities = self.model.predict_proba(features_scaled)[0]
            classes = self.model.classes_

            top_idx = np.argmax(probabilities)
            obj_type = classes[top_idx]
            confidence = probabilities[top_idx]

            # No longer filtering UNKNOWN
            radar_distance = obj.get("radar_distance") or 0.0
            speed_kmh = obj.get("speed_kmh") or 0.0
            obj_type = obj.get("type", "UNKNOWN")

            if obj_type == 'VEHICLE' and speed_kmh < 6 and radar_distance < 10:
                obj["type"] = "HUMAN"
                confidence = min(confidence, 0.8)

            obj.update({
                'object_id': obj_id,
                'type': obj_type,
                'confidence': round(confidence, 3),
                'raw_probabilities': dict(zip(classes, probabilities.round(3)))
            })

            classified.append(obj)

        return classified

    def _extract_features(self, obj):
        return [
            obj.get('speed_kmh', 0.0),
            obj.get('radar_distance', 0.0),
            abs(obj.get('velocity', 0.0)),
            obj.get('signal_level', 0.0),
            obj.get('doppler_frequency', 0.0)
        ]

    def _generate_id(self, obj):
        dist_val = obj.get('radar_distance')
        dist = round(dist_val, 1) if dist_val is not None else 0.0
        vel = round(obj.get('velocity', 0), 2)
        return f"obj_{dist}_{vel}"
