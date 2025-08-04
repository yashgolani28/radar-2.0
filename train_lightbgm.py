import psycopg2
import numpy as np
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
from lightgbm import early_stopping, log_evaluation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter
import os
from collections import Counter
from sklearn.dummy import DummyClassifier

MODEL_PATH = "radar_lightgbm_model.pkl"

DB_CONFIG = {
    "dbname": "iwr6843_db",
    "user": "radar_user",                                                                   
    "password": "securepass123",
    "host": "localhost"
}

def fetch_training_data():
    query = """
        SELECT speed_kmh, distance, velocity, signal_level, doppler_frequency,
        x, y, z, velX, velY, velZ, snr, noise,
        accx, accy, accz,
        range_profile, noise_profile,
        type
        FROM radar_data
        WHERE type IS NOT NULL AND TRIM(type) != ''
        AND speed_kmh IS NOT NULL AND distance IS NOT NULL
    """
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()
        return rows
    except Exception as e:
        print(f"[ERROR] Failed to fetch training data: {e}")
        return []

def train_and_save_model(rows):
    X = []
    y = []

    for row in rows:
        *base, accx, accy, accz, range_profile, noise_profile, label = row

        # Velocity magnitude
        if base[2] is not None:
            base[2] = abs(base[2])
        else:
            base[2] = 0.0

        # Acceleration
        acc_feats = [accx or 0.0, accy or 0.0, accz or 0.0]

        # Profile statistics
        def stats_safe(arr):
            try:
                arr = [float(x) for x in arr]
                return [np.mean(arr), np.std(arr)]
            except:
                return [0.0, 0.0]

        range_feats = stats_safe(range_profile)
        noise_feats = stats_safe(noise_profile)

        # Combine all features
        features = [f if f is not None else 0.0 for f in base] + acc_feats + range_feats + noise_feats
        X.append(features)
        label_clean = label.strip().upper() if label and label.strip() else "UNKNOWN"
        y.append(label_clean)

    X = np.array(X)
    y = np.array(y)

    print(f"[DEBUG] Label distribution: {Counter(y)}")

    if len(set(y)) == 0:
        print("[ABORT] No valid labels to train model.")
        return False

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if len(set(y)) == 1:
        print(f"[WARN] Only one class present: {set(y)}")
        print("[INFO] Using DummyClassifier to simulate model training.")
        model = DummyClassifier(strategy="constant", constant=y[0])
        model.fit(X_scaled, y)
        
    else:
        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, stratify=y)
        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=len(set(y)),
            learning_rate=0.05,
            n_estimators=100,
            max_depth=8
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                early_stopping(stopping_rounds=10),
                log_evaluation(period=0)
            ]
        )
        acc = model.score(X_val, y_val)
        print(f"[INFO] ACCURACY: {acc*100:.2f}%")

    joblib.dump((model, scaler), MODEL_PATH)
    print(f"[SUCCESS] Model saved to {MODEL_PATH}")

    feature_names = [
        "speed_kmh", "distance", "velocity", "signal_level", "doppler_frequency",
        "x", "y", "z", "velX", "velY", "velZ", "snr", "noise",
        "accX", "accY", "accZ",
        "range_mean", "range_std", "noise_mean", "noise_std"
    ]

    # Plot
    if hasattr(model, "feature_importance"):
        importances = model.feature_importance()
        feature_names = [
            "speed_kmh", "distance", "velocity", "signal_level", "doppler_frequency",
            "x", "y", "z", "velX", "velY", "velZ", "snr", "noise",
            "accX", "accY", "accZ",
            "range_mean", "range_std", "noise_mean", "noise_std"
        ]
        plt.figure(figsize=(10, 6))
        plt.barh(feature_names, importances)
        plt.title("LightGBM Feature Importance")
        plt.xlabel("Importance Score")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        plt.show()
        print("[INFO] Saved feature importance to feature_importance.png")
    else:
        print("[SKIP] Feature importance skipped (DummyClassifier in use).")

    return True

def main():
    rows = fetch_training_data()
    print(f"[INFO] Fetched {len(rows)} samples from DB")
    if rows:
        train_and_save_model(rows)
    else:
        print("[ERROR] No valid training data found.")

if __name__ == "__main__":
    main()
