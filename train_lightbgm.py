import psycopg2
import numpy as np
import joblib
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter
import os

MODEL_PATH = "radar_lightgbm_model.pkl"

DB_CONFIG = {
    "dbname": "iwr6843_db",
    "user": "radar_user",
    "password": "securepass123",
    "host": "localhost"
}

def fetch_training_data():
    query = """
        SELECT speed_kmh, radar_distance, velocity, signal_level, doppler_frequency,
               x, y, z, velX, velY, velZ, snr, noise, type
        FROM radar_data
        WHERE type IS NOT NULL AND TRIM(type) != ''
          AND speed_kmh IS NOT NULL AND radar_distance IS NOT NULL
          AND velocity IS NOT NULL AND signal_level IS NOT NULL AND doppler_frequency IS NOT NULL
          AND x IS NOT NULL AND y IS NOT NULL AND z IS NOT NULL
          AND velX IS NOT NULL AND velY IS NOT NULL AND velZ IS NOT NULL
          AND snr IS NOT NULL AND noise IS NOT NULL
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
        *features, label = row
        features[2] = abs(features[2])  # velocity magnitude
        X.append(features)
        y.append(label.strip().upper())

    X = np.array(X)
    y = np.array(y)

    if len(set(y)) == 0:
        print("[ABORT] Not enough class diversity to train model.")
        return False

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

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
    joblib.dump((model, scaler), MODEL_PATH)
    print(f"[SUCCESS] Model trained and saved to {MODEL_PATH}")
    print(f"ACCURACY: {acc * 100:.2f}")
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
