import json
import os

CONFIG_FILE = "config.json"

def load_config():
    default_config = {
        "cooldown_seconds": 0.5,
        "retention_days": 30,
        "selected_camera": 0,
        "speed_limit_kmh": 3.0,
        "annotation_conf_threshold": 0.5,
        "label_format": "{type} | {speed:.1f} km/h",
        "cameras": [
            {
                "url": "",
                "username": "",
                "password": ""
            }
        ],
        "dynamic_speed_limits": {
            "default": 3.0,
            "HUMAN": 4.0,
            "CAR": 70.0,
            "TRUCK": 50.0,
            "BUS": 50.0,
            "BIKE": 60.0,
            "BICYCLE": 10.0,
            "UNKNOWN": 50.0
        }
    }
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value

            # Ensure cameras is a list of dicts with proper keys
            if not isinstance(config.get("cameras"), list) or not all(isinstance(cam, dict) for cam in config["cameras"]):
                config["cameras"] = default_config["cameras"]

            return config
    except (FileNotFoundError, json.JSONDecodeError):
        return default_config
