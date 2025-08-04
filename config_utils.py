import json
import os
from logger import logger

CONFIG_FILE = "config.json"

def load_config():
    default_config = {
        "cooldown_seconds": 0.5,
        "retention_days": 30,
        "selected_camera": 0,
        "annotation_conf_threshold": 0.5,
        "label_format": "{type} | {speed:.1f} km/h",
        "acceleration_threshold": 2.0,
        "min_acc_violation_frames": 3,
        "iwr_ports": {
            "cli": "/dev/ttyACM1",
            "data": "/dev/ttyACM2",
            "cfg_path": "isk_config.cfg"
        },
        "cameras": [
            {
                "url": "",
                "snapshot_url": "",
                "username": "",
                "password": "",
                "stream_type": "mjpeg"
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

            # Fill in missing keys
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value

            # Validate nested camera structure
            if not isinstance(config.get("cameras"), list) or not all(isinstance(cam, dict) for cam in config["cameras"]):
                config["cameras"] = default_config["cameras"]

            # Validate nested iwr_ports structure
            if "iwr_ports" not in config or not isinstance(config["iwr_ports"], dict):
                config["iwr_ports"] = default_config["iwr_ports"]

            # Validate dynamic speed limits
            if "dynamic_speed_limits" not in config or not isinstance(config["dynamic_speed_limits"], dict):
                config["dynamic_speed_limits"] = default_config["dynamic_speed_limits"]

            if "default" not in config["dynamic_speed_limits"]:
                config["dynamic_speed_limits"]["default"] = default_config["dynamic_speed_limits"]["default"]

            return config

    except (FileNotFoundError, json.JSONDecodeError):
        logger.warning("[CONFIG] Using default config")
        return default_config
