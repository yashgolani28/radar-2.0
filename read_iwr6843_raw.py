from iwr6843_interface import IWR6843Interface
from radar_logger import IWR6843Logger
import time
import socket
import json
from config_utils import load_config
from classify_objects import ObjectClassifier

classifier = ObjectClassifier()                                                                                                                                                                                         
cfg = load_config()
ports = cfg.get("iwr_ports", {})
cli_port = ports.get("cli", "COM19")
data_port = ports.get("data", "COM20")
cfg_path = ports.get("cfg_path", "isk_config.cfg")

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
DEST = ("127.0.0.1", 5005)

radar = IWR6843Interface(cli=cli_port, data=data_port, cfg_path=cfg_path)
logger = IWR6843Logger()

print("[INFO] Started reading radar data... Press Ctrl+C to stop.")

try:
    while True:
       targets = radar.get_targets()
       if targets:
        classified = classifier.classify_objects(targets)
        sock.sendto(json.dumps(classified).encode(), DEST)
        print(f"\n[Frame @ {time.strftime('%H:%M:%S')}] {len(classified)} target(s)")
        for obj in classified:
            print(
                f"ID: {obj['id']:>2} | "
                f"Type: {obj['type']:<8} | "
                f"Speed: {obj['speed_kmh']:>6.2f} km/h | Dist: {obj['distance']:>5.2f} m | "
                f"Conf: {obj['confidence']:.2f} | G: {obj['g']:.2f}\n"
                f" ↳ Pos [X:{obj['posX']:>5.2f} Y:{obj['posY']:>5.2f} Z:{obj['posZ']:>5.2f}] m | "
                f"Vel [X:{obj['velX']:>5.2f} Y:{obj['velY']:>5.2f} Z:{obj['velZ']:>5.2f}] m/s | "
                f"Acc [X:{obj['accX']:>5.2f} Y:{obj['accY']:>5.2f} Z:{obj['accZ']:>5.2f}] m/s²"
            )
            logger.log_targets(classified)
        else:
            print("No targets.")
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\n[INFO] Stopped radar read loop.")