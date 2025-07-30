import time
from gui_parser import uartParser

class IWR6843Interface:
    def __init__(self, cli='COM19', data='COM20', cfg_path='isk_config.cfg'):
        self.cli = cli
        self.data = data
        self.cfg_path = cfg_path
        self.parser = uartParser(type='3D People Counting')
        self.parser.connectComPorts(cli, data)
        with open(cfg_path, 'r') as f:
            self.parser.sendCfg(f.readlines())

    def get_targets(self):
        try:
            out = self.parser.readAndParseUart()
            if out.get("error") == 0:
                targets = []
                for t in out.get("trackData", []):
                    try:
                        distance = (t["posX"]**2 + t["posY"]**2)**0.5
                        speed = (t["velX"]**2 + t["velY"]**2)**0.5 * 3.6
                        target = {
                            "id": int(t["id"]),
                            "distance": round(distance, 2),
                            "speed": round(speed, 2),
                            "confidence": round(t.get("confidence", 0), 2),
                            "timestamp": time.time(),
                            "source": "IWR6843-TRACK",
                            "type": t.get("type", "UNKNOWN"),
                            "posX": t.get("posX", 0.0),
                            "posY": t.get("posY", 0.0),
                            "posZ": t.get("posZ", 0.0),
                            "velX": t.get("velX", 0.0),
                            "velY": t.get("velY", 0.0),
                            "velZ": t.get("velZ", 0.0),
                            "accX": t.get("accX", 0.0),
                            "accY": t.get("accY", 0.0),
                            "accZ": t.get("accZ", 0.0),
                            "g": t.get("g", 0.0),
                            "speed_kmh": t.get("speed_kmh", speed),
                        }
                        targets.append(target)
                    except Exception as e:
                        print(f"[WARN] Malformed target skipped: {e}")
                return targets
        except Exception as e:
            print(f"[IWR6843] Error while parsing UART: {e}")
        return []