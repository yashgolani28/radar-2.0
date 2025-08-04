import time
from gui_parser import uartParser
import os

from config_utils import load_config

class IWR6843Interface:
    def __init__(self):
        config = load_config()
        ports = config.get("iwr_ports", {})
        self.cli = ports.get("cli", "/dev/serial/by-id/usb-Silicon_Labs_CP2105_Dual_USB_to_UART_Bridge_Controller_0151A9D6-if00-port0")
        self.data = ports.get("data", "/dev/serial/by-id/usb-Silicon_Labs_CP2105_Dual_USB_to_UART_Bridge_Controller_0151A9D6-if01-port0")
        self.cfg_path = ports.get("cfg_path", "isk_config.cfg")

        self.parser = uartParser(type='3D People Counting')
        self.parser.connectComPorts(self.cli, self.data)

        self.config_flag_path = "config_sent.flag"
        self._ensure_config()

    def _ensure_config(self):
        self.config_flag_path = "config_sent.flag"
        should_send = True

        if os.path.exists(self.config_flag_path):
            print("[INFO] Detected existing config_sent.flag, testing radar response...")
            try:
                # Wait up to 3 seconds for radar to respond
                for attempt in range(10):  # 10 × 0.3s = 3s total
                    frame = self.parser.readAndParseUart()
                    if isinstance(frame, dict) and "trackData" in frame and isinstance(frame["trackData"], list):
                        print("[INFO] Radar already configured and responding.")
                        should_send = False
                        break
                    time.sleep(0.3)
            except Exception as e:
                print(f"[WARN] Radar read error during validation: {e}")

            if should_send:
                print("[WARN] Radar not responding. Assuming reboot. Removing config_sent.flag.")
                os.remove(self.config_flag_path)

        if should_send:
            print("[INFO] Sending radar config...")
            try:
                with open(self.cfg_path, 'r') as f:
                    self.parser.sendCfg(f.readlines())
                open(self.config_flag_path, "w").close()
                print("[INFO] Radar config sent successfully.")
            except Exception as e:
                print(f"[ERROR] Failed to send config: {e}")

    def get_targets(self):
        try:
            frame = self.parser.readAndParseUart()
            return frame 
        except Exception as e:
            print(f"[ERROR] Failed to parse radar targets: {e}")
            return {}
        
def check_radar_connection(port="/dev/ttyACM1", baudrate=115200, timeout=2):
    try:
        import serial
        ser = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
        if ser.is_open:
            return ser
        else:
            ser.close()
            return None
    except Exception:
        return None
