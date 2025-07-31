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

        config_flag_path = "config_sent.flag"

        if not os.path.exists(config_flag_path):
            print("[INFO] Config not sent yet — sending now.")
            try:
                open(config_flag_path, "w").close()  
                with open(self.cfg_path, 'r') as f:
                    self.parser.sendCfg(f.readlines())
                print("[INFO] Radar config sent successfully.")
            except Exception as e:
                print(f"[ERROR] Failed to send config to radar: {e}")

    def get_targets(self):
        try:
            frame = self.parser.readAndParseUart()
            if frame and "trackData" in frame:
                return frame["trackData"]                                                                                                                                        
            return []
        except Exception as e:
            print(f"[ERROR] Failed to parse radar targets: {e}")
            return []
