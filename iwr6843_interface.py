# iwr6843_interface.py
from gui_parser import uartParser
from config_utils import load_config
import numpy as np
import time, os, sys

class IWR6843Interface:
    """
    Sends cfg once (guarded by config_sent.flag)
    Normalizes TLVs to: range_azimuth_heatmap, range_doppler_heatmap, azimuth_elevation_heatmap
    Keeps last good heatmap if a frame is partial.
    """
    def __init__(self):
        self.cfg = load_config()
        ports = self.cfg.get("iwr_ports", {}) or {}
        self.cli_port  = ports.get("cli",  "/dev/ttyUSB0")
        self.data_port = ports.get("data", "/dev/ttyUSB1")
        self.cfg_path  = ports.get("cfg_path", "isk_config.cfg")
        parser_type    = ports.get("parser_type", "3D People Counting")

        self._last_ra = None
        self._last_rd = None
        self._last_ae = None

        self.parser = uartParser(type=parser_type)
        self.parser.connectComPorts(self.cli_port, self.data_port)
        self._ensure_config_sent()

    def get_targets(self):
        try:
            frame = self.parser.readAndParseUart()
        except Exception as e:
            print(f"[ERROR] readAndParseUart failed: {e}", file=sys.stderr)
            return {}
        if not isinstance(frame, dict):
            return {}

        ra = self._extract_heatmap(frame, ["range_azimuth_heatmap","RANGE_AZIMUTH_HEATMAP","rangeAzimuthHeatMap","azimuthHeatMap"])
        rd = self._extract_heatmap(frame, ["range_doppler_heatmap","RANGE_DOPPLER_HEAT_MAP","rangeDopplerHeatMap","dopplerHeatMap"])
        ae = self._extract_heatmap(frame, ["azimuth_elevation_heatmap","AZIMUTH_ELEVATION_HEATMAP","azimuthElevationHeatMap"])

        if ra is None and self._last_ra is not None: frame["range_azimuth_heatmap"] = self._last_ra.copy()
        elif ra is not None: frame["range_azimuth_heatmap"] = ra; self._last_ra = ra

        if rd is None and self._last_rd is not None: frame["range_doppler_heatmap"] = self._last_rd.copy()
        elif rd is not None: frame["range_doppler_heatmap"] = rd; self._last_rd = rd

        if ae is None and self._last_ae is not None: frame["azimuth_elevation_heatmap"] = self._last_ae.copy()
        elif ae is not None: frame["azimuth_elevation_heatmap"] = ae; self._last_ae = ae

        return frame

    def _ensure_config_sent(self):
        flag = "config_sent.flag"
        def _send_cfg():
            with open(self.cfg_path, "r") as f:
                lines = [ln for ln in f if ln.strip() and not ln.strip().startswith("%")]
            self.parser.sendCfg(lines)
            open(flag, "w").close()
            print("[INFO] Radar config sent successfully.")

        if os.path.exists(flag):
            print("[INFO] Detected config_sent.flag, validating...")
            try:
                for _ in range(10):
                    tf = self.parser.readAndParseUart()
                    if isinstance(tf, dict) and isinstance(tf.get("trackData"), list):
                        print("[INFO] Radar already configured.")
                        return
                    time.sleep(0.3)
            except Exception as e:
                print(f"[WARN] Validation read failed: {e}")
            try: os.remove(flag)
            except: pass

        print("[INFO] Sending radar config...")
        _send_cfg()

    def _extract_heatmap(self, frame, keys):
        for k in keys:
            if k in frame and frame[k] is not None:
                arr = self._to_flat_array(frame[k])
                if arr is None: continue
                n = arr.size
                if n in (4096, 2048, 1024) or (n % 32 == 0 and n >= 512):
                    return arr
        return None

    def _to_flat_array(self, raw):
        try:
            if raw is None: return None
            if isinstance(raw, np.ndarray): return raw.astype(np.float32, copy=False).ravel()
            if isinstance(raw, (list, tuple)): return np.asarray(raw, dtype=np.float32).ravel()
            if isinstance(raw, (bytes, bytearray, memoryview)):
                return np.frombuffer(raw, dtype=np.int16).astype(np.float32, copy=False).ravel()
        except Exception as e:
            print(f"[DEBUG] _to_flat_array failed: {e}", file=sys.stderr)
        return None

def check_radar_connection(port="/dev/ttyACM1", baudrate=115200, timeout=2):
    try:
        import serial
        ser = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
        if ser.is_open: return ser
        ser.close()
    except Exception:
        pass
    return None
