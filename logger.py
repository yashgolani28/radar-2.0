import logging
import os
import threading

LOG_FILE = "system-logs/radar.log"  

class SafeFileHandler(logging.FileHandler):
    _lock = threading.Lock()

    def emit(self, record):
        with self._lock:
            try:
                super().emit(record)
            except RuntimeError as e:
                if "reentrant call" not in str(e):
                    raise  # only suppress specific error

def setup_logger(name="radar_logger"):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers: 
        return logger

    logger.setLevel(logging.DEBUG)
    file_handler = SafeFileHandler(LOG_FILE)
    console_handler = logging.StreamHandler()

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = setup_logger()
