import logging
import os

LOG_FILE = "system-logs/radar.log"  

def setup_logger(name="radar_logger"):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers: 
        return logger

    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(LOG_FILE)
    console_handler = logging.StreamHandler()

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = setup_logger()
