import os
import json
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime
import logging


import logging
from logging.handlers import TimedRotatingFileHandler
import os
from datetime import datetime
import time
import json


def configuelogging():

    # Load logging configuration from JSON file
    with open(os.path.join(os.path.dirname(__file__), 'logging_config.json'), 'r') as config_file:
        log_config = json.load(config_file)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(lineno)d - %(message)s')

    logs_folder = 'logs'  # Set the folder name

    if not os.path.exists(logs_folder):
        os.makedirs(logs_folder)

    # Full path including log file name
    log_file_path = os.path.join(logs_folder, 'a2p')

    rotation_logging_handler = TimedRotatingFileHandler(log_file_path,
                                                        when=log_config.get(
                                                            'rotation_when', 'M'),
                                                        interval=log_config.get(
                                                            'rotation_interval', 5),
                                                        backupCount=log_config.get('backup_count', 2016))
    rotation_logging_handler.setLevel(logging.INFO)
    rotation_logging_handler.setFormatter(formatter)
    rotation_logging_handler.suffix = '%Y-%m-%d_%H-%M-%S.log'

    logger.addHandler(rotation_logging_handler)
    return logger
