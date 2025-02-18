import logging
import os
from datetime import datetime


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))


class Logger:
    log_dir = "logs"
    logs_to_zip = []
    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def __init__(self, name: str, log_file):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        log_path = os.path.join(self.log_dir, log_file)

        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s]: %(message)s', datefmt='%H:%M:%S')

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)

        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def get_logger(self):
        return self.logger
