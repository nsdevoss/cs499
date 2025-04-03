import logging
import os
from datetime import datetime


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))


class Logger:
    """
    Logger Class

    This class is pretty simple as all it does is create the logger that python handles.

    To use this class you need to make an instance of it and pass in the name of the logger and the file name
    Python handles all of the logging but to get the instance of the logger just use get_logger()

    Example of logging:
    logger = Logger(name="LoggerClass", log_file="my_instance.log")

    logger.get_logger().info("This is an info text")
    logger.get_logger().warning("This is a warning text")
    logger.get_logger().error("This is an error text")
    """

    log_dir = "logs"
    logs_to_zip = []
    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def __init__(self, name: str, log_file):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        dir_path = os.path.join(ROOT_DIR, self.log_dir)
        log_path = os.path.join(ROOT_DIR, self.log_dir, log_file)

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s]: %(message)s', datefmt='%H:%M:%S')

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)

        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def get_logger(self):
        return self.logger


server_logger = Logger(name="ServerLogger", log_file="server.log")
client_logger = Logger(name="ClientLogger", log_file="client.log")
webserver_logger = Logger(name="WebServerLogger", log_file="webserver.log")
