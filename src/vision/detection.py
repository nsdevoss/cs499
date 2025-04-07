import time
import json
import threading
import numpy as np
import src.LocalCommon as lc
from src.utils.config import camera_server_arguments


class Detection:
    def __init__(self, scale, time_frame):
        H, W = lc.DEFAULT_FRAME_DIMENSIONS[0], lc.DEFAULT_FRAME_DIMENSIONS[1]
        self.dimensions = (int(H * scale), int(W // 2 * scale))
        self.time_frame = time_frame

    def start_detecting(self):
        pass

    def set_dimensions(self, dimension):
        self.dimensions = dimension

    def get_dimensions(self):
        return self.dimensions

    def export(self):
        pass


# Make these values from the config
detector = Detection(camera_server_arguments.get("scale"), 10000)
