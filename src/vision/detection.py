import time
import json
import threading
import numpy as np
import src.LocalCommon as lc


class Detection:
    def __init__(self, scale, time_frame, timeout_seconds=1.0):
        H, W = lc.DEFAULT_FRAME_DIMENSIONS[0], lc.DEFAULT_FRAME_DIMENSIONS[1]
        self.dimensions = (int(H * scale), int(W // 2 * scale))
        self.time_frame = time_frame
        self.timeout_seconds = timeout_seconds
        self.detected_point = None

        self._last_detection_time = None
        self._object_detected = False


    def get_detection(self, point, center_point):
        self._last_detection_time = time.time()
        self._object_detected = True
        self.detected_point = point
        if center_point < 0.35:
            print(f"Detected point: {self.detected_point} with distance {center_point:.2f}m")

    def has_recent_connection(self):
        if self._last_detection_time is None:
            return False

        elapsed = time.time() - self._last_detection_time
        if elapsed <= self.timeout_seconds:
            return True
        else:
            self._object_detected = False
            return False

    def set_dimensions(self, dimension):
        self.dimensions = dimension

    def get_object_detected(self):
        return self.has_recent_connection()

    def get_dimensions(self):
        return self.dimensions


# Make these values from the config
detector = Detection(0.2, 10000)
