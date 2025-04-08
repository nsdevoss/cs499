import json
import os
from src.server.logger import server_logger


class Config:
    _config_data = {}

    DEFAULT_CONFIG = {
        "emulator_arguments": {
            "enabled": True,
            "stream_enabled": False,
            "video_name": "shovel",
            "encode_quality": 100
        },
        "camera_server_arguments": {
            "port": 9000,
            "host": "0.0.0.0",
            "socket_type": "TCP",
            "fps": 60,
            "scale": 0.2
        },
        "vision_arguments": {
            "enabled": True,
            "depth_map_capture": True,
            "calibration_file": "calib_50/calibration_50.npz",
            "StereoSGBM_args": {
                "minDisparity": 0,
                "numDisparities": 32,
                "blockSize": 5,
                "uniquenessRatio": 30,
                "speckleWindowSize": 100,
                "speckleRange": 2,
                "disp12MaxDiff": 1
            },
            "distance_args": {
                "max_dist": 0.5,
                "min_dist": 0,
                "color": [12, 237, 16],
                "alpha": 0.5,
                "min_area": 800
            }
        }
    }

    @classmethod
    def get_config_path(cls):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))
        return os.path.join(root_dir, "config.json")

    @classmethod
    def load_config(cls):
        CONFIG_PATH = cls.get_config_path()
        if not os.path.exists(CONFIG_PATH):
            server_logger.get_logger().warning(f"Config file '{CONFIG_PATH}' not found. Using default values.")
            cls._config_data = cls.DEFAULT_CONFIG.copy()
            return

        try:
            with open(CONFIG_PATH, "r") as f:
                cls._config_data = json.load(f)

            for key, default_value in cls.DEFAULT_CONFIG.items():
                if key not in cls._config_data:
                    server_logger.get_logger().warning(
                        f"Missing key '{key}' in config file. Using default: {default_value}")
                    cls._config_data[key] = default_value
                else:
                    server_logger.get_logger().info(f"Config value loaded: {key}: {cls._config_data[key]}")

            server_logger.get_logger().info(f"Config file '{CONFIG_PATH}' loaded successfully.")

        except Exception as e:
            server_logger.get_logger().error(f"Error loading config file '{CONFIG_PATH}': {e}. Using default values.")
            cls._config_data = cls.DEFAULT_CONFIG.copy()

    @classmethod
    def get(cls, key, default=None):
        keys = key.split('.')
        value = cls._config_data
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            server_logger.get_logger().warning(f"Config key '{key}' not found. Using default: {default}")
            return default
