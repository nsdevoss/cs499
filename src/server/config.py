import json
import os
from src.server.logger import server_logger


class Config:
    _config_data = {}

    DEFAULT_CONFIG = {
      "emulator_arguments": {
        "enabled": True,
        "stream_enabled": True,
        "video_name": "cube"
      },
      "server_port": 9000,
      "video_arguments": {
        "display": True,
        "fps": 30,
        "resolution": [2560, 720]
      },
      "vision_arguments": {
        "enabled": False,
        "depth_threshold": 0.8,
        "StereoSGBM_args": {
          "minDisparity": 0,
          "numDisparities": 32,
          "blockSize": 5,
          "uniquenessRatio": 15,
          "speckleWindowSize": 100,
          "speckleRange": 2,
          "disp12MaxDiff": 1
        },
        "scale": 0.5,
        "calibration_file": "calib_50/calibration_50.npz",
        "camera_parameters": {
          "baseline": 0.07,
          "viewing_angle": 120
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
