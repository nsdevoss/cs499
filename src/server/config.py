import json
import os


class Config:
    _config_data = {}
    _logger = None

    DEFAULT_CONFIG = {
        "emulator_arguments": {
            "enabled": False,
            "stream_enabled": False,
            "video_name": "dog"
        },
        "server_port": 9000,
        "video_arguments": {
            "display": True,
            "fps": 30
        },
        "vision_arguments": {
            "stitch": False,
            "depth_estimation": False,
            "calibration_file": ""
        },
        "camera_parameters": {
            "baseline": 70,
            "focal_length": 25,
            "viewing_angle": 120
        }
    }

    @classmethod
    def set_logger(cls, logger):
        cls._logger = logger

    @classmethod
    def get_config_path(cls):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))
        return os.path.join(root_dir, "config.json")

    @classmethod
    def load_config(cls):
        CONFIG_PATH = cls.get_config_path()
        if not os.path.exists(CONFIG_PATH):
            cls._logger.get_logger().warning(f"Config file '{CONFIG_PATH}' not found. Using default values.")
            cls._config_data = cls.DEFAULT_CONFIG.copy()
            return

        try:
            with open(CONFIG_PATH, "r") as f:
                cls._config_data = json.load(f)

            for key, default_value in cls.DEFAULT_CONFIG.items():
                if key not in cls._config_data:
                    cls._logger.get_logger().warning(
                        f"Missing key '{key}' in config file. Using default: {default_value}")
                    cls._config_data[key] = default_value

            cls._logger.get_logger().info(f"Config file '{CONFIG_PATH}' loaded successfully.")

        except Exception as e:
            cls._logger.get_logger().error(f"Error loading config file '{CONFIG_PATH}': {e}. Using default values.")
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
            cls._logger.get_logger().warning(f"Config key '{key}' not found. Using default: {default}")
            return default
