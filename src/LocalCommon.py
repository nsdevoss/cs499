import os

#### OS Paths ####
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
CALIBRATION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../calibrations"))

#### Networkk Constants ####
MAX_UDP_PACKET = 65536

#### Frame Constants ####
CAMERA_BASELINE = 0.07
DEFAULT_CAMERA_FPS = 60
DEFAULT_FRAME_DIMENSIONS = (720, 2560)
