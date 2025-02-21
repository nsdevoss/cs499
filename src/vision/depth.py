import cv2
import numpy as np

class DepthEstimator:
    def __init__(self, baseline, focal_length):
        self.baseline = baseline
        self.focal_length = focal_length

        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=64,
            blockSize=9,
            uniquenessRatio=15,
            speckleWindowSize=50,
            speckleRange=32,
            disp12MaxDiff=1
        )

    def compute_disparity(self, left_frame, right_frame):
        left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

        disparity = self.stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
        return disparity

    def compute_depth(self, disparity_map):
        disparity_map[disparity_map == 0] = 1.0
        depth_map = (self.focal_length * self.baseline) / disparity_map
        return depth_map
