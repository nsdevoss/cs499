import cv2
import numpy as np


def compute_disparity(left_frame, right_frame):
    left_gray = cv2.cvtColor(left_frame, cv2.IMREAD_GRAYSCALE)
    right_gray = cv2.cvtColor(right_frame, cv2.IMREAD_GRAYSCALE)

    window_size = 7
    min_disp = 16
    nDispFactor = 14
    num_disp = 16 * nDispFactor - min_disp

    stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                   numDisparities=num_disp,
                                   blockSize=window_size,
                                   P1=8 * 3 * window_size ** 2,
                                   P2=32 * 3 * window_size ** 2,
                                   disp12MaxDiff=1,
                                   uniquenessRatio=15,
                                   speckleWindowSize=0,
                                   speckleRange=2,
                                   preFilterCap=63,
                                   mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
    return disparity


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

    def compute_depth(self, disparity_map):
        disparity_map[disparity_map == 0] = 1.0
        depth_map = (self.focal_length * self.baseline) / disparity_map
        return depth_map
