import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.utils.utils import split_frame

import cv2
import numpy as np


def compute_disparity(left_img, right_img):
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
    disparity = stereo.compute(left_img, right_img).astype(np.float32) / 16.0
    plt.imshow(disparity)
    plt.colorbar()
    plt.show()


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


if __name__ == "__main__":
    img = cv2.imread("/Users/nicholasburczyk/Desktop/CS CLASS/CS499sp25/cs499/src/vision/calibration/stereo/img_00.jpg")
    left, right = split_frame(img)
    #right = cv2.imread("/Users/nicholasburczyk/Desktop/CS CLASS/CS499sp25/cs499/assets/images/stereo/skates/im1.png")

    compute_disparity(left, right)