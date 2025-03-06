### From Intel Liberalsense: https://github.com/IntelRealSense/librealsense/blob/master/doc/depth-from-stereo.md

import numpy
import cv2
from src.utils.utils import split_frame
from matplotlib import pyplot as plt


left  = cv2.imread("/Users/nicholasburczyk/Desktop/CS CLASS/CS499sp25/cs499/assets/images/stereo/motorcycle/im0.png", cv2.IMREAD_GRAYSCALE)
right = cv2.imread("/Users/nicholasburczyk/Desktop/CS CLASS/CS499sp25/cs499/assets/images/stereo/motorcycle/im1.png", cv2.IMREAD_GRAYSCALE)

fx = 200     # lense focal length
baseline = 193.001   # distance in mm between the two cameras
disparities = 256 # num of disparities to consider
block =  19       # block size to match
units = 0.512     # depth units, adjusted for the output to fit in one byte

sbm = cv2.StereoBM_create(numDisparities=disparities,
                          blockSize=block)

# calculate disparities
disparity = sbm.compute(left, right)
valid_pixels = disparity > 0

# calculate depth data
depth = numpy.zeros(shape=left.shape).astype("uint8")
depth[valid_pixels] = (fx * baseline) / (units * disparity[valid_pixels])

# visualize depth data
depth = cv2.equalizeHist(depth)
colorized_depth = numpy.zeros((left.shape[0], left.shape[1], 3), dtype="uint8")
temp = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
colorized_depth[valid_pixels] = temp[valid_pixels]
plt.imshow(colorized_depth)
plt.show()
