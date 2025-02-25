import cv2
import numpy as np
import matplotlib.pyplot as plt


class DepthMap:
    def __init__(self, left, right):
        self.imgLeft = cv2.imread(left, cv2.IMREAD_GRAYSCALE)
        self.imgRight = cv2.imread(right, cv2.IMREAD_GRAYSCALE)

    def computeDepthMapPM(self):
        nDispFactor = 12
        stereo = cv2.StereoBM.create(numDisparities=16*nDispFactor,
                                     blockSize=21)
        disparity = stereo.compute(self.imgLeft, self.imgRight)
        cv2.imwrite("output.png", disparity)
        plt.imshow(disparity, 'gray')
        plt.show()

    def computeDepthMapSGBM(self):
        window_size = 7
        min_disp = 16
        nDispFactor = 14
        num_disp = 16*nDispFactor-min_disp

        stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                       numDisparities=num_disp,
                                       blockSize=window_size,
                                       P1=8*3*window_size**2,
                                       P2=32*3*window_size**2,
                                       disp12MaxDiff=1,
                                       uniquenessRatio=15,
                                       speckleWindowSize=0,
                                       speckleRange=2,
                                       preFilterCap=63,
                                       mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        disparity = stereo.compute(self.imgLeft, self.imgRight).astype(np.float32) / 16.0
        cv2.imwrite("output.png", disparity)
        plt.imshow(disparity)
        plt.colorbar()
        plt.show()

def main():
    image_left = "/Users/nicholasburczyk/Desktop/CS CLASS/CS499sp25/cs499/assets/images/stereo/im0_chair.png"
    image_right = "/Users/nicholasburczyk/Desktop/CS CLASS/CS499sp25/cs499/assets/images/stereo/im1_chair.png"

    depth_obj = DepthMap(image_left, image_right)
    depth_obj.computeDepthMapSGBM()


if __name__ == "__main__":
    # img = cv2.imread("output_2.png")
    # img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    # cv2.imshow("image",img)
    # cv2.waitKey(0)
    main()
