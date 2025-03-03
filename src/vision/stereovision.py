import numpy as np
import cv2

cv_file = cv2.FileStorage()
cv_file.open('stereoMap.xml', cv2.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()
cv_file.release()

cap_right = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
cap_left = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)

print("Before While")
while cap_right.isOpened() and cap_left.isOpened():

    success_right, frame_right = cap_right.read()
    success_left, frame_left = cap_left.read()

    if not success_right or not success_left:
        print("Error: Couldn't read frames")
        break

    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)

    rectified_right = cv2.remap(gray_right, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    rectified_left = cv2.remap(gray_left, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

    disparity = stereo.compute(rectified_left, rectified_right)

    disparity_visual = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disparity_visual = np.uint8(disparity_visual)

    cv2.imshow("Left Camera", rectified_left)
    cv2.imshow("Right Camera", rectified_right)
    cv2.imshow("Depth Map", disparity_visual)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_right.release()
cap_left.release()
cv2.destroyAllWindows()
