import numpy as np
import cv2 as cv
import glob
import os
import matplotlib.pyplot as plt
from src.utils.utils import split_frame


def calibrate(showPics=True, max_reprojection_error=0.05):
    root = os.getcwd()
    calibrationDir = os.path.join(root, 'calibration//stereo')
    imgPathList = glob.glob(os.path.join(calibrationDir, '*.jpg'))

    nRows = 8
    nCols = 6
    termCriteris = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    worldPtsCur = np.zeros((nRows * nCols, 3), np.float32)
    worldPtsCur[:, :2] = np.mgrid[0:nRows, 0:nCols].T.reshape(-1, 2)
    worldPtsList = []
    imgPtsList = []
    successful_images = []

    print(f"Processing {len(imgPathList)} calibration images...")

    for curImgPath in imgPathList:
        imgBGR = cv.imread(curImgPath)
        imgGray = cv.cvtColor(imgBGR, cv.COLOR_BGR2GRAY)

        # Extract only the left half of the image
        height, width = imgGray.shape
        left_imgGray = imgGray[:, :width // 2]

        cornersFound, cornersOrg = cv.findChessboardCorners(left_imgGray, (nRows, nCols), None)

        if cornersFound:
            cornersRefined = cv.cornerSubPix(left_imgGray, cornersOrg, (11, 11), (-1, -1), termCriteris)
            worldPtsList.append(worldPtsCur)
            imgPtsList.append(cornersRefined)
            successful_images.append(curImgPath)

            if showPics:
                # Visualizing on left half of BGR image
                left_imgBGR = imgBGR[:, :width // 2]
                cv.drawChessboardCorners(left_imgBGR, (nRows, nCols), cornersRefined, cornersFound)
                cv.imshow("Chessboard - Left Camera", left_imgBGR)
                cv.waitKey(500)
    cv.destroyAllWindows()

    print(f"Found chessboard in {len(worldPtsList)}/{len(imgPathList)} images")

    if len(worldPtsList) < 3:
        print("Error: Not enough images with detected chessboards. Need at least 3.")
        return None, None

    # Initial calibration
    repError, camMatrix, distCoeff, rvecs, tvecs = cv.calibrateCamera(
        worldPtsList, imgPtsList, left_imgGray.shape[::-1], None, None
    )

    print(f"Initial calibration complete with reprojection error: {repError:.4f} pixels")

    # Filter out bad measurements
    good_world_pts = []
    good_img_pts = []
    removed_count = 0
    good_images = []

    for i, (world_pts, img_pts, img_path) in enumerate(zip(worldPtsList, imgPtsList, successful_images)):
        # Project world points to image points using calibration results
        projected_img_pts, _ = cv.projectPoints(world_pts, rvecs[i], tvecs[i], camMatrix, distCoeff)

        # Calculate error for this image
        error = cv.norm(img_pts, projected_img_pts, cv.NORM_L2) / len(img_pts)

        if error < max_reprojection_error:
            good_world_pts.append(world_pts)
            good_img_pts.append(img_pts)
            good_images.append(img_path)
        else:
            removed_count += 1
            print(f"Discarding image {os.path.basename(img_path)} with error {error:.4f}")

    if removed_count > 0:
        print(f"Removed {removed_count} images with high reprojection error")

    if len(good_world_pts) < 3:
        print("Error: Not enough good calibration images left after filtering. Using original calibration.")
        return camMatrix, distCoeff

    # Recalibrate with only good measurements
    repError, camMatrix, distCoeff, rvecs, tvecs = cv.calibrateCamera(
        good_world_pts, good_img_pts, left_imgGray.shape[::-1], None, None
    )

    print(f"Refined calibration complete with {len(good_world_pts)} images")
    print(f"Final reprojection error: {repError:.4f} pixels")
    print('Camera Matrix:\n', camMatrix)
    print('Distortion Coefficients:\n', distCoeff)

    # Calculate per-image reprojection errors for the final results
    image_errors = []
    for i in range(len(good_world_pts)):
        projected_img_pts, _ = cv.projectPoints(
            good_world_pts[i], rvecs[i], tvecs[i], camMatrix, distCoeff
        )
        error = cv.norm(good_img_pts[i], projected_img_pts, cv.NORM_L2) / len(good_img_pts[i])
        image_errors.append((os.path.basename(good_images[i]), error))

    print("\nReprojection errors by image:")
    for img_name, error in sorted(image_errors, key=lambda x: x[1]):
        print(f"{img_name}: {error:.4f} pixels")

    curFolder = os.path.dirname(os.path.abspath(__file__))
    paramPath = os.path.join(curFolder, 'calibration.npz')
    np.savez(paramPath, repError=repError, camMatrix=camMatrix, distCoeff=distCoeff,
             rvecs=rvecs, tvecs=tvecs, good_images=good_images)

    return camMatrix, distCoeff


def removeDistortion(camMatrix, distCoeff):
    imgpath = '/Users/nicholasburczyk/Desktop/CS CLASS/CS499sp25/cs499/src/vision/calibration/stereo/img_01.jpg'
    img = cv.imread(imgpath)

    if img is None:
        print(f"Error: Could not load image {imgpath}")
        return

    # Extract only the left half
    height, width = img.shape[:2]
    left_img = img[:, :width // 2]

    camMatrixNew, roi = cv.getOptimalNewCameraMatrix(camMatrix, distCoeff, (width // 2, height), 1,
                                                     (width // 2, height))
    imgUndist = cv.undistort(left_img, camMatrix, distCoeff, None, camMatrixNew)

    # Display original and undistorted images
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.title("Original Left Image")
    plt.imshow(cv.cvtColor(left_img, cv.COLOR_BGR2RGB))
    plt.subplot(122)
    plt.title("Undistorted Left Image")
    plt.imshow(cv.cvtColor(imgUndist, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()

    # Save the comparison image
    comparison = np.hstack((left_img, imgUndist))
    curFolder = os.path.dirname(os.path.abspath(__file__))
    cv.imwrite(os.path.join(curFolder, 'calibration_comparison.jpg'), comparison)


def runCalibtration():
    calibrate(showPics=True, max_reprojection_error=1.0)


def runRemoveDistortion():
    camMatrix, distCoeff = calibrate(showPics=False, max_reprojection_error=1.0)
    if camMatrix is not None and distCoeff is not None:
        removeDistortion(camMatrix, distCoeff)
    else:
        print("Calibration failed, cannot remove distortion")


if __name__ == "__main__":
    # runCalibtration()
    runRemoveDistortion()