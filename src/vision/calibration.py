import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
from src.utils.utils import split_frame
import time

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
CALIBRATION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../calibrations"))

checkerboard_size = (8, 11)
square_size = 20.0


def calibrate(max_reprojection_error=1.0, chessboard_size=checkerboard_size, scale_factor=1.0):
    root = os.getcwd()
    calibration_dir = os.path.join(root, 'calibration', 'stereo')
    output_dir = os.path.dirname(os.path.abspath(__file__))

    os.makedirs(output_dir, exist_ok=True)

    imgPathList = glob.glob(os.path.join(calibration_dir, '*.jpg'))

    if not imgPathList:
        print(f"Error: No images found in {calibration_dir}")
        return None, None

    nRows, nCols = chessboard_size
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    world_points = np.zeros((nRows * nCols, 3), np.float32)
    world_points[:, :2] = np.mgrid[0:nRows, 0:nCols].T.reshape(-1, 2)
    worldPtsList = []
    imgPtsList = []
    successful_images = []

    print(f"Processing {len(imgPathList)} calibration images...")

    for idx, curImgPath in enumerate(imgPathList):
        print(f"Processing image {idx + 1}/{len(imgPathList)}: {os.path.basename(curImgPath)}")
        imgBGR = cv2.imread(curImgPath)
        if imgBGR is None:
            print(f"Warning: Could not read {curImgPath}")
            continue

        imgGray = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2GRAY)

        height, width = imgGray.shape
        left_imgGray = imgGray[:, :width // 2]

        for flag in [cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
                     cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK]:
            try:
                cornersFound, cornersOrg = cv2.findChessboardCorners(
                    left_imgGray, (nRows, nCols), flags=flag
                )
                if cornersFound:
                    break
            except Exception as e:
                print(f"Error during chessboard detection: {e}")
                cornersFound = False

        if cornersFound:
            cornersRefined = cv2.cornerSubPix(
                left_imgGray, cornersOrg, (11, 11), (-1, -1), criteria
            )

            worldPtsList.append(world_points)
            imgPtsList.append(cornersRefined)
            successful_images.append(curImgPath)
        else:
            print(f"Could not find chessboard in {os.path.basename(curImgPath)}")

    cv2.destroyAllWindows()
    print(f"Found chessboard in {len(worldPtsList)}/{len(imgPathList)} images")

    if len(worldPtsList) < 5:
        print("Error: Need at least 5 images with chessboards detected")
        return None, None

    flags = cv2.CALIB_RATIONAL_MODEL  # Bcz we have a wide FOV camera
    repError, camMatrix, distCoeff, rvecs, tvecs = cv2.calibrateCamera(
        worldPtsList, imgPtsList, left_imgGray.shape[::-1], None, None, flags=flags
    )

    print(f"Initial calibration complete with reprojection error: {repError:.4f} pixels")

    good_world_pts = []
    good_img_pts = []
    removed_count = 0
    good_images = []

    for i, (world_pts, img_pts, img_path) in enumerate(zip(worldPtsList, imgPtsList, successful_images)):
        projected_img_pts, _ = cv2.projectPoints(
            world_pts, rvecs[i], tvecs[i], camMatrix, distCoeff
        )

        error = cv2.norm(img_pts, projected_img_pts, cv2.NORM_L2) / len(img_pts)

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

    flags = cv2.CALIB_RATIONAL_MODEL

    if len(good_world_pts) > 10:
        flags = cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_THIN_PRISM_MODEL

    repError, camMatrix, distCoeff, rvecs, tvecs = cv2.calibrateCamera(
        good_world_pts, good_img_pts, left_imgGray.shape[::-1], None, None, flags=flags
    )

    print(f"Final reprojection error: {repError:.4f} pixels")
    print('Camera Matrix:\n', camMatrix)
    print('Distortion Coefficients:\n', distCoeff)

    image_errors = []
    for i in range(len(good_world_pts)):
        projected_img_pts, _ = cv2.projectPoints(
            good_world_pts[i], rvecs[i], tvecs[i], camMatrix, distCoeff
        )
        error = cv2.norm(good_img_pts[i], projected_img_pts, cv2.NORM_L2) / len(good_img_pts[i])
        image_errors.append((os.path.basename(good_images[i]), error))

    print("\nReprojection errors by image:")
    for img_name, error in sorted(image_errors, key=lambda x: x[1]):
        print(f"{img_name}: {error:.4f} pixels")

    left_size = (left_imgGray.shape[1], left_imgGray.shape[0])
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camMatrix, distCoeff, left_size, scale_factor, left_size
    )

    calibration_data = {
        'repError': repError,
        'camMatrix': camMatrix,
        'distCoeff': distCoeff,
        'rvecs': rvecs,
        'tvecs': tvecs,
        'good_images': good_images,
        'newCamMatrix': new_camera_matrix,
        'roi': roi,
        'imageSize': left_size
    }

    # Save as NPZ
    npz_path = os.path.join(output_dir, 'calibration_50.npz')
    np.savez(npz_path, **calibration_data)
    print(f"Calibration saved to {npz_path}")

    return camMatrix, distCoeff


def test_undistortion():
    calibration_path = os.path.join(CALIBRATION_DIR, "calib_50", "calibration_50.npz")
    if not os.path.exists(calibration_path):
        print(f"Calibration file not found at {calibration_path}")
        return

    data = np.load(calibration_path)
    camMatrix = data['camMatrix']
    distCoeff = data['distCoeff']
    newCamMatrix = data['newCamMatrix']
    test_img_path = os.path.join(ROOT_DIR, 'calibration', 'stereo', 'img_12.jpg')

    img = cv2.imread(test_img_path)
    if img is None:
        print(f"Could not read image {test_img_path}")
        return

    height, width = img.shape[:2]
    left_img = img[:, :width // 2]

    imgUndist = cv2.undistort(left_img, camMatrix, distCoeff, None, newCamMatrix)

    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.title("Original Left Image")
    plt.imshow(cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB))
    plt.subplot(122)
    plt.title("Undistorted Left Image")
    plt.imshow(cv2.cvtColor(imgUndist, cv2.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()


def get_calib_images(num_images=25):
    os.makedirs('calibration/stereo')

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return False

    count = 0

    while count < num_images:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to grab frame")
            break

        left_frame, right_frame = split_frame(frame)

        gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

        ret_left, _ = cv2.findChessboardCorners(gray_left, checkerboard_size,
                                                cv2.CALIB_CB_FAST_CHECK)
        ret_right, _ = cv2.findChessboardCorners(gray_right, checkerboard_size,
                                                 cv2.CALIB_CB_FAST_CHECK)

        vis_frame = frame.copy()

        print(f"Captured: {count}/{num_images}")

        cv2.imshow('Stereo Camera - Calibration Capture', vis_frame)

        key = cv2.waitKey(1) & 0xFF

        current_time = time.time()
        if key == ord('c'):
            if ret_left and ret_right:
                filename = f'calibration/stereo/img_{count:02d}.jpg'
                cv2.imwrite(filename, frame)

                print(f"Captured image {count + 1}/{num_images} - {filename}")
                count += 1
            else:
                print("Cannot capture: Checkerboard not detected in both views")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"Captured {count} images for calibration")


def run_calibration():
    camMatrix, distCoeff = calibrate(
        max_reprojection_error=0.2,
        chessboard_size=(8, 6),
        scale_factor=0.8,
    )
    return camMatrix, distCoeff


if __name__ == "__main__":
    get_calib_images(50)
    run_calibration()
    test_undistortion()
