import numpy as np
import cv2
import glob
import os
import time
import matplotlib.pyplot as plt
from vision import compute_disparity

checkerboard_size = (8, 6)  # 8x6 internal corners
square_size = 24.0  # 25mm


def split_stereo_frame(frame):
    height, width = frame.shape[:2]
    mid = width // 2
    left_frame = frame[:, :mid]
    right_frame = frame[:, mid:]
    return left_frame, right_frame


def calibrate_stereo_cameras(visualize=True, discard_threshold=1.0):
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp = objp * square_size

    objpoints = []
    imgpoints_left = []
    imgpoints_right = []

    successful_images = []

    calibration_images = sorted(glob.glob('calibration/stereo/*.jpg'))

    if len(calibration_images) == 0:
        print("No calibration images found!")
        return None

    print(f"Found {len(calibration_images)} stereo images")

    img = cv2.imread(calibration_images[0], cv2.IMREAD_COLOR)

    if img is None:
        print("Failed to load images")
        return None

    img_left, img_right = split_stereo_frame(img)
    img_size = (img_left.shape[1], img_left.shape[0])

    for img_path in calibration_images:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_left, img_right = split_stereo_frame(img)

        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        ret_left, corners_left = cv2.findChessboardCorners(gray_left, checkerboard_size,
                                                           cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                           cv2.CALIB_CB_NORMALIZE_IMAGE +
                                                           cv2.CALIB_CB_FAST_CHECK)

        ret_right, corners_right = cv2.findChessboardCorners(gray_right, checkerboard_size,
                                                             cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                             cv2.CALIB_CB_NORMALIZE_IMAGE +
                                                             cv2.CALIB_CB_FAST_CHECK)

        if ret_left and ret_right:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
            corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)

            if visualize:
                vis_left = img_left.copy()
                vis_right = img_right.copy()
                cv2.drawChessboardCorners(vis_left, checkerboard_size, corners_left, ret_left)
                cv2.drawChessboardCorners(vis_right, checkerboard_size, corners_right, ret_right)

                vis = np.concatenate((vis_left, vis_right), axis=1)
                cv2.putText(vis, f"Image: {os.path.basename(img_path)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow('Corner Detection', vis)
                key = cv2.waitKey(500)

                if key == ord('s'):
                    print(f"Skipping image: {os.path.basename(img_path)}")
                    continue
                elif key == ord('q'):
                    cv2.destroyAllWindows()
                    print("Calibration process canceled by user")
                    return None

            objpoints.append(objp)
            imgpoints_left.append(corners_left)
            imgpoints_right.append(corners_right)
            successful_images.append(img_path)

            print(f"Processed: {os.path.basename(img_path)}")
        else:
            print(f"Failed to find corners in {os.path.basename(img_path)}")

    cv2.destroyAllWindows()

    print(f"Successfully processed {len(objpoints)} images")

    if len(objpoints) < 10:
        print("Warning: Less than 10 valid images. Calibration may not be accurate.")
        if len(objpoints) < 5:
            print("Error: At least 5 valid images are required for calibration.")
            return None

    ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
        objpoints, imgpoints_left, img_size, None, None)
    ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
        objpoints, imgpoints_right, img_size, None, None)

    print(f"Left camera RMS re-projection error: {ret_left}")
    print(f"Right camera RMS re-projection error: {ret_right}")

    print("Evaluating individual image quality...")
    left_errors = []
    right_errors = []

    for i in range(len(objpoints)):
        imgpoints2_left, _ = cv2.projectPoints(objpoints[i], rvecs_left[i], tvecs_left[i],
                                               mtx_left, dist_left)
        error_left = cv2.norm(imgpoints_left[i], imgpoints2_left, cv2.NORM_L2) / len(imgpoints2_left)
        left_errors.append(error_left)

        imgpoints2_right, _ = cv2.projectPoints(objpoints[i], rvecs_right[i], tvecs_right[i],
                                                mtx_right, dist_right)
        error_right = cv2.norm(imgpoints_right[i], imgpoints2_right, cv2.NORM_L2) / len(imgpoints2_right)
        right_errors.append(error_right)

        print(
            f"Image {i}: Left error = {error_left:.4f}, Right error = {error_right:.4f} - {os.path.basename(successful_images[i])}")

    high_error_indices = []
    for i in range(len(left_errors)):
        if left_errors[i] > discard_threshold or right_errors[i] > discard_threshold:
            high_error_indices.append(i)

    if high_error_indices:
        print(f"Found {len(high_error_indices)} images with high error (>{discard_threshold}).")

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(left_errors)), left_errors, alpha=0.5, color='b', label='Left')
        plt.bar(range(len(right_errors)), right_errors, alpha=0.5, color='r', label='Right')
        plt.axhline(y=discard_threshold, color='g', linestyle='--', label=f'Threshold ({discard_threshold})')
        plt.xlabel('Image Index')
        plt.ylabel('Reprojection Error')
        plt.title('Reprojection Error for Each Image')
        plt.legend()
        plt.tight_layout()
        plt.savefig('calibration_errors.png')

        discard = input(f"Do you want to discard {len(high_error_indices)} images with high error? (y/n): ")

        if discard.lower() == 'y':
            for idx in sorted(high_error_indices, reverse=True):
                print(
                    f"Discarding image with error L:{left_errors[idx]:.4f}, R:{right_errors[idx]:.4f} - {os.path.basename(successful_images[idx])}")
                objpoints.pop(idx)
                imgpoints_left.pop(idx)
                imgpoints_right.pop(idx)
                successful_images.pop(idx)

            # Recalibrate
            print("Recalibrating with filtered dataset...")
            ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
                objpoints, imgpoints_left, img_size, None, None)
            ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
                objpoints, imgpoints_right, img_size, None, None)

            print(f"After filtering - Left camera RMS error: {ret_left}")
            print(f"After filtering - Right camera RMS error: {ret_right}")

    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

    ret_stereo, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        mtx_left, dist_left,
        mtx_right, dist_right,
        img_size, criteria=criteria_stereo, flags=flags)

    print(f"Stereo calibration RMS re-projection error: {ret_stereo}")

    rectification_alpha = 0.5  # Try different values between 0 and 1

    rect_left, rect_right, proj_left, proj_right, Q, roi_left, roi_right = cv2.stereoRectify(
        mtx_left, dist_left,
        mtx_right, dist_right,
        img_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=rectification_alpha)

    print(f"Left ROI: {roi_left}")
    print(f"Right ROI: {roi_right}")

    map_left_x, map_left_y = cv2.initUndistortRectifyMap(
        mtx_left, dist_left, rect_left, proj_left, img_size, cv2.CV_32FC1)
    map_right_x, map_right_y = cv2.initUndistortRectifyMap(
        mtx_right, dist_right, rect_right, proj_right, img_size, cv2.CV_32FC1)

    calibration_data = {
        'image_size': img_size,
        'camera_matrix_left': mtx_left.tolist(),
        'dist_coeffs_left': dist_left.tolist(),
        'camera_matrix_right': mtx_right.tolist(),
        'dist_coeffs_right': dist_right.tolist(),
        'R': R.tolist(),
        'T': T.tolist(),
        'R1': rect_left.tolist(),
        'R2': rect_right.tolist(),
        'P1': proj_left.tolist(),
        'P2': proj_right.tolist(),
        'Q': Q.tolist(),
        'map_left_x': map_left_x,
        'map_left_y': map_left_y,
        'map_right_x': map_right_x,
        'map_right_y': map_right_y,
        'roi_left': roi_left,
        'roi_right': roi_right,
        'baseline_mm': abs(T[0]) * 1000,  # Convert to mm
        'rectification_alpha': rectification_alpha
    }

    np.save('stereo_calibration.npy', calibration_data)

    fs = cv2.FileStorage('stereo_calibration.xml', cv2.FILE_STORAGE_WRITE)
    fs.write('image_width', img_size[0])
    fs.write('image_height', img_size[1])
    fs.write('camera_matrix_left', mtx_left)
    fs.write('dist_coeffs_left', dist_left)
    fs.write('camera_matrix_right', mtx_right)
    fs.write('dist_coeffs_right', dist_right)
    fs.write('R', R)
    fs.write('T', T)
    fs.write('R1', rect_left)
    fs.write('R2', rect_right)
    fs.write('P1', proj_left)
    fs.write('P2', proj_right)
    fs.write('Q', Q)
    fs.release()

    if len(successful_images) > 0:
        test_rectification(successful_images[0], calibration_data)

    print("Calibration complete and files saved!")
    return calibration_data


def test_rectification(image_path, calibration_data):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load test image: {image_path}")
        return

    left_frame, right_frame = split_stereo_frame(img)

    map_left_x = calibration_data['map_left_x']
    map_left_y = calibration_data['map_left_y']
    map_right_x = calibration_data['map_right_x']
    map_right_y = calibration_data['map_right_y']
    roi_left = calibration_data['roi_left']
    roi_right = calibration_data['roi_right']

    # Rectify images
    rectified_left = cv2.remap(left_frame, map_left_x, map_left_y, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(right_frame, map_right_x, map_right_y, cv2.INTER_LINEAR)

    x, y, w, h = roi_left
    cv2.rectangle(rectified_left, (x, y), (x + w, y + h), (0, 255, 0), 2)
    x, y, w, h = roi_right
    cv2.rectangle(rectified_right, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for i in range(0, rectified_left.shape[0], 50):
        cv2.line(rectified_left, (0, i), (rectified_left.shape[1], i), (0, 255, 0), 1)
        cv2.line(rectified_right, (0, i), (rectified_right.shape[1], i), (0, 255, 0), 1)

    original = np.concatenate((left_frame, right_frame), axis=1)
    rectified = np.concatenate((rectified_left, rectified_right), axis=1)

    comparison = np.concatenate((original, rectified), axis=0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, "Original", (10, 30), font, 1, (0, 0, 255), 2)
    cv2.putText(comparison, "Rectified", (10, original.shape[0] + 30), font, 1, (0, 0, 255), 2)

    cv2.imwrite('rectification_test.jpg', comparison)

    height, width = comparison.shape[:2]
    max_height = 900
    if height > max_height:
        scale = max_height / height
        comparison = cv2.resize(comparison, None, fx=scale, fy=scale)

    cv2.imshow('Rectification Test', comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def capture_calibration_images(cam_id=0, num_images=25, delay=2):
    os.makedirs('calibration/stereo', exist_ok=True)

    cap = cv2.VideoCapture(cam_id)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return False

    count = 0
    last_capture_time = time.time() - delay

    print("\n=== CHECKERBOARD CAPTURE GUIDELINES ===")
    print("1. Hold the checkerboard so it's FULLY VISIBLE in both camera views")
    print("2. Vary positions to cover the entire field of view")
    print("3. Include positions at different distances (near, medium, far)")
    print("4. Rotate the checkerboard in different orientations")
    print("5. Include some tilted positions (but not too extreme)")
    print("6. Avoid motion blur - hold steady when capturing")
    print("7. Ensure good, even lighting")
    print("8. Make sure the checkerboard is flat and not bent")
    print("=== CONTROLS ===")
    print("- Press 'c' to capture an image")
    print("- Press 'd' to delete last captured image")
    print("- Press 'q' to quit")
    print("=======================================\n")

    while count < num_images:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to grab frame")
            break

        left_frame, right_frame = split_stereo_frame(frame)

        gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

        ret_left, _ = cv2.findChessboardCorners(gray_left, checkerboard_size,
                                                cv2.CALIB_CB_FAST_CHECK)
        ret_right, _ = cv2.findChessboardCorners(gray_right, checkerboard_size,
                                                 cv2.CALIB_CB_FAST_CHECK)

        vis_frame = frame.copy()

        status = "READY TO CAPTURE" if ret_left and ret_right else "CHECKERBOARD NOT DETECTED"
        color = (0, 255, 0) if ret_left and ret_right else (0, 0, 255)

        cv2.putText(vis_frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(vis_frame, f"Captured: {count}/{num_images}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow('Stereo Camera - Calibration Capture', vis_frame)

        key = cv2.waitKey(1) & 0xFF

        current_time = time.time()
        if (key == ord('c')) and (current_time - last_capture_time >= delay):
            if ret_left and ret_right:
                filename = f'calibration/stereo/img_{count:02d}.jpg'
                cv2.imwrite(filename, frame)

                print(f"Captured image {count + 1}/{num_images} - {filename}")
                count += 1
                last_capture_time = current_time
            else:
                print("Cannot capture: Checkerboard not detected in both views")

        elif key == ord('d') and count > 0:
            count -= 1
            filename = f'calibration/stereo/img_{count:02d}.jpg'
            if os.path.exists(filename):
                os.remove(filename)
                print(f"Deleted last image: {filename}")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"Captured {count} images for calibration")
    return True


def live_rectification(cam_id=0, alpha=None):
    try:
        calibration_data = np.load('stereo_calibration.npy', allow_pickle=True).item()

        if alpha is not None and (0 <= alpha <= 1):
            print(f"Adjusting rectification with alpha = {alpha}")
            img_size = calibration_data['image_size']
            mtx_left = np.array(calibration_data['camera_matrix_left'])
            dist_left = np.array(calibration_data['dist_coeffs_left'])
            mtx_right = np.array(calibration_data['camera_matrix_right'])
            dist_right = np.array(calibration_data['dist_coeffs_right'])
            R = np.array(calibration_data['R'])
            T = np.array(calibration_data['T'])

            rect_left, rect_right, proj_left, proj_right, Q, roi_left, roi_right = cv2.stereoRectify(
                mtx_left, dist_left, mtx_right, dist_right, img_size, R, T,
                flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha)

            map_left_x, map_left_y = cv2.initUndistortRectifyMap(
                mtx_left, dist_left, rect_left, proj_left, img_size, cv2.CV_32FC1)
            map_right_x, map_right_y = cv2.initUndistortRectifyMap(
                mtx_right, dist_right, rect_right, proj_right, img_size, cv2.CV_32FC1)

            calibration_data['R1'] = rect_left.tolist()
            calibration_data['R2'] = rect_right.tolist()
            calibration_data['P1'] = proj_left.tolist()
            calibration_data['P2'] = proj_right.tolist()
            calibration_data['Q'] = Q.tolist()
            calibration_data['map_left_x'] = map_left_x
            calibration_data['map_left_y'] = map_left_y
            calibration_data['map_right_x'] = map_right_x
            calibration_data['map_right_y'] = map_right_y
            calibration_data['roi_left'] = roi_left
            calibration_data['roi_right'] = roi_right
            calibration_data['rectification_alpha'] = alpha

            np.save('stereo_calibration.npy', calibration_data)

            print(f"Updated rectification maps with alpha = {alpha}")
            print(f"New Left ROI: {roi_left}")
            print(f"New Right ROI: {roi_right}")

        map_left_x = calibration_data['map_left_x']
        map_left_y = calibration_data['map_left_y']
        map_right_x = calibration_data['map_right_x']
        map_right_y = calibration_data['map_right_y']
        roi_left = calibration_data['roi_left']
        roi_right = calibration_data['roi_right']

        cap = cv2.VideoCapture(cam_id)

        if not cap.isOpened():
            print("Error: Could not open camera.")
            return

        show_lines = True
        show_roi = True

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: Failed to grab frame")
                break

            left_frame, right_frame = split_stereo_frame(frame)

            rectified_left = cv2.remap(left_frame, map_left_x, map_left_y, cv2.INTER_LINEAR)
            rectified_right = cv2.remap(right_frame, map_right_x, map_right_y, cv2.INTER_LINEAR)

            vis_left = rectified_left.copy()
            vis_right = rectified_right.copy()

            disp = compute_disparity(vis_left, vis_right)
            cv2.imshow("Disparity", disp)

            if show_roi:
                x, y, w, h = roi_left
                cv2.rectangle(vis_left, (x, y), (x + w, y + h), (0, 255, 0), 2)
                x, y, w, h = roi_right
                cv2.rectangle(vis_right, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if show_lines:
                for i in range(0, vis_left.shape[0], 50):
                    cv2.line(vis_left, (0, i), (vis_left.shape[1], i), (0, 255, 0), 1)
                    cv2.line(vis_right, (0, i), (vis_right.shape[1], i), (0, 255, 0), 1)

            vis = np.concatenate((vis_left, vis_right), axis=1)

            cv2.putText(vis, "L: Toggle lines | R: Toggle ROI | Q: Quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            alpha_val = calibration_data.get('rectification_alpha', 'N/A')
            cv2.putText(vis, f"Alpha: {alpha_val}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow('Original Stereo Feed', frame)
            cv2.imshow('Rectified Stereo Feed', vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('l'):
                show_lines = not show_lines
                print(f"Epipolar lines: {'On' if show_lines else 'Off'}")
            elif key == ord('r'):
                show_roi = not show_roi
                print(f"ROI display: {'On' if show_roi else 'Off'}")

        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error in live rectification: {e}")


# Main execution
if __name__ == "__main__":
    print("Stereo Camera Calibration Tool")
    print("1. Capture calibration images")
    print("2. Run calibration")
    print("3. Show live rectified feed")
    print("4. Adjust rectification (alpha parameter)")
    choice = input("Enter your choice (1/2/3/4): ")

    if choice == '1':
        cam_id = int(input("Enter camera ID (default 0): ") or 0)
        num_imgs = int(input("Number of images to capture (default 25): ") or 25)
        capture_calibration_images(cam_id, num_imgs)
    elif choice == '2':
        visualize = input("Visualize corner detection? (y/n, default: y): ").lower() != 'n'
        threshold = float(input("Discard threshold for reprojection error (default: 1.0): ") or 1.0)
        calibrate_stereo_cameras(visualize=visualize, discard_threshold=threshold)
    elif choice == '3':
        cam_id = int(input("Enter camera ID (default 0): ") or 0)
        live_rectification(cam_id=cam_id)
    elif choice == '4':
        alpha = float(input("Enter alpha value (0-1, where 0=full crop, 1=no crop): "))
        cam_id = int(input("Enter camera ID (default 0): ") or 0)
        live_rectification(cam_id=cam_id, alpha=alpha)
    else:
        print("Invalid choice!")