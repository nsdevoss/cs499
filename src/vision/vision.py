import cv2
import queue
import numpy as np
import os
from src.server.logger import server_logger

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
CALIBRATION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../calibrations"))


class Vision:
    """
    The vision class that will handle all the calculations.

    Params:
    :param frame_queue: Frame queue containing frames from the cameras that are used FOR THE ALGORITHM ONLY!!!
    :param vison_args: Dictionary defining the parameters mainly for the depth alrogithm
    :param display_queue: This is the global queue used for DISPLAYING THE FRAMES ONLY!!!
    """

    def __init__(self, frame_queue, display_queue, vision_args):
        self.frame_queue = frame_queue
        self.display_queue = display_queue
        self.vision_args = vision_args
        self.calibration_file = os.path.join(CALIBRATION_DIR, self.vision_args.get("calibration_file"))
        # Depth args
        self.display_frame = None
        self.cam_matrix = None
        self.dist_coeffs = None

    def start(self):
        if self.vision_args.get("enabled"):
            self.depth_estimation()
        else:
            return

    def depth_estimation(self):
        try:
            scale = self.vision_args.get("scale", 1.0)
            stereo = cv2.StereoSGBM_create(
                minDisparity=self.vision_args.get("StereoSGBM_args").get("minDisparity"),
                numDisparities=self.vision_args.get("StereoSGBM_args").get("numDisparities"),
                blockSize=self.vision_args.get("StereoSGBM_args").get("blockSize"),
                uniquenessRatio=self.vision_args.get("StereoSGBM_args").get("uniquenessRatio"),
                speckleWindowSize=self.vision_args.get("StereoSGBM_args").get("speckleWindowSize"),
                speckleRange=self.vision_args.get("StereoSGBM_args").get("speckleRange"),
                disp12MaxDiff=self.vision_args.get("StereoSGBM_args").get("disp12MaxDiff"),
                P1=8 * 3 * self.vision_args.get("StereoSGBM_args").get("blockSize") ** 2,
                P2=32 * 3 * self.vision_args.get("StereoSGBM_args").get("blockSize") ** 2,
                mode=cv2.STEREO_SGBM_MODE_HH
            )
            self.load_calibration(scale)

            right_matcher = cv2.ximgproc.createRightMatcher(stereo)
            wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo)
            wls_filter.setLambda(8000)
            wls_filter.setSigmaColor(1.5)

            frame = self.frame_queue.get()
            print(frame.shape)
            h, w = frame.shape[:2]
            mid = w // 2
            if self.cam_matrix is not None and self.dist_coeffs is not None:
                map1_x, map1_y, map2_x, map2_y, Q = self.generate_rectify_maps((mid, h), scale)
                server_logger.get_logger().info("Rectification maps generated")
            else:
                raise ValueError(f"cam_matrix is set to: {self.cam_matrix}, dist_coeffs is set to: {self.dist_coeffs}")

            while True:
                try:
                    frame = self.frame_queue.get()
                    display_frame = None
                    if frame is not None:
                        h, w = frame.shape[:2]
                        mid = w // 2
                        left_frame, right_frame = frame[:, :mid], frame[:, mid:]

                        left_frame = cv2.resize(left_frame, (int(mid * scale), int(h * scale)))
                        right_frame = cv2.resize(right_frame, (int(mid * scale), int(h * scale)))

                        left = cv2.remap(left_frame, map1_x, map1_y, cv2.INTER_LINEAR)
                        right = cv2.remap(right_frame, map2_x, map2_y, cv2.INTER_LINEAR)

                        gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
                        gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

                        disp_left = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
                        disp_right = right_matcher.compute(gray_right, gray_left).astype(np.float32) / 16.0

                        filtered_disp = wls_filter.filter(disp_left, gray_left, disparity_map_right=disp_right)
                        filtered_disp = cv2.resize(filtered_disp, (gray_left.shape[1], gray_left.shape[0]),
                                                   interpolation=cv2.INTER_LINEAR)

                        valid_mask = ~(np.isnan(filtered_disp) | np.isinf(filtered_disp) | (filtered_disp < 0))
                        filtered_disp = filtered_disp * valid_mask

                        min_val = np.min(filtered_disp[valid_mask]) if np.any(valid_mask) else 0
                        max_val = np.max(filtered_disp[valid_mask]) if np.any(valid_mask) else 1

                        if max_val - min_val < 1:
                            max_val = min_val + 1

                        disp_normalized = np.zeros_like(filtered_disp)
                        disp_normalized[valid_mask] = (
                                (filtered_disp[valid_mask] - min_val) / (max_val - min_val) * 255).astype(
                            np.uint8)

                        disp_colored = cv2.applyColorMap(disp_normalized.astype(np.uint8), cv2.COLORMAP_INFERNO)

                        if display_frame is None or display_frame.shape[1] != left.shape[1] + disp_colored.shape[1]:
                            display_frame = np.zeros((max(left.shape[0], disp_colored.shape[0]),
                                                      left.shape[1] + disp_colored.shape[1], 3), dtype=np.uint8)

                        display_frame[:left.shape[0], :left.shape[1]] = left
                        display_frame[:disp_colored.shape[0], left.shape[1]:left.shape[1] + disp_colored.shape[1]] = disp_colored

                        if self.display_queue is not None:
                            self.display_queue.put((frame, display_frame))

                        del frame, disp_colored, display_frame


                except queue.Empty:
                    server_logger.get_logger().warning("Frame queue is empty.")
                    break

            cv2.destroyAllWindows()
        except Exception as e:
            server_logger.get_logger().error(f"Error during depth estimation: {e}")
            return

    def object_detect(self):
        pass

    def export(self):
        # Womp womp
        return

    def load_calibration(self, scale=1.0):
        if not os.path.exists(self.calibration_file):
            server_logger.get_logger().error(f"No valid calibration file found: {self.calibration_file}")
        try:
            data = np.load(self.calibration_file)
            cam_matrix = data['camMatrix']
            dist_coeffs = data['distCoeff']
            if scale != 1.0:
                server_logger.get_logger().info(f"Scaling calibration matrix by factor: {scale}")

                cam_matrix[0, 0] *= scale  # fx
                cam_matrix[1, 1] *= scale  # fy
                cam_matrix[0, 2] *= scale  # cx
                cam_matrix[1, 2] *= scale  # cy

            self.cam_matrix = cam_matrix
            self.dist_coeffs = dist_coeffs
            server_logger.get_logger().info(f"Calibration file loaded and scaled: {self.calibration_file}")
        except Exception as e:
            server_logger.get_logger().error(f"Error loading calibration file: {e}")

    def generate_rectify_maps(self, image_size, scale=1.0):
        w, h = image_size

        w = int(w * scale)
        h = int(h * scale)

        R = np.eye(3)
        T = np.array([self.vision_args.get("camera_parameters").get("baseline"), 0, 0])

        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            self.cam_matrix, self.dist_coeffs,
            self.cam_matrix, self.dist_coeffs,
            (w, h), R, T, alpha=0
        )
        Q[3, 2] = np.abs(Q[3, 2])

        map1_x, map1_y = cv2.initUndistortRectifyMap(self.cam_matrix, self.dist_coeffs, R1, P1, (w, h), cv2.CV_32FC1)
        map2_x, map2_y = cv2.initUndistortRectifyMap(self.cam_matrix, self.dist_coeffs, R2, P2, (w, h), cv2.CV_32FC1)

        return map1_x, map1_y, map2_x, map2_y, Q

def detect_close_objects(disp_map, threshold_percent=0.8, min_area=500, max_area=100000):
        roi_image = cv2.cvtColor(disp_map, cv2.COLOR_GRAY2BGR) if len(disp_map.shape) == 2 else disp_map.copy()

        # Higher values in disparity map = closer objects
        threshold = int(255 * threshold_percent)
        _, binary = cv2.threshold(disp_map if len(disp_map.shape) == 2 else cv2.cvtColor(disp_map, cv2.COLOR_BGR2GRAY),
                                  threshold, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rois = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                rois.append((x, y, w, h))
                cv2.rectangle(roi_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(roi_image, "CLOSE", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return roi_image, rois
