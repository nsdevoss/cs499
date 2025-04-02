import cv2
import queue
import numpy as np
import os
import time
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

        self.map1_x = None
        self.map1_y = None
        self.map2_x = None
        self.map2_y = None
        self.Q = None

        self.highlight_min_dist = self.vision_args.get("distance_args").get("min_dist")
        self.highlight_max_dist = self.vision_args.get("distance_args").get("max_dist")
        self.highlight_color = tuple(int(c) for c in self.vision_args.get("distance_args").get("color"))
        self.highlight_alpha = self.vision_args.get("distance_args").get("alpha")
        self.highlight_min_area = self.vision_args.get("distance_args").get("min_area")

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
                self.map1_x, self.map1_y, self.map2_x, self.map2_y, self.Q = self.generate_rectify_maps((mid, h), scale)
                server_logger.get_logger().info("Rectification maps generated")
            else:
                raise ValueError(f"cam_matrix is set to: {self.cam_matrix}, dist_coeffs is set to: {self.dist_coeffs}")

            prev_time = time.time()  # Store the initial timestamp
            frame_count = 0
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

                        left = cv2.remap(left_frame, self.map1_x, self.map1_y, cv2.INTER_LINEAR)
                        right = cv2.remap(right_frame, self.map2_x, self.map2_y, cv2.INTER_LINEAR)

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

                        distance_map = self.disparity_to_distance(filtered_disp)
                        highlighted_frame = self.highlight_distance_range(left, distance_map)

                        disp_colored = cv2.applyColorMap(disp_normalized.astype(np.uint8), cv2.COLORMAP_INFERNO)

                        if display_frame is None or display_frame.shape[1] != highlighted_frame.shape[1] + \
                                disp_colored.shape[1]:
                            display_frame = np.zeros((max(highlighted_frame.shape[0], disp_colored.shape[0]),
                                                      highlighted_frame.shape[1] + disp_colored.shape[1], 3),
                                                     dtype=np.uint8)

                        display_frame[:highlighted_frame.shape[0], :highlighted_frame.shape[1]] = highlighted_frame
                        display_frame[:disp_colored.shape[0],
                        highlighted_frame.shape[1]:highlighted_frame.shape[1] + disp_colored.shape[1]] = disp_colored

                        frame_count += 1
                        elapsed_time = time.time() - prev_time
                        if elapsed_time > 1:
                            fps = frame_count / elapsed_time
                            prev_time = time.time()
                            frame_count = 0

                            fps_text = f"FPS: {fps:.2f}"
                            cv2.putText(display_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
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

    def highlight_distance_range(self, frame, distance_map):
        """
        This function draws the colored outline on the image that tells us what is in range
        :param frame: This is the visual color normal frame, 3D array
        :param distance_map: This is the array that holds all of the distances 2D array
        :return: A visual frame with the green color on it of where the close section is
        """
        valid_pixels = ~np.isnan(distance_map)  # Here we make sure they are all valid

        range_mask = np.zeros_like(distance_map, dtype=np.uint8)  # We create an empty matrix with the same size and stuff as the map
        in_range = (distance_map >= self.highlight_min_dist) & (distance_map <= self.highlight_max_dist) & valid_pixels
        range_mask[in_range] = 1  # We populate each value that meets the criteria of our distance with 1 so we know what is in range

        # We find contours to filter out what we highlight, contours basically act as a filter as it makes areas of congested regions
        contours, _ = cv2.findContours(range_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        highlight_mask = np.zeros_like(range_mask)  # This mask is the same dimensions as the range 0,1 mask
        for contour in contours:
            if cv2.contourArea(contour) >= self.highlight_min_area:  # If the contour is big enough to be considered valid
                cv2.drawContours(highlight_mask, [contour], -1, self.highlight_color, 2)
                cv2.fillPoly(highlight_mask, [contour], self.highlight_color)

        highlight = np.zeros_like(frame)  # This is what we will display, we create a new matrix the same size as the orig frame
        highlight[highlight_mask > 0] = self.highlight_color  # We populate valid values with our color we choose

        return cv2.addWeighted(highlight, self.highlight_alpha, frame, 1 - self.highlight_alpha, 0)  # We overlay it onto the visual frame


    def disparity_to_distance(self, disparity):
        """
        This is a pretty simple function we just do a simple calculation to get the distance for each value in the disparity array
        :param disparity: The disparity map 2D
        :return: The distance map, a 2D array with each value being the distance at that point
        """
        valid_mask = (disparity > 0.1)  # We check for valid areas

        points_3d = cv2.reprojectImageTo3D(disparity, self.Q)  # Pretty self explanatory
        distance_map = points_3d[:, :, 2]  # We get the Z value from the 3D matrix
        distance_map[~valid_mask] = np.nan  # And we fill in the matrix with valid values

        return distance_map
