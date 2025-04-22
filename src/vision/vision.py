import math
import time
import cv2
import queue
import os
import uuid
import numpy as np
import src.LocalCommon as lc
from ultralytics import YOLO
from src.server.logger import server_logger
from collections import deque


class Vision:
    """
    The vision class that will handle all the calculations.

    Params:
    :param frame_queue: Frame queue containing frames from the cameras that are used FOR THE ALGORITHM ONLY!!!
    :param vison_args: Dictionary defining the parameters mainly for the depth alrogithm
    :param display_queue: This is the global queue used for DISPLAYING THE FRAMES ONLY!!!
    """

    def __init__(self, frame_queue, display_queue, info_queue, object_detect_queue, vision_args, scale):
        self.frame_queue = frame_queue
        self.display_queue = display_queue
        self.info_queue = info_queue
        self.vision_args = vision_args
        self.scale = scale
        self.percent_scaled_down = vision_args.get("distance_args").get("percent_border_scaled_down")
        self.calibration_file = os.path.join(lc.CALIBRATION_DIR, self.vision_args.get("calibration_file"))
        self.model_file = os.path.join(lc.MODEL_DIR, self.vision_args.get("model_file"))
        # Depth args
        self.display_frame = None
        self.cam_matrix = None
        self.dist_coeffs = None

        self.map1_x = None
        self.map1_y = None
        self.map2_x = None
        self.map2_y = None
        self.Q = None

        self.point_cloud_refresh_rate = self.vision_args.get("3d_render_args").get("refresh_rate")

        self.highlight_min_dist = self.vision_args.get("distance_args").get("min_dist")
        self.highlight_max_dist = self.vision_args.get("distance_args").get("max_dist")
        self.highlight_color = tuple(int(c) for c in self.vision_args.get("distance_args").get("color"))
        self.highlight_alpha = self.vision_args.get("distance_args").get("alpha")
        self.highlight_min_area = self.vision_args.get("distance_args").get("min_area")

        self.object_detect_queue = object_detect_queue
        self.object_queue = deque(maxlen=50)
        self.previous_objects = []
        self.object_persistence_threshold = self.vision_args.get("distance_args").get("object_persistence_threshold")

        if self.vision_args.get("object_determination_enabled"):
            self.model = YOLO(self.vision_args.get("model_file"))

    def start(self):
        self.depth_estimation()

    def depth_estimation(self):
        try:
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
            self.load_calibration(self.scale)

            right_matcher = cv2.ximgproc.createRightMatcher(stereo)
            wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo)
            wls_filter.setLambda(8000)
            wls_filter.setSigmaColor(1.5)

            h, w = (720, 2560)
            mid = w // 2
            if self.cam_matrix is not None and self.dist_coeffs is not None:
                self.map1_x, self.map1_y, self.map2_x, self.map2_y, self.Q = self.generate_rectify_maps((mid, h),
                                                                                                        self.scale)
                server_logger.get_logger().info("Rectification maps generated")
            else:
                raise ValueError(f"cam_matrix is set to: {self.cam_matrix}, dist_coeffs is set to: {self.dist_coeffs}")

            contour_refresh_map = time.time()
            frame = self.frame_queue.get()
            h, w = frame.shape[:2]
            mid = w // 2
            left_frame = frame[:, :mid]
            contour_map = np.zeros_like(left_frame)

            visualization_time = time.time()
            while True:
                try:
                    frame = self.frame_queue.get()
                    display_frame = None
                    if frame is not None:
                        h, w = frame.shape[:2]
                        mid = w // 2
                        left_frame, right_frame = frame[:, :mid], frame[:, mid:]

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

                        distance_map, points_3d, valid_dist_mask = self.disparity_to_distance(filtered_disp)
                        highlighted_frame, contour_centers, contour_map, object_in_frame = self.highlight_distance_range(left,
                                                                                                        distance_map,
                                                                                                        contour_map)
                            
                        disp_colored = cv2.applyColorMap(disp_normalized.astype(np.uint8), cv2.COLORMAP_INFERNO)

                        if display_frame is None or display_frame.shape[1] != highlighted_frame.shape[1] + \
                                disp_colored.shape[1]:
                            display_frame = np.zeros(
                                (max(highlighted_frame.shape[0], disp_colored.shape[0]) * 3,
                                 highlighted_frame.shape[1] + disp_colored.shape[1], 3),
                                dtype=np.uint8
                            )

                        display_frame[:highlighted_frame.shape[0],
                        :highlighted_frame.shape[1]] = highlighted_frame  # Top-left
                        display_frame[:disp_colored.shape[0],
                        highlighted_frame.shape[1]:highlighted_frame.shape[1] + disp_colored.shape[
                            1]] = disp_colored  # Top-right

                        # Second row (middle)
                        middle_y_offset = highlighted_frame.shape[0]
                        display_frame[middle_y_offset:middle_y_offset + contour_centers.shape[0],
                        :contour_centers.shape[1]] = contour_centers
                        display_frame[middle_y_offset:middle_y_offset + contour_map.shape[0],
                        highlighted_frame.shape[1]:highlighted_frame.shape[1] + contour_map.shape[
                            1]] = contour_map

                        # Third row (bottom) - original frames
                        bottom_y_offset = middle_y_offset + highlighted_frame.shape[0]
                        display_frame[bottom_y_offset:bottom_y_offset + left.shape[0],
                        :left.shape[1]] = left_frame
                        display_frame[bottom_y_offset:bottom_y_offset + right.shape[0],
                        highlighted_frame.shape[1]:highlighted_frame.shape[1] + right.shape[
                            1]] = right_frame

                        if self.display_queue is not None:
                            self.display_queue.put(display_frame)

                        end_time = time.time()
                        if end_time - contour_refresh_map >= 10.0:
                            contour_map = np.zeros_like(left_frame)
                            contour_refresh_map = end_time

                        if end_time - visualization_time >= self.point_cloud_refresh_rate and self.info_queue is not None:
                            self.info_queue.put((left, points_3d, valid_dist_mask))
                            visualization_time = end_time

                        del frame, disp_colored, display_frame

                except queue.Empty:
                    server_logger.get_logger().warning("Frame queue is empty.")
                    break

        except Exception as e:
            server_logger.get_logger().error(f"Error during depth estimation: {e}")
            return

    def load_calibration(self, scale=1.0):
        """
        This function loads the calibration file we generated and its parameters we set.
        :param scale: We have to take into account the scale of the images for the matrix
        """
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
        T = np.array([lc.CAMERA_BASELINE, 0, 0])

        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            self.cam_matrix, self.dist_coeffs,
            self.cam_matrix, self.dist_coeffs,
            (w, h), R, T, alpha=0
        )
        Q[3, 2] = np.abs(Q[3, 2])

        map1_x, map1_y = cv2.initUndistortRectifyMap(self.cam_matrix, self.dist_coeffs, R1, P1, (w, h), cv2.CV_32FC1)
        map2_x, map2_y = cv2.initUndistortRectifyMap(self.cam_matrix, self.dist_coeffs, R2, P2, (w, h), cv2.CV_32FC1)

        return map1_x, map1_y, map2_x, map2_y, Q

    def highlight_distance_range(self, frame, distance_map, contour_map):
        """
        This function draws the colored outline on the image that tells us what is in range
        :param frame: This is the visual color normal frame, 3D array
        :param distance_map: This is the array that holds all of the distances 2D array
        :return: A visual frame with the green color on it of where the close section is
        """
        valid_pixels = ~np.isnan(distance_map)  # Here we make sure they are all valid
        kernel = np.ones((5, 5), np.uint8)

        range_mask = np.zeros_like(distance_map,
                                   dtype=np.uint8)  # We create an empty matrix with the same size and stuff as the map
        in_range = (distance_map >= self.highlight_min_dist) & (distance_map <= self.highlight_max_dist) & valid_pixels
        range_mask[
            in_range] = 255
        range_mask = cv2.morphologyEx(range_mask, cv2.MORPH_CLOSE, kernel)
        range_mask = cv2.morphologyEx(range_mask, cv2.MORPH_OPEN, kernel)

        height, width = range_mask.shape
        margin_x = int(width * self.percent_scaled_down * 0.5)
        margin_y = int(height * self.percent_scaled_down * 0.5)

        middle_mask = np.zeros_like(range_mask)
        middle_mask[margin_y:height - margin_y, margin_x:width - margin_x] = 255

        range_mask = cv2.bitwise_and(range_mask, middle_mask)
        # We find contours to filter out what we highlight, contours basically act as a filter as it makes areas of congested regions
        contours, _ = cv2.findContours(range_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        highlight_mask = np.zeros_like(range_mask)
        highlight_mask_colored = np.zeros_like(frame)
        highlight_annotations = np.zeros_like(frame)
        contour_centers = np.zeros_like(frame)
        object_found = False
        current_objects = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= self.highlight_min_area:  # If the contour is big enough to be considered valid
                x, y, w, h = cv2.boundingRect(contour)
                moments = cv2.moments(contour)
                if moments['m00'] != 0:
                    Cx = int(moments['m10'] / moments['m00'])
                    Cy = int(moments['m01'] / moments['m00'])

                    region_size = 5
                    region_x_start = max(0, Cx - region_size)
                    region_x_end = min(distance_map.shape[1] - 1, Cx + region_size)
                    region_y_start = max(0, Cy - region_size)
                    region_y_end = min(distance_map.shape[0] - 1, Cy + region_size)
                    region = distance_map[region_y_start:region_y_end, region_x_start:region_x_end]
                    valid_region = region[~np.isnan(region)]

                    if len(valid_region) > 0:
                        object_distance = np.median(valid_region)

                        current_objects.append({
                            'center': (Cx, Cy),
                            'bbox': (x, y, w, h),
                            'distance': float(object_distance),
                            'area': area,
                            'contour': contour
                        })

                        object_found = True
                        contour_centers = cv2.circle(contour_centers, (Cx, Cy), radius=4, color=(0, 255, 0), thickness=-1)
                        contour_map = cv2.circle(contour_map, (Cx, Cy), radius=3, color=(0, 255, 0), thickness=-1)

                cv2.drawContours(highlight_mask, [contour], -1, 255, -1)

        # Match with previous objects
        for current_obj in current_objects:
            cx, cy = current_obj['center']
            matched = False
            for prev_obj in self.previous_objects:
                px, py = prev_obj['center']
                dist = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
                if dist < 50:
                    current_obj['persistence'] = prev_obj.get('persistence', 0) + 1
                    matched = True
                    break
            if not matched:
                current_obj['persistence'] = 1

        stable_objects = [obj for obj in current_objects if obj.get('persistence', 0) >= 2]

        for obj in stable_objects:
            x, y, w, h = obj['bbox']
            cx, cy = obj['center']
            distance = obj['distance']

            if cx >= 170:
                position = 'right'
            elif cx <= 85:
                position = 'left'
            else:
                position = 'center'

            object_info = {
                'center': (cx, cy),
                'position': position,
                'distance': distance,
                'bbox': (x, y, w, h),
                'persistence': obj.get('persistence')
            }

            self.object_queue.append(object_info)

            if self.object_detect_queue is not None and obj.get('persistence', 0) >= self.object_persistence_threshold:
                try:
                    self.object_detect_queue.put_nowait(object_info)
                    cv2.putText(highlight_annotations, f"{distance:.2f}m", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.drawContours(highlight_annotations, [obj['contour']], -1, self.highlight_color, 1)
                    cv2.rectangle(highlight_annotations, (x, y), (x + w, y + h), (255, 255, 255), 2)
                except queue.Full:
                    server_logger.get_logger().warning("object_detect_queue is full. Dropping object.")

        cv2.rectangle(highlight_annotations, (int(frame.shape[1] * self.percent_scaled_down * 0.5),int(frame.shape[0] * self.percent_scaled_down * 0.5)),(frame.shape[1] - math.floor(frame.shape[1]*self.percent_scaled_down * 0.5),frame.shape[0] - math.floor(frame.shape[0]*self.percent_scaled_down * 0.5)), (0, 255, 0), 2)
        highlight_mask_colored[highlight_mask > 0] = self.highlight_color
        highlight_combined = cv2.add(highlight_mask_colored, highlight_annotations)
        blended_frame = cv2.addWeighted(highlight_combined, self.highlight_alpha, frame, 1 - self.highlight_alpha, 0)
        blended_contour_centers = cv2.addWeighted(contour_centers, self.highlight_alpha, frame,
                                                  1 - self.highlight_alpha, 0)

        self.previous_objects = current_objects
        return blended_frame, blended_contour_centers, contour_map, object_found

    def disparity_to_distance(self, disparity):
        """
        This is a pretty simple function we just do a simple calculation to get the distance for each value in the disparity array
        :param disparity: The disparity map 2D
        :return: The distance map, a 2D array with each value being the distance at that point
        """
        valid_mask = (disparity > 0.1)  # We check for valid areas

        points_3d = cv2.reprojectImageTo3D(disparity, self.Q)  # Pretty self-explanatory
        distance_map = points_3d[:, :, 2]  # We get the Z value from the 3D matrix
        distance_map[~valid_mask] = np.nan  # And we fill in the matrix with valid values

        return distance_map, points_3d, valid_mask

    def determine_object(self):
        pass
