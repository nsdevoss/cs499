import time
import cv2
import queue
import numpy as np
import os
import open3d as o3d
import src.LocalCommon as lc
from src.server.logger import server_logger
from src.vision.detection import detector


class Vision:
    """
    The vision class that will handle all the calculations.

    Params:
    :param frame_queue: Frame queue containing frames from the cameras that are used FOR THE ALGORITHM ONLY!!!
    :param vison_args: Dictionary defining the parameters mainly for the depth alrogithm
    :param display_queue: This is the global queue used for DISPLAYING THE FRAMES ONLY!!!
    """

    def __init__(self, frame_queue, display_queue, vision_args, scale, object_detected):
        self.frame_queue = frame_queue
        self.display_queue = display_queue
        self.vision_args = vision_args
        self.scale = scale
        self.calibration_file = os.path.join(lc.CALIBRATION_DIR, self.vision_args.get("calibration_file"))
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
        
        self.object_detected = object_detected

    def start(self):
        if self.vision_args.get("enabled"):
            self.depth_estimation()
        else:
            return

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
                        self.object_detected.value = object_in_frame
                            
                        disp_colored = cv2.applyColorMap(disp_normalized.astype(np.uint8), cv2.COLORMAP_INFERNO)

                        if display_frame is None or display_frame.shape[1] != highlighted_frame.shape[1] + \
                                disp_colored.shape[1]:
                            # Increase height to accommodate 3 rows instead of 2
                            display_frame = np.zeros(
                                (max(highlighted_frame.shape[0], disp_colored.shape[0]) * 3,
                                 # Triple the height for 3 rows
                                 highlighted_frame.shape[1] + disp_colored.shape[1], 3),
                                dtype=np.uint8
                            )

                        # First row (top)
                        display_frame[:highlighted_frame.shape[0],
                        :highlighted_frame.shape[1]] = highlighted_frame  # Top-left
                        display_frame[:disp_colored.shape[0],
                        highlighted_frame.shape[1]:highlighted_frame.shape[1] + disp_colored.shape[
                            1]] = disp_colored  # Top-right

                        # Second row (middle)
                        middle_y_offset = highlighted_frame.shape[0]
                        display_frame[middle_y_offset:middle_y_offset + contour_centers.shape[0],
                        :contour_centers.shape[1]] = contour_centers  # Middle-left
                        display_frame[middle_y_offset:middle_y_offset + contour_map.shape[0],
                        highlighted_frame.shape[1]:highlighted_frame.shape[1] + contour_map.shape[
                            1]] = contour_map  # Middle-right

                        # Third row (bottom) - original frames
                        bottom_y_offset = middle_y_offset + highlighted_frame.shape[0]
                        display_frame[bottom_y_offset:bottom_y_offset + left_frame.shape[0],
                        :left_frame.shape[1]] = left_frame  # Bottom-left (original left frame)
                        display_frame[bottom_y_offset:bottom_y_offset + right_frame.shape[0],
                        highlighted_frame.shape[1]:highlighted_frame.shape[1] + right_frame.shape[
                            1]] = right_frame  # Bottom-right (original right frame)

                        if self.display_queue is not None:
                            self.display_queue.put((display_frame, left, points_3d, valid_dist_mask))

                        end_time = time.time()
                        if end_time - contour_refresh_map >= 10.0:
                            contour_map = np.zeros_like(left_frame)
                            contour_refresh_map = end_time

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

        range_mask = np.zeros_like(distance_map,
                                   dtype=np.uint8)  # We create an empty matrix with the same size and stuff as the map
        in_range = (distance_map >= self.highlight_min_dist) & (distance_map <= self.highlight_max_dist) & valid_pixels
        range_mask[
            in_range] = 1  # We populate each value that meets the criteria of our distance with 1 so we know what is in range

        # We find contours to filter out what we highlight, contours basically act as a filter as it makes areas of congested regions
        contours, _ = cv2.findContours(range_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        highlight_mask = np.zeros_like(range_mask)  # This mask is the same dimensions as the range 0,1 mask
        highlight = np.zeros_like(
            frame)  # This is what we will display, we create a new matrix the same size as the orig frame
        contour_centers = np.zeros_like(frame)
        object_found = False
        for contour in contours:
            if cv2.contourArea(
                    contour) >= self.highlight_min_area:  # If the contour is big enough to be considered valid
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx_contour = cv2.approxPolyDP(contour, epsilon, True)

                cv2.drawContours(highlight_mask, [approx_contour], -1, self.highlight_color, 2)
                cv2.fillPoly(highlight_mask, [contour], self.highlight_color)
                moments = cv2.moments(contour)
                if moments['m00'] != 0:
                    Cx = int(moments['m10'] / moments['m00'])
                    Cy = int(moments['m01'] / moments['m00'])
                    object_found = True
                    contour_centers = cv2.circle(contour_centers, (Cx, Cy), radius=3, color=(0, 255, 0), thickness=-1)
                    contour_map = cv2.circle(contour_map, (Cx, Cy), radius=3, color=(0, 255, 0), thickness=-1)
                    if not np.isnan(distance_map[Cy, Cx]):
                        center_distance = distance_map[Cy, Cx]
                        detector.get_detection((Cx, Cy), center_distance)
                    else:
                        center_distance = None
                
        highlight[highlight_mask > 0] = self.highlight_color

        highlight = cv2.GaussianBlur(highlight, (5, 5), 0)  # Smoothing

        # We blend the overlays with the frame
        blended_frame = cv2.addWeighted(highlight, self.highlight_alpha, frame, 1 - self.highlight_alpha, 0)
        blended_contour_centers = cv2.addWeighted(contour_centers, self.highlight_alpha, frame,
                                                  1 - self.highlight_alpha, 0)

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


def create_3d_map(points_3d, valid_mask, frame):
    # Filtering initially for valid points (not nan, infinity, etc)
    valid_points = points_3d[valid_mask].reshape(-1, 3)
    valid_colors = frame[valid_mask][:, [2, 1, 0]].reshape(-1, 3) / 255.0

    valid_points[:, 2] = -valid_points[:, 2]  # Inverse Z
    valid_points[:, 1] = -valid_points[:, 1]  # Flip over X axis

    # Making the actual Open3D object and just populating what we need
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(valid_points.astype(np.float64))
    point_cloud.colors = o3d.utility.Vector3dVector(valid_colors.astype(np.float64))

    # Filtering points
    _, good_indices = point_cloud.remove_statistical_outlier(nb_neighbors=80, std_ratio=2.0)
    filtered_cloud = point_cloud.select_by_index(good_indices)
    o3d.visualization.draw_geometries([filtered_cloud])


if __name__ == "__main__":
    # dont listen to this
    array = np.ones((1000, 1000, 3), dtype=np.uint8)
    create_3d_map(array)
