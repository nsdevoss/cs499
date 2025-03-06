import cv2
import queue
from src.vision import stitching
import numpy as np


class Vision:
    """
    The vision class that will handle all the calculations.

    Params:
    :param frame_queue: Frame queue containing frames from both cameras.
    :param action_arguments: Dictionary defining which functionalities to enable.
                            {"stitch": False, "depth_estimation": False}
    :param server_logger: The server logger
    """
    def __init__(self, frame_queue, action_arguments: dict, server_logger, port=None):
        self.frame_queue = frame_queue
        self.action_arguments = action_arguments
        self.server_logger = server_logger
        self.port = port

    def start(self):
        for action in self.action_arguments:
            if self.action_arguments[action]:
                eval(f"self.{action}()")

    def stitch(self):
        self.server_logger.get_logger().info("Running stitching.")
        stitching.frame_stitcher(self.frame_queue)

    def depth_estimation(self):
        while True:
            try:
                left_frame, right_frame = self.frame_queue.get()

                if left_frame is not None and right_frame is not None:
                    try:
                        disparity = compute_disparity(left_frame, right_frame)
                        cv2.imshow("Frame Left", left_frame)
                        cv2.imshow("Frame Right", right_frame)
                        cv2.imshow("Depth Map", disparity)

                        if cv2.waitKey(1) == ord("q"):
                            break
                    except Exception as e:
                        self.server_logger.get_logger().error(f"Error computing depth estimation: {e}")

            except queue.Empty:
                self.server_logger.get_logger().warning("Frame queue is empty.")
                break  # Exit the loop if there are no frames left to process

        cv2.destroyAllWindows()

    def export(self):
        # Womp womp
        return


def compute_disparity(left, right):
    fx = 250  # Lens focal length
    baseline = 25  # Distance in mm between the two cameras
    disparities = 256  # Number of disparities to consider
    block = 15  # Block size to match
    units = 0.512  # Depth units, adjusted for output

    # Ensure images are grayscale
    if len(left.shape) > 2:
        left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    if len(right.shape) > 2:
        right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    sbm = cv2.StereoBM_create(numDisparities=disparities, blockSize=block)

    disparity = sbm.compute(left, right)
    valid_pixels = disparity > 0

    depth = np.zeros_like(left, dtype=np.uint8)
    depth[valid_pixels] = (fx * baseline) / (units * disparity[valid_pixels])
    depth = cv2.equalizeHist(depth)

    colorized_depth = np.zeros((left.shape[0], left.shape[1], 3), dtype=np.uint8)
    temp = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
    colorized_depth[valid_pixels] = temp[valid_pixels]

    return colorized_depth
