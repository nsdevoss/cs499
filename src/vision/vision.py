import cv2
from src.vision import stitching
from src.vision.depth import DepthEstimator


class Vision:
    """
    The vision class that will handle all the calculations.

    Params:
    :param frame_queue: Frame queue containing frames from both cameras.
    :param action_arguments: Dictionary defining which functionalities to enable.
                       {"stitch": False, "depth_perception": False}
    :param server_logger: The server logger
    """
    def __init__(self, frame_queue, action_arguments: dict, server_logger, port=None):
        self.frame_queue = frame_queue
        self.action_arguments = action_arguments
        self.server_logger = server_logger
        self.port = port

        self.depth_estimator = DepthEstimator(baseline=0.1, focal_length=700)

    def start(self):
        for action in self.action_arguments:
            if self.action_arguments[action]:
                eval(f"self.{action}()")

    def stitching(self):
        self.server_logger.get_logger().info("Running stitching.")
        stitching.frame_stitcher(self.frame_queue)

    def depth_perception(self):
        frames = {9000: None, 9001: None}

        while True:
            port, frame = self.frame_queue.get()
            frames[port] = frame

            if frames[9000] is not None and frames[9001] is not None:
                try:
                    disparity = self.depth_estimator.compute_disparity(frames[9000], frames[9001])
                    depth_map = self.depth_estimator.compute_depth(disparity)

                    cv2.imshow("Disparity Map", (disparity / disparity.max() * 255).astype('uint8'))
                    cv2.imshow("Depth Map", (depth_map / depth_map.max() * 255).astype('uint8'))

                    if cv2.waitKey(1) == ord("q"):
                        break

                except Exception as e:
                    self.server_logger.get_logger().error(f"Error computing depth: {e}")

        cv2.destroyAllWindows()

    def export(self):
        # Womp womp
        return
