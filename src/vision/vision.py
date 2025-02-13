import cv2
from src.vision import stitching


'''
TODO: Make a vision class that handles all of the calculations
      This class should ONLY output the frame needed to show and output the information needed for the app/raspberry pi
      Might need to make a data class that wraps everything together for easy exporting
      
      The goal for this class is to only be instantiated once in the main loop and have an export function that exports
      the necessary data needed
      
      ex:  main.py
        def main(x, y, z):
            frame_queue = multiprocessing.Queue()
            vision = Vision(frame_queue)
            process = multiprocessing.Process(target=vision.start(), args=(x, y, z))
            process.start()
            process.join()
            
            data = vision.export()
'''
class Vision:
    """
    The vision class that will handle all the calculations

    Params:
    @frame_queue: this is the frame queue with all the frames in it (see main.py).
    @arguments: a dictionary that determines which functionalities are ran at runtime
                False means it does not run and True means it runs, default for each is False.
                {"stitch": False, "depth_perception": False, "object_detection": False, "etc.": False}
    @port: [not implemented] determines what port the data will be exported on.
    """
    def __init__(self, frame_queue, action_arguments: dict, server_logger, port=None):
        self.frame_queue = frame_queue
        self.action_arguments = action_arguments
        self.server_logger = server_logger

    def start(self):
        for action in self.action_arguments:
            if self.action_arguments[action]:
                eval(f"self.{action}()")

    def stitching(self):
        print("We are stitching")
        stitching.frame_stitcher(self.frame_queue)

    # Not yet implemented
    def export(self):
        return cv2.Canny(self.frame_queue, 10, 200)