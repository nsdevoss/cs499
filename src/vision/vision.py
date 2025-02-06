import cv2
import numpy as np


'''
TODO: Make a vision class that handles all of the calculations
      This class should ONLY output the frame needed to show and output the information needed for the app/raspberry pi
      Might need to make a data class that wraps everything together for easy exporting
      
      The goal for this class is to only be instantiated once in the server loop and before the next loop it should output
      the information needed and then delete itself.
      
      ex:
            while True:
                raw_frame = get_raw_data()
                calc = Vision(raw_frame)
                data = calc.export()
                calc.del()
'''

class Vision():
    def __init__(self, frame):
        self.frame = frame

