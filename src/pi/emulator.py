import socket
import time
import cv2
import os
from src.utils import utils
import pickle
import struct

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

"""
Emulator Class

This class is a copy of the Raspberry PI and provides the same functionality but on your machine.

Params:
@server_ip: The IP address of your computer for the emulator to connect to
@video: The video that will be played on the instance of the emulator.
@server_port: The port that the emulator will attempt to connect to on the server
"""
class Emulator:
    def __init__(self, server_ip, video, server_port):
        self.server_ip = server_ip
        self.server_port = server_port
        # Boilerplate socket set up stuff
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        video_path = os.path.join(ROOT_DIR, "assets/videos", f"{video}.mp4")
        self.video = cv2.VideoCapture(video_path)
        self.connect_to_server()

    def connect_to_server(self):
        """
        Attempts to connect to the server
        """
        while True:
            try:
                self.client_socket.connect((self.server_ip, self.server_port))
                print(f"Emulator Connected to socket")
                break
            except ConnectionRefusedError:
                print("Connection refused Retrying...")
                time.sleep(3)

    def send_video(self):
        """
        Sends the emulator video to the server on the port
        """
        frame_counter = 0
        while True:
            try:
                ret, frame = self.video.read()  # Load the video, ret is a bool that states if the frame was read correctly
                if not ret:
                    print("Error: Could not read frame.")
                    break

                # This is only here for the emulator to loop the video when it finishes
                frame_counter += 1
                if frame_counter == self.video.get(cv2.CAP_PROP_FRAME_COUNT):
                    frame_counter = 0
                    self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)

                _, buffer = cv2.imencode('.jpg', frame)  # We compress the image into I think a numpy array
                data = pickle.dumps(buffer)  # We serialize the frame into a byte stream
                size = struct.pack("Q", len(data))  # We get the size of the frame and add 8 bytes to the front b/c thats what the server just needs

                self.client_socket.sendall(size + data)

            except (BrokenPipeError, ConnectionResetError) as e:
                print(f"Connection lost: {e}. Reconnecting...")
                self.client_socket.close()
                self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.connect_to_server()

            except Exception as e:
                print(f"Client error: {e}")
                break

        self.video.release()  # Release video because we are coders good at managing resources
        self.client_socket.close()  # CLose connection to server
        print("Connection closed.")


if __name__ == "__main__":
    SERVER_IP = utils.get_ip_address()
    SERVER_PORT = 9000

    client = Emulator(SERVER_IP, "rotate", SERVER_PORT)
    client.send_video()
