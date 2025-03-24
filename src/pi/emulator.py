import socket
import time
import cv2
import os
from src.utils import utils
import pickle
import struct
import ipaddress
from src.server.logger import client_logger

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

"""
Emulator Class

This class is a copy of the Raspberry PI and provides the same functionality but on your machine.

Params:
:param server_ip: The IP address of your computer for the emulator to connect to
:param video: The video that will be played on the instance of the emulator.
:param server_port: The port that the emulator will attempt to connect to on the server
:param logger: The logger passed into here
"""
class Emulator:
    def __init__(self, server_ip, video, stream_enabled: bool, server_port: int, resolution=(2560,720), fps=30):
        self.server_ip = server_ip
        self.server_port = server_port
        self.shutdown = False
        self.resolution = resolution
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if stream_enabled:
            self.video = cv2.VideoCapture(0)
        else:
            video_path = os.path.join(ROOT_DIR, "assets/videos", f"{video}.mp4")
            client_logger.get_logger().info(f"Video path: {video_path}")
            if os.path.exists(video_path):
                self.video = cv2.VideoCapture(video_path)
            else:
                client_logger.get_logger().warning(f"Video path doesn't exist, using default video")
                self.video = cv2.VideoCapture(os.path.join(ROOT_DIR, "assets/videos", "chair2.mp4"))
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.video.set(cv2.CAP_PROP_FPS, fps)
        self.video.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.connect_to_server()
        client_logger.get_logger().info("Emulator initialized successfully.")

    def connect_to_server(self):
        """
        This is where we connect to the server and check if it is valid before trying
        """
        log_writer = client_logger.get_logger()
        while True:
            try:
                log_writer.info(f"Trying address: {self.server_ip} on {self.server_port}")

                try:
                    # Validate IP and Port
                    assert isinstance(self.server_port, int), f"Port must be an integer, got {type(self.server_port)}"
                    assert 1 <= self.server_port <= 65535, f"Port {self.server_port} is out of range"
                    ipaddress.ip_address(self.server_ip)
                except AssertionError as e:
                    log_writer.error(str(e))
                    raise e

                while not is_port_open(self.server_ip, self.server_port):
                    log_writer.warning(f"Port {self.server_port} not open yet, chill...")
                    time.sleep(2)

                self.client_socket.connect((self.server_ip, self.server_port))
                log_writer.info(f"Connected to socket")
                break

            except ConnectionRefusedError:
                log_writer.error("Connection refused, retrying...")
                time.sleep(3)

            except OSError as e:
                log_writer.error(f"OSError: {e}, retrying...")
                time.sleep(3)

    def send_video(self):
        """
        Sends the emulator video to the server on the port
        """
        log_writer = client_logger.get_logger()
        frame_counter = 0
        while True:
            try:
                ret, frame = self.video.read()  # Load the video, ret is a bool that states if the frame was read correctly
                if not ret:
                    log_writer.error("Could not read frame.")
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
                del data

                if self.shutdown:
                    break

            except (BrokenPipeError, ConnectionResetError) as e:
                log_writer.error(f"Connection lost: {e}. Reconnecting...")
                self.client_socket.close()
                self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.connect_to_server()

            except Exception as e:
                log_writer.error(f"Error: {e}")
                break

        self.video.release()  # Release video because we are good programmers and care about our resources
        self.client_socket.close()  # CLose connection to server
        log_writer.info("Connection closed.")

    def send_video_stream(self):
        log_writer = client_logger.get_logger()
        while True:
            try:
                while self.video.isOpened():
                    # ret is a bool that determines if the frame was read correctly or not, frame is the video frame
                    ret, frame = self.video.read()
                    if not ret:
                        break

                    _, buffer = cv2.imencode('.jpg', frame)  # Encode the frame into a numpy array into the buffer
                    data = pickle.dumps(buffer)  # Serialize the encoded frame
                    size = struct.pack("Q", len(data))  # Get the size of the frame to send
                    self.client_socket.sendall(size + data)  # Send it

            except (BrokenPipeError, ConnectionResetError) as e:
                log_writer.error(f"Connection lost: {e}, Reconnecting...")
                self.client_socket.close()
                self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.connect_to_server()

            except Exception as e:
                log_writer.error(f"Error: {e}")
                break

        self.video.release()
        self.client_socket.close()
        log_writer.info("Client Connection closed.")

    def shutdown(self):
        log_writer = client_logger.get_logger()
        log_writer.info(f"Shutting down emulator...")
        self.client_socket.close()
        self.shutdown = True
        log_writer.info(f"Successfully shut down emulator")



def is_port_open(ip, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(2)
        return sock.connect_ex((ip, port)) == 0


if __name__ == "__main__":
    SERVER_IP = utils.get_ip_address()
    SERVER_PORT = 9000

    client = Emulator(SERVER_IP, "rotate", SERVER_PORT)
    client.send_video()
