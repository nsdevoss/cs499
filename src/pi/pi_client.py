import ipaddress
import socket
import cv2
import pickle
import struct
import time
import multiprocessing
from src.server.logger import client_logger
from src.utils.utils import get_ip_address



class CameraClient:
    """
    Raspberry PI Class

    This is supposed to run on the Raspberry PI, right now we have to figure out how to remote execute this on it.
    IMPORTANT!!!!!: The Raspberry PI and the Laptop(server) need to be on the same WIFI to work.

    Params:
    :param server_ip: The IP address of your computer for the pi to connect to.
    :param server_port: The port that the pi will attempt to connect to on the server.
    """
    def __init__(self, server_ip, server_port, camera_index=0):
        self.server_ip = server_ip
        self.server_port = server_port
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.video_capture = cv2.VideoCapture(camera_index)  # Open the camera
        self.connect_to_server()

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

    def send_video_stream(self):
        log_writer = client_logger.get_logger()
        while True:
            try:
                while self.video_capture.isOpened():
                    # ret is a bool that determines if the frame was read correctly or not, frame is the video frame
                    ret, frame = self.video_capture.read()
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

        self.video_capture.release()
        self.client_socket.close()
        log_writer.info("Client Connection closed.")


def start_camera_stream(server_ip, port, camera_index):
    client = CameraClient(server_ip, port, camera_index)
    if client.video_capture.isOpened():
        client.send_video_stream()

def is_port_open(ip, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(2)
        return sock.connect_ex((ip, port)) == 0


if __name__ == "__main__":
    SERVER_IP = get_ip_address()
    SERVER_PORT = 9000
    process = multiprocessing.Process(target=start_camera_stream, args=(SERVER_IP, SERVER_PORT))
    process.start()
    process.join()
