import socket
import cv2
import pickle
import struct
import time
import multiprocessing

"""
Raspberry PI Class

This is supposed to run on the Raspberry PI, right now we have to figure out how to remote execute this on it.
IMPORTANT!!!!!: The Raspberry PI and the Laptop(server) need to be on the same WIFI to work.

Params:
@server_ip: The IP address of your computer for the pi to connect to.
@server_port: The port that the pi will attempt to connect to on the server.
@camera_index: The index of the camera that will be used to send, the values are 0 and 2 for each camera, NOT 1!!!! (idk why but 1 doesn't work)
"""

class CameraClient:
    def __init__(self, server_ip, server_port, camera_index: int):
        self.server_ip = server_ip
        self.server_port = server_port
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.video_capture = cv2.VideoCapture(camera_index)  # Open the camera
        self.connect_to_server()

    def connect_to_server(self):
        while True:
            try:
                self.client_socket.connect((self.server_ip, self.server_port))
                print(f"Raspberry PI Connected to socket")
                break
            except ConnectionRefusedError:
                print("Connection refused Retrying...")
                time.sleep(3)

    def send_video_stream(self):
        while True:
            try:
                while self.video_capture.isOpened():
                    ret, frame = self.video_capture.read()  # ret is a bool that determines if the frame was read correctly or not, frame is the video frame
                    if not ret:
                        break

                    _, buffer = cv2.imencode('.jpg', frame)  # Encode the frame into a numpy array in the buffer
                    data = pickle.dumps(buffer)  # Serialize the encoded frame
                    size = struct.pack("Q", len(data))  # Get the size of the frame to send
                    self.client_socket.sendall(size + data)  # Send it

            except (BrokenPipeError, ConnectionResetError) as e:
                print(f"Connection lost: {e}, Reconnecting...")
                self.client_socket.close()
                self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.connect_to_server()

            except Exception as e:
                print(f"Client error: {e}")
                break

        self.video_capture.release()
        self.client_socket.close()
        print("Client Connection closed.")


def start_camera_stream(server_ip, port, camera_index):
    client = CameraClient(server_ip, port, camera_index)
    if client.video_capture.isOpened():
        client.send_video_stream()


if __name__ == "__main__":
    SERVER_IP = "192.168.1.69" # This needs to be the IP of the laptop (will fix hardcode later)

    cameras = [
        {"camera_index": 0, "port": 9000},
        {"camera_index": 2, "port": 9001},
    ]

    processes = []
    for cam in cameras:
        process = multiprocessing.Process(target=start_camera_stream, args=(SERVER_IP, cam["port"], cam["camera_index"]))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()
