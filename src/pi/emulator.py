import socket
import time
import cv2
from src.utils import utils
import pickle
import struct


class Emulator:
    def __init__(self, server_ip, video, server_port):
        self.server_ip = server_ip
        self.server_port = server_port
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.video = cv2.VideoCapture(f'/Users/nicholasburczyk/Desktop/CS CLASS/CS499sp25/cs499/assets/videos/{video}.mp4')
        self.connect_to_server()

    def connect_to_server(self):
        while True:
            try:
                self.client_socket.connect((self.server_ip, self.server_port))
                print(f"Emulator Connected to socket")
                break
            except ConnectionRefusedError:
                print("Connection refused Retrying...")
                time.sleep(3)

    def send_video(self):
        frame_counter = 0
        while True:
            try:
                ret, frame = self.video.read()
                if not ret:
                    print("Error: Could not read frame.")
                    break

                frame_counter += 1
                if frame_counter == self.video.get(cv2.CAP_PROP_FRAME_COUNT):
                    frame_counter = 0
                    self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)

                _, buffer = cv2.imencode('.jpg', frame)
                data = pickle.dumps(buffer)
                size = struct.pack("Q", len(data))

                self.client_socket.sendall(size + data)

            except (BrokenPipeError, ConnectionResetError) as e:
                print(f"Connection lost: {e}. Reconnecting...")
                self.client_socket.close()
                self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.connect_to_server()

            except Exception as e:
                print(f"Client error: {e}")
                break

        self.video.release()
        self.client_socket.close()
        print("Connection closed.")


if __name__ == "__main__":
    SERVER_IP = utils.get_ip_address()
    SERVER_PORT = 9000

    client = Emulator(SERVER_IP, SERVER_PORT)
    client.send_video()