# client_pi_camera.py (Runs on Raspberry Pi)
import socket
import cv2
import pickle
import struct

class RaspberryPiCameraClient:
    def __init__(self, server_ip, server_port):
        self.server_ip = server_ip
        self.server_port = server_port
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.server_ip, self.server_port))
        self.video_capture = cv2.VideoCapture(0)

    def send_video_stream(self):
        try:
            while self.video_capture.isOpened():
                ret, frame = self.video_capture.read()
                if not ret:
                    break

                _, buffer = cv2.imencode('.jpg', frame)
                data = pickle.dumps(buffer) 
                size = struct.pack("Q", len(data))
                self.client_socket.sendall(size + data)

        except Exception as e:
            print(f"Client error: {e}")
        finally:
            self.video_capture.release()
            self.client_socket.close()
            print("Connection closed.")

if __name__ == "__main__":
    SERVER_IP = "192.168.1.69"  # This needs to be the IP of the laptop
    SERVER_PORT = 9000

    client = RaspberryPiCameraClient(SERVER_IP, SERVER_PORT)
    client.send_video_stream()
