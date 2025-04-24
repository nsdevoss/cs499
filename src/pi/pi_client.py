import argparse
import ipaddress
import socket
import cv2
import struct
import time
import concurrent.futures
from turbojpeg import TurboJPEG, TJFLAG_FASTDCT

MAX_UDP_PACKET = 8216


class CameraClient:
    """
    Raspberry PI Class

    This is supposed to run on the Raspberry PI, right now we have to figure out how to remote execute this on it.
    IMPORTANT!!!!!: The Raspberry PI and the Laptop(server) need to be on the same WIFI to work.

    Params:
    :param server_ip: The IP address of your computer for the pi to connect to.
    :param server_port: The port that the pi will attempt to connect to on the server.
    """

    def __init__(self, server_ip, server_port: int, socket_type="TCP", encode_quality=70, scale=0.2, fps=30):
        self.server_ip = server_ip
        self.server_port = server_port
        self.shutdown = False
        self.socket_type = socket_type
        self.scale = scale
        self.fps = fps
        self.jpeg = TurboJPEG()
        self.encode_quality = encode_quality
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            assert self.socket_type == "TCP" or self.socket_type == "UDP", f'Socket type must be "TCP" or "UDP", got: {self.socket_type}'
            if self.socket_type == "TCP":
                socket_type = socket.SOCK_STREAM
            elif self.socket_type == "UDP":
                socket_type = socket.SOCK_DGRAM
        except AssertionError as e:
            print(f"Error: {e}")
        self.client_socket = socket.socket(socket.AF_INET, socket_type)
        self.video = cv2.VideoCapture(0)
        self.video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.video.set(cv2.CAP_PROP_FPS, self.fps)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.connect_to_server()

    def connect_to_server(self):
        if self.socket_type == "TCP":
            while True:
                try:
                    print(f"Trying address: {self.server_ip} on {self.server_port}")
                    try:
                        # Validate IP and Port
                        assert isinstance(self.server_port,
                                          int), f"Port must be an integer, got {type(self.server_port)}"
                        assert 1 <= self.server_port <= 65535, f"Port {self.server_port} is out of range"
                        ipaddress.ip_address(self.server_ip)
                    except AssertionError as e:
                        print(str(e))
                        raise e

                    while not is_port_open(self.server_ip, self.server_port):
                        print(f"Port {self.server_port} not open yet, chill...")
                        time.sleep(2)

                    self.client_socket.connect((self.server_ip, self.server_port))
                    print(f"Connected to socket")
                    break

                except ConnectionRefusedError:
                    print("Connection refused, retrying...")
                    time.sleep(3)

                except OSError as e:
                    print(f"OSError: {e}, retrying...")
                    time.sleep(3)

        elif self.socket_type == "UDP":
            try:
                assert isinstance(self.server_port, int), f"Port must be an integer, got {type(self.server_port)}"
                assert 1 <= self.server_port <= 65535, f"Port {self.server_port} is out of range"
                ipaddress.ip_address(self.server_ip)

                self.client_socket.sendto(b"PING", (self.server_ip, self.server_port))
                print(f"UDP client ready to send to {self.server_ip}:{self.server_port}")
            except AssertionError as e:
                print(str(e))
                raise e

    def send_video_stream(self):
        print("Starting live camera feed...")
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        for _ in range(30):
            self.video.read()

        ret, frame = self.video.read()
        if not ret:
            print("Could not read initial frame.")
            return
        print(f"Read initial frame with dimensions: {frame.shape}")
        frame = cv2.resize(frame, (int(frame.shape[1] * self.scale), int(frame.shape[0] * self.scale)))
        print(f"Resizing frame dimensions to: {frame.shape}")

        future_encode = executor.submit(self.jpeg.encode, frame, quality=self.encode_quality, flags=TJFLAG_FASTDCT)

        frame_count = 0
        fps_count = time.time()
        while True:
            try:
                ret, next_frame = self.video.read()
                if not ret:
                    print("Could not read frame.")
                    break

                next_frame = cv2.resize(next_frame,
                                        (int(next_frame.shape[1] * self.scale), int(next_frame.shape[0] * self.scale)))

                buffer = future_encode.result()
                data = buffer

                future_encode = executor.submit(self.jpeg.encode, next_frame, quality=self.encode_quality, flags=TJFLAG_FASTDCT)

                if self.socket_type == "TCP":
                    size = struct.pack("Q", len(data))
                    self.client_socket.sendall(size + data)
                    frame_count += 1

                    now = time.time()
                    if now - fps_count >= 60.0:
                        fps = frame_count / (now - fps_count)
                        print(f"CLIENT FPS: {fps:.2f}")
                        frame_count = 0
                        fps_count = now

                elif self.socket_type == "UDP":
                    chunk_size = MAX_UDP_PACKET - 64
                    chunks = [data[i:i + chunk_size] for i in range(0, len(data),
                                                                    chunk_size)]

                    header = struct.pack("QQ", len(chunks),
                                         len(data))
                    self.client_socket.sendto(header, (self.server_ip,
                                                       self.server_port))

                    for i, chunk in enumerate(chunks):
                        packet = struct.pack("Q", i) + chunk
                        try:
                            self.client_socket.sendto(packet, (self.server_ip, self.server_port))
                        except OSError as e:
                            if "Message too long" in str(e):
                                print(f"Packet too large: {len(packet)} bytes.")
                                break
                            else:
                                raise e
                    frame_count += 1

                    now = time.time()
                    if now - fps_count >= 60.0:
                        fps = frame_count / (now - fps_count)
                        print(f"CLIENT FPS: {fps:.2f}")
                        frame_count = 0
                        fps_count = now

                if self.shutdown:
                    break

            except (BrokenPipeError, ConnectionResetError) as e:
                print(f"Connection lost: {e}. Reconnecting...")
                if self.socket_type == "TCP":
                    self.client_socket.close()
                    self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self.connect_to_server()

            except Exception as e:
                print(f"Streaming error: {e}")
                break

        self.video.release()
        self.client_socket.close()
        executor.shutdown()
        print("Video stream closed.")


def is_port_open(ip, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(2)
        return sock.connect_ex((ip, port)) == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--server_ip", required=True)
    parser.add_argument("--server_port", type=int, required=True)
    parser.add_argument("--socket_type", choices=["TCP", "UDP"], default="TCP")
    parser.add_argument("--encode_quality", type=int, default=70)
    parser.add_argument("--scale", type=float, default=0.2)
    parser.add_argument("--fps", type=int, default=30)

    args = parser.parse_args()

    client = CameraClient(
        server_ip=args.server_ip,
        server_port=args.server_port,
        socket_type=args.socket_type,
        encode_quality=args.encode_quality,
        scale=args.scale,
        fps=args.fps
    )

    client.send_video_stream()
