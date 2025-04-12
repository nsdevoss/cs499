import socket
import cv2
import os
import struct
import ipaddress
import time
import concurrent.futures
import platform
import src.LocalCommon as lc
from src.utils import utils
from src.server.logger import client_logger
from turbojpeg import TurboJPEG, TJFLAG_FASTDCT

MAX_UDP_PACKET = 8216


class Emulator:
    """
    Emulator Class

    This class is a copy of the Raspberry PI and provides the same functionality but on your machine.

    Params:
    :param server_ip: The IP address of your computer for the emulator to connect to
    :param video: The video that will be played on the instance of the emulator.
    :param stream_enabled: This defines if the emulator will use the external camera rather than load a video to send
    :param server_port: The port that the emulator will attempt to connect to on the server
    :param encode_quality: Determines the quality the video will be sent as a percent (1-100)
    :param resolution: The video resolution that will be resized to (Might be hardcoded into the raspberry Pi if i can't figure out how to remote exec)
    """

    def __init__(self, server_ip, video, stream_enabled: bool, server_port: int, socket_type="TCP", encode_quality=70, scale=1.0):
        self.server_ip = server_ip
        self.server_port = server_port
        self.shutdown = False
        self.socket_type = socket_type
        self.scale = scale
        system = platform.system()
        if system == "Windows":
            self.jpeg = TurboJPEG("C:/libjpeg-turbo-gcc64/bin/libturbojpeg.dll")
        else:
            self.jpeg = TurboJPEG()
        self.encode_quality = encode_quality
        try:
            assert self.socket_type == "TCP" or self.socket_type == "UDP", f'Socket type must be "TCP" or "UDP", got: {self.socket_type}'
            if self.socket_type == "TCP":
                socket_type = socket.SOCK_STREAM
            elif self.socket_type == "UDP":
                socket_type = socket.SOCK_DGRAM
        except AssertionError as e:
            client_logger.get_logger().error(e)

        self.client_socket = socket.socket(socket.AF_INET, socket_type)
        if stream_enabled:
            self.video = cv2.VideoCapture(0)
        else:
            video_path = os.path.join(lc.ROOT_DIR, "assets/videos", f"{video}.mp4")
            client_logger.get_logger().info(f"Video path: {video_path}")
            if os.path.exists(video_path):
                self.video = cv2.VideoCapture(video_path)
            else:
                client_logger.get_logger().warning(f"Video path doesn't exist, using default video")
                self.video = cv2.VideoCapture(os.path.join(lc.ROOT_DIR, "assets/videos", "car.mp4"))
        self.video.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.connect_to_server()
        client_logger.get_logger().info("Emulator initialized successfully.")

    def connect_to_server(self):
        # We connect to the server here and do some validating things
        log_writer = client_logger.get_logger()
        if self.socket_type == "TCP":
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

        elif self.socket_type == "UDP":
            # For UDP, we don't need to establish a connection, just validate the IP and port
            try:
                assert isinstance(self.server_port, int), f"Port must be an integer, got {type(self.server_port)}"
                assert 1 <= self.server_port <= 65535, f"Port {self.server_port} is out of range"
                ipaddress.ip_address(self.server_ip)

                # Ping the server
                self.client_socket.sendto(b"PING", (self.server_ip, self.server_port))
                log_writer.info(f"UDP client ready to send to {self.server_ip}:{self.server_port}")
            except AssertionError as e:
                log_writer.error(str(e))
                raise e

    def send_video_stream(self):
        log_writer = client_logger.get_logger()
        log_writer.info("Starting live camera feed...")
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        for _ in range(30):
            self.video.read()

        ret, frame = self.video.read()
        if not ret:
            log_writer.error("Could not read initial frame.")
            return
        log_writer.info(f"Read initial frame with dimensions: {frame.shape}")
        frame = cv2.resize(frame, (int(frame.shape[1] * self.scale), int(frame.shape[0] * self.scale)))
        log_writer.info(f"Resizing frame dimensions to: {frame.shape}")

        future_encode = executor.submit(self.jpeg.encode, frame, quality=self.encode_quality, flags=TJFLAG_FASTDCT)

        frame_count = 0
        fps_count = time.time()
        while True:
            try:
                ret, next_frame = self.video.read()
                if not ret:
                    log_writer.error("Could not read frame.")
                    break

                next_frame = cv2.resize(next_frame, (int(next_frame.shape[1] * self.scale), int(next_frame.shape[0] * self.scale)))

                # Serialize the frame
                buffer = future_encode.result()
                data = buffer

                future_encode = executor.submit(self.jpeg.encode, next_frame, quality=self.encode_quality,flags=TJFLAG_FASTDCT)

                if self.socket_type == "TCP":
                    """
                    A TCP packet is structured like [Size of data][DATA]. The first 8 bytes of the packet say how big the data will be.
                    To make a packet, we need to add this size to the first 8 bytes of our packet so the server will know how to decode it and what to expect.
                    Then we attach the actual data and send it across to the server.
                    
                    That was easy, now lets see how UDP does it......
                    """
                    size = struct.pack("Q", len(data))
                    self.client_socket.sendall(size + data)
                    frame_count += 1

                    now = time.time()
                    if now - fps_count >= 60.0:
                        fps = frame_count / (now - fps_count)
                        log_writer.info(f"CLIENT FPS: {fps:.2f}")
                        frame_count = 0
                        fps_count = now

                elif self.socket_type == "UDP":
                    """
                    A UDP packet is structured similarly to a TCP except that in TCP, both sides know what to expect. Think of a phone call
                    UDP just sends data and does not need to hear back from the other side. Think of sending a letter
                    
                    Since UDP has a size limit of 65536 bytes per packet we limit to 8KB for safety.
                    We break down the frame into chunks with sequence numbers so each chunk is small enough to be sent over the network
                    Here is an example of what a UDP packet sequence looks like:
                    Example:                Packet 1 [Header]  → [Num Chunks: 4] [Total Size: 65536 bytes]
                                            Packet 2 [Chunk 1] → [Sequence Number: 0] [Data]
                                            Packet 3 [Chunk 2] → [Sequence Number: 1] [Data]
                                            Packet 4 [Chunk 3] → [Sequence Number: 2] [Data]
                                            Packet 5 [Chunk 4] → [Sequence Number: 3] [Data]
                    Since UDP does not guarantee packet order, or even the packet at all! Each frame is split up into chunks
                    The first sent packet tells the server the number of chunks the frame was split into, and the total size of the frame.
                    We send these over to the server while it builds all of them up until it receives the specified number of chunks.
                    """
                    chunk_size = MAX_UDP_PACKET - 64  # Header will be 16 bytes, so we have an error buffer of 2
                    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]  # We split the frame up into its chunks with sequence numbers

                    header = struct.pack("QQ", len(chunks), len(data))  # This is the header where we declare the number of chunks and the total size of the frame
                    self.client_socket.sendto(header, (self.server_ip, self.server_port))  # We send the header first, in the above example this is Packet 1

                    # For each chunk we send a packet, the first 8 bytes is the sequence number and the rest is the data
                    # Example: Packet 2 [Chunk 1] → [Sequence Number: 0] [Data]
                    for i, chunk in enumerate(chunks):
                        packet = struct.pack("Q", i) + chunk  # This is the packet [Number][Data]
                        try:
                            self.client_socket.sendto(packet, (self.server_ip, self.server_port))
                        except OSError as e:
                            if "Message too long" in str(e):
                                log_writer.error(f"Packet too large: {len(packet)} bytes.")
                                break
                            else:
                                raise e
                    frame_count += 1

                    now = time.time()
                    if now - fps_count >= 60.0:
                        fps = frame_count / (now - fps_count)
                        log_writer.info(f"CLIENT FPS: {fps:.2f}")
                        frame_count = 0
                        fps_count = now

                if self.shutdown:
                    break

            except (BrokenPipeError, ConnectionResetError) as e:
                log_writer.error(f"Connection lost: {e}. Reconnecting...")
                if self.socket_type == "TCP":
                    self.client_socket.close()
                    self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self.connect_to_server()

            except Exception as e:
                log_writer.error(f"Streaming error: {e}")
                break

        self.video.release()
        self.client_socket.close()
        executor.shutdown()
        log_writer.info("Video stream closed.")

    def send_video(self):
        """
        Sends the emulator video to the server on the port, same thing as feed but changed up the way we get frames
        See send_video_stream() to see how we do it, but this is basically a copy of that
        """
        log_writer = client_logger.get_logger()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        frame_counter = 0

        ret, frame = self.video.read()
        if not ret:
            log_writer.error("Could not read initial frame.")
            return
        frame = cv2.resize(frame, (int(frame.shape[1] * self.scale), int(frame.shape[0] * self.scale)))

        future_encode = executor.submit(self.jpeg.encode, frame, quality=self.encode_quality, flags=TJFLAG_FASTDCT)

        frame_count = 0
        fps_count = time.time()

        while True:
            try:
                if frame_counter == self.video.get(cv2.CAP_PROP_FRAME_COUNT):
                    frame_counter = 0
                    self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)

                ret, next_frame = self.video.read()
                if not ret:
                    self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                next_frame = cv2.resize(next_frame, (int(next_frame.shape[1]*self.scale), int(next_frame.shape[0]*self.scale)))
                frame_counter += 1


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
                        log_writer.info(f"CLIENT FPS: {fps:.2f}")
                        frame_count = 0
                        fps_count = now

                elif self.socket_type == "UDP":
                    chunk_size = MAX_UDP_PACKET - 16
                    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

                    header = struct.pack("QQ", len(chunks), len(data))
                    self.client_socket.sendto(header, (self.server_ip, self.server_port))

                    for i, chunk in enumerate(chunks):
                        packet = struct.pack("Q", i) + chunk
                        try:
                            self.client_socket.sendto(packet, (self.server_ip, self.server_port))
                            time.sleep(0.001)
                        except OSError as e:
                            if "Message too long" in str(e):
                                log_writer.error(
                                    f"Packet too large: {len(packet)} bytes. Try reducing quality or resolution.")
                                break
                            else:
                                raise e
                    frame_count += 1

                    now = time.time()
                    if now - fps_count >= 60.0:
                        fps = frame_count / (now - fps_count)
                        log_writer.info(f"CLIENT FPS: {fps:.2f}")
                        frame_count = 0
                        fps_count = now

                if self.shutdown:
                    break

                time.sleep(1.0 / self.video.get(cv2.CAP_PROP_FPS))

            except (BrokenPipeError, ConnectionResetError) as e:
                log_writer.error(f"Connection lost: {e}. Reconnecting...")
                if self.socket_type == "TCP":
                    self.client_socket.close()
                    self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self.connect_to_server()
                # For UDP, we can just continue bcz it's connectionless so no need for a handshake

            except Exception as e:
                log_writer.error(f"Error: {e}")
                break

        self.video.release()
        self.client_socket.close()
        executor.shutdown()
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
