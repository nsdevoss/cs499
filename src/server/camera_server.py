import gc
import socket
import cv2
import struct
import time
import platform
import numpy as np
import src.LocalCommon as lc
from turbojpeg import TurboJPEG
from src.server.server import SocketServer
from src.server.logger import server_logger


class StreamCameraServer(SocketServer):
    """
    StreamCameraServer Class

    This is in charge of making a server and opening a port to find a camera client. Each server only opens up 1 port so we run 2 of them in main.py

    Params:
    :param host: This is who we are looking for, 0.0.0.0 just means that we are looking for any IP address on the network. You could change this to the client's IP address to directly look for them. (Plz don't do, default is fine)
    :param port: The port we are opening up on this instance. Default doesn't mean anything since we handle all of this in main.py, I'm just too lazy to change the argument order
    :param frame_queue: This passes the frame queue (more on this in main.py and below)
    """
    def __init__(self, host="0.0.0.0", port=9000, socket_type="TCP", vision_queue=None, display=True, fps=60):
        self.vision_queue = vision_queue
        self.display = display
        self.fps = fps
        self.frame_interval = 1.0 / fps
        self.shutdown = False
        system = platform.system()
        if system == "Windows":
            self.jpeg = TurboJPEG("C:/libjpeg-turbo-gcc64/bin/libturbojpeg.dll")
        else:
            self.jpeg = TurboJPEG()
        super().__init__(host, port, socket_type)

        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1048576)

    def receive_video_stream(self):
        log_writer = server_logger.get_logger()

        frame_rate = int(lc.DEFAULT_CAMERA_FPS / self.fps)
        while True:
            log_writer.info("Waiting for a connection...")

            if self.socket_type == "TCP":
                conn, addr = self.server_socket.accept()
                log_writer.info(f"Got connection from {addr}")
                self._handle_tcp_stream(conn, addr, frame_rate)
            elif self.socket_type == "UDP":
                conn = self.server_socket
                log_writer.info(f"Waiting for some UDP data on {self.host}:{self.port}")
                self._handle_udp_stream(conn, frame_rate)

    def _handle_tcp_stream(self, conn, addr, frame_rate):
        log_writer = server_logger.get_logger()
        data = b""
        payload_size = struct.calcsize("Q")
        frame_count = 0
        start_time = time.time()
        try:
            while True:
                # We get the raw data from the client here
                while len(data) < payload_size:
                    packet = conn.recv(16384)
                    if not packet:
                        log_writer.error(f"No packet received")
                        raise ConnectionResetError("No packet received")
                    data += packet

                # Extract the message size
                packed_msg_size = data[:payload_size]
                data = data[payload_size:]
                msg_size = struct.unpack("Q", packed_msg_size)[0]

                # Get the full message
                while len(data) < msg_size:
                    packet = conn.recv(16384)
                    if not packet:
                        raise ConnectionResetError("Client disconnected")
                    data += packet

                frame_data = data[:msg_size]
                data = data[msg_size:]

                # Process the frame
                try:
                    frame = self.jpeg.decode(frame_data)
                except Exception as e:
                    log_writer.warning(f"JPEG decoding failed: {e}")
                    continue


                if frame is None:
                    log_writer.warning("Received an empty or corrupted frame.")
                    continue

                frame_count += 1

                if self.vision_queue is not None:
                    self.vision_queue.put(frame)
                now = time.time()
                if now - start_time >= 30.0:
                    fps = frame_count / (now - start_time)
                    log_writer.info(f"SERVER FPS: {fps:.2f}")
                    frame_count = 0
                    start_time = now
                del frame, frame_data
                gc.collect()

                # This is for cleanup, but it doesn't really do anything
                if len(data) > 10_000_000:
                    data = b""

                if self.shutdown:
                    break

        except (ConnectionResetError, BrokenPipeError) as e:
            log_writer.error(f"{e}. Client disconnected. Waiting for a new connection...")
        except Exception as e:
            log_writer.error(f"{e}")
        finally:
            if conn and conn != self.server_socket:
                try:
                    conn.close()
                except Exception:
                    pass
            log_writer.info(f"Closed socket connection with {addr}.")

    def _handle_udp_stream(self, conn, frame_rate):
        """
        UDP is a lot more complex than TCP so I am going to try to explain as much as I can about this process.
        For UDP we do not make a handshake, we are just looking for information being sent to us from anywhere.
        We need to know how big each packet is, we get this from the header of the packet.

        Example:            [Header]  → [Num Chunks: 4] [Total Size: 65536 bytes]
                            [Chunk 1] → [Sequence Number: 0] [Data]
                            [Chunk 2] → [Sequence Number: 1] [Data]
                            [Chunk 3] → [Sequence Number: 2] [Data]
                            [Chunk 4] → [Sequence Number: 3] [Data]
        :param conn: Who we are receiving from
        :param frame_rate: This is for the FPS for inputting into the frame queue
        """
        log_writer = server_logger.get_logger()
        frame_count = 0

        fragments = {}  # Need this because UDP doesn't guarente packet order so we store them until we get the full frame
        frame_sizes = {}    # How big the frames are

        try:
            while True:
                try:
                    conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1048576)

                    conn.settimeout(0.5)

                    packet, addr = conn.recvfrom(65536)  # We expect at most 65536 bytes coming in, we get this packet and who sent it

                    if packet == b"PING":                # On the client we PING the server to make sure it is alive before we send
                        log_writer.info(f"Received ping from {addr}")
                        continue

                    """
                    Here we get the header and get information about the packet we receive
                    The header is 16 bytes, 8 for the amount of chunks sent, and 8 for the total size of the whole frame being sent and how many packets
                    Example:                Packet 1 [Header]  → [Num Chunks: 4] [Total Size: 65536 bytes]
                                            Packet 2 [Chunk 1] → [Sequence Number: 0] [Data]
                                            Packet 3 [Chunk 2] → [Sequence Number: 1] [Data]
                                            Packet 4 [Chunk 3] → [Sequence Number: 2] [Data]
                                            Packet 5 [Chunk 4] → [Sequence Number: 3] [Data]
                    """
                    if len(packet) == 16:  # 8 bytes for num_chunks + 8 bytes for total_size
                        num_chunks, total_size = struct.unpack("QQ", packet)  # Here we get the info about the packet sent. "QQ" just means 16 bytes, we are getting the first 16 bytes from the package (aka the header)
                        log_writer.info(f"Expecting {num_chunks} chunks for a {total_size} byte frame from {addr}")

                        client_key = f"{addr[0]}:{addr[1]}"  # The client's address, example: "192.168.1.45:9182"
                        fragments[client_key] = {}  # Initialize fragments to store X amount of chunks from x.x.x.x:yyyy
                        frame_sizes[client_key] = (num_chunks, total_size)
                        continue

                    # If we get here... then it's a data packet
                    # This is where we get the sequence number from the packet like above, ex: [Chunk 2] → [Sequence Number: 1] [Data]
                    seq_num = struct.unpack("Q", packet[:8])[0]  # This is the number, ex: seq_num  = [Sequence Number: 1]
                    chunk = packet[8:]  # This is the chunk from the packet, ex: chunk = [Data]

                    if len(chunk) > lc.MAX_UDP_PACKET:
                        log_writer.error(f"Chunk too large: {len(chunk)} bytes. Dropping...")
                        continue

                    client_key = f"{addr[0]}:{addr[1]}"
                    if client_key not in fragments:
                        log_writer.warning(f"Received chunk but no header from {addr}")
                        continue

                    fragments[client_key][seq_num] = chunk  # We store the chunk using the sequence number as the key so we can pack it all together later

                    # Check if we have all chunks
                    if len(fragments[client_key]) == frame_sizes[client_key][0]:
                        num_chunks, total_size = frame_sizes[client_key]

                        sorted_chunks = [fragments[client_key][i] for i in range(num_chunks) if  # We assemble the chunks in order
                                         i in fragments[client_key]]

                        # If we're missing chunks then we skip the frame
                        if len(sorted_chunks) != num_chunks:
                            log_writer.warning(f"Missing chunks for frame: {len(sorted_chunks)}/{num_chunks}")
                            fragments[client_key] = {}
                            continue

                        frame_data = b"".join(sorted_chunks)  # This says "Treat me as a byte sequence"

                        # Process the frame (Might optimize later bcz pickle is slow)
                        try:
                            frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)

                            if frame is None:
                                log_writer.warning("Received an empty or corrupted frame.")
                                fragments[client_key] = {}
                                continue

                            frame_count += 1
                            if frame_count % frame_rate != 0:
                                fragments[client_key] = {}
                                continue

                            if self.vision_queue is not None:
                                self.vision_queue.put(frame)

                            # Cleanup
                            fragments[client_key] = {}

                            del frame, frame_data
                            gc.collect()

                        except Exception as e:
                            log_writer.error(f"Error processing frame: {e}")
                            fragments[client_key] = {}

                except socket.timeout:
                    pass

                except Exception as e:
                    log_writer.error(f"UDP receive error: {e}")

                if self.shutdown:
                    break

        except Exception as e:
            log_writer.error(f"UDP stream error: {e}")

    def shutdown(self):
        server_logger.get_logger().info("Shutting down server")
        self.server_socket.close()
        self.shutdown = True
        server_logger.get_logger().info("Successfully shut down server")


if __name__ == "__main__":
    server = StreamCameraServer()
    server.receive_video_stream()