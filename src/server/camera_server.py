import gc
import socket
import cv2
import pickle
import msgpack
import struct
from src.server.logger import server_logger
from src.server.server import SocketServer

MAX_QUEUE_SIZE = 10
CAMERA_DEFAULT_FPS = 60

"""
StreamCameraServer Class

This is in charge of making a server and opening a port to find a camera client. Each server only opens up 1 port so we run 2 of them in main.py

Params:
:param host: This is who we are looking for, 0.0.0.0 just means that we are looking for any IP address on the network. You could change this to the client's IP address to directly look for them. (Plz don't do, default is fine)
:param port: The port we are opening up on this instance. Default doesn't mean anything since we handle all of this in main.py, I'm just too lazy to change the argument order
:param frame_queue: This passes the frame queue (more on this in main.py and below)
"""

class StreamCameraServer(SocketServer):
    def __init__(self, host="0.0.0.0", port=9000, socket_type="TCP", vision_queue=None, display=True, fps=60):
        self.vision_queue = vision_queue
        self.display = display
        self.fps = fps
        self.frame_interval = 1.0 / fps
        self.shutdown = False
        super().__init__(host, port, socket_type)

    def receive_video_stream(self):
        log_writer = server_logger.get_logger()

        frame_rate = int(CAMERA_DEFAULT_FPS / self.fps)
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

        try:
            while True:
                # We get the raw data from the client here
                while len(data) < payload_size:
                    packet = conn.recv(4096)
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
                    packet = conn.recv(4096)
                    if not packet:
                        raise ConnectionResetError("Client disconnected")
                    data += packet

                frame_data = data[:msg_size]
                data = data[msg_size:]

                # Process the frame
                frame = pickle.loads(frame_data)
                frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

                if frame is None:
                    log_writer.warning("Received an empty or corrupted frame.")
                    continue

                # This is where our FPS comes from
                frame_count += 1
                if frame_count % frame_rate != 0:
                    continue

                if self.vision_queue is not None:
                    self.vision_queue.put(frame)

                del frame, frame_data
                gc.collect()

                # This is for cleanup but it doesnt really do anyting
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
        log_writer = server_logger.get_logger()
        frame_count = 0

        # Dictionary to store the fragments from the different clients
        fragments = {}
        frame_sizes = {}

        try:
            while True:
                try:
                    # First packet should contain header information
                    packet, addr = conn.recvfrom(65536)

                    if packet == b"PING":
                        log_writer.info(f"Received ping from {addr}")
                        continue

                    # Check if it's a header packet (contains number of chunks and total size)
                    if len(packet) == 16:  # 8 bytes for num_chunks + 8 bytes for total_size
                        num_chunks, total_size = struct.unpack("QQ", packet)
                        log_writer.info(f"Expecting {num_chunks} chunks for a {total_size} byte frame from {addr}")

                        client_key = f"{addr[0]}:{addr[1]}"
                        fragments[client_key] = {}
                        frame_sizes[client_key] = (num_chunks, total_size)
                        continue

                    # If we get here... then it's a data packet
                    # get the first sequence number from the packet(first 8 bytes)
                    seq_num = struct.unpack("Q", packet[:8])[0]
                    chunk = packet[8:]

                    client_key = f"{addr[0]}:{addr[1]}"
                    if client_key not in fragments:
                        log_writer.warning(f"Received chunk but no header from {addr}")
                        continue

                    fragments[client_key][seq_num] = chunk

                    # Check if we have all chunks
                    if client_key in frame_sizes and len(fragments[client_key]) == frame_sizes[client_key][0]:
                        num_chunks, total_size = frame_sizes[client_key]

                        sorted_chunks = [fragments[client_key][i] for i in range(num_chunks) if
                                         i in fragments[client_key]]

                        # If we're missing chunks then we skip the frame
                        if len(sorted_chunks) != num_chunks:
                            log_writer.warning(f"Missing chunks for frame: {len(sorted_chunks)}/{num_chunks}")
                            fragments[client_key] = {}
                            continue

                        frame_data = b"".join(sorted_chunks)

                        # Process the frame (Might optimize later bcz pickle is slow)
                        try:
                            frame = pickle.loads(frame_data)
                            frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

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