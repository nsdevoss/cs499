import gc
import socket
import cv2
import pickle
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
    def __init__(self, host="0.0.0.0", port=9000, socket_type="TCP", frame_queue=None, display=True, fps=60):
        self.frame_queue = frame_queue
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
            conn, addr = self.server_socket.accept()
            log_writer.info(f"Got connection from {addr}")

            # This just makes an empty byte buffer for incoming data
            data = b""
            payload_size = struct.calcsize("Q")  # This is the first 8 bytes that the server reads from the incoming data
            frame_count = 0
            try:
                while True:
                    # We get the raw data from the client here
                    while len(data) < payload_size:
                        packet = conn.recv(4096)
                        if not packet:
                            log_writer.error(f"No packet received: {packet}")
                            raise ConnectionResetError("No packet received")
                        data += packet

                    packed_msg_size = data[:payload_size]  # This gets the first 8 bytes which contain the frame size
                    data = data[payload_size:]  # This is everything else
                    msg_size = struct.unpack("Q", packed_msg_size)[0]  # This unpacks the frame size to get the actual size of the incoming frame

                    # This makes sure that the whole frame is being received
                    while len(data) < msg_size:
                        packet = conn.recv(4096)
                        if not packet:
                            raise ConnectionResetError("Client disconnected")
                        data += packet

                    frame_data = data[:msg_size]  # This is the received frame data
                    data = data[msg_size:]  # This removes the extracted frame data from "data"

                    frame = pickle.loads(frame_data)  # Deserialize the frame into a format OpenCV can process
                    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)  # THIS IS THE ACTUAL FRAME IMAGE!!!!!!!!!!!
                    # obj = vision.Vision(frame)

                    if frame is None:
                        log_writer.warning("Received an empty or corrupted frame.")
                        continue

                    # We input the frame into the frame queue along with its server port
                    # We do this because when we get the frame out we will know where it came from
                    frame_count += 1
                    if frame_count % frame_rate != 0:
                        continue
                    if self.frame_queue is not None:
                        self.frame_queue.put(frame)

                    del frame, frame_data
                    gc.collect()

                    if self.shutdown:
                        break

            except (ConnectionResetError, BrokenPipeError) as e:
                log_writer.error(f"{e}. Client disconnected. Waiting for a new connection...")

            except Exception as e:
                log_writer.error(f"{e}")

            finally:
                conn.close()
                cv2.destroyAllWindows()
                log_writer.info(f"Closed socket connection with {addr}.")

    # Need to find a way to use this, rn we just KILL everything
    def shutdown(self):
        server_logger.get_logger().info("Shutting down server")
        self.server_socket.close()
        self.shutdown = True
        server_logger.get_logger().info("Successfully shut down server")


if __name__ == "__main__":
    server = StreamCameraServer()
    server.receive_video_stream()
