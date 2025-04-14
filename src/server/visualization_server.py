import struct
import pickle
from src.server.server import SocketServer
from src.server.logger import server_logger


class VisualizationServer(SocketServer):
    def __init__(self, info_queue, host="0.0.0.0", port=9002):
        super().__init__(host, port)
        self.info_queue = info_queue
        self.log_writer = server_logger.get_logger()

    def connect(self):
        while True:
            self.log_writer.info(f"Waiting for a connection on {self.host}:{self.port} for visualization")
            conn, addr = self.server_socket.accept()
            self.log_writer.info(f"Got a connection from: {addr}")

            while True:
                try:
                    self.send_data(conn, addr)
                except Exception as e:
                    self.log_writer.error(f"Connection with {addr} lost: {e}")
                    break

    def send_data(self, conn, addr):
        try:
            if self.info_queue is not None and not self.info_queue.empty():
                frame, points_3d, valid_mask = self.info_queue.get()
                data = (frame, points_3d, valid_mask)

                serialized = pickle.dumps(data)

                msg_len = len(serialized)
                conn.sendall(struct.pack('>I', msg_len))

                conn.sendall(serialized)
                self.log_writer.info(f"Sent message to {addr}")

        except (ConnectionResetError, BrokenPipeError) as e:
            self.log_writer.error(f"{e}, Visualization Client disconnected, Waiting for a new connection...")
        except Exception as e:
            self.log_writer.error(f"Error sending data to client: {e}")
            raise
