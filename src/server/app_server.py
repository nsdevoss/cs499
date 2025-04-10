import socket
import time
from src.vision.detection import detector
from src.server.logger import server_logger
from src.server.server import SocketServer


class AppCommunicationServer(SocketServer):
    """
    ###### Setting up the socket is already done in parent class, so you just need to worry about send/receive ######
    Examples:
        For detailed explanations on how TCP and UDP receiving work check src/server/camera_server.py
        For detailed explanations on how TCP and UDP sending work check src/pi/emulator.py
        For examples on how to start the server process look at main.py on any start_process()
    Tips:
        You can choose whether it is better for TCP or UDP for connecting to the app, but it will most likely be TCP.
    """
    def __init__(self, object_detected, host="0.0.0.0", port=9001, socket_type="TCP"):
        super().__init__(host, port, socket_type)
        self.log_writer = server_logger.get_logger()
        self.object_detected = object_detected

    def connect_to_app(self):
        while True:
            self.log_writer.info("Waiting for a connection...")
            conn, addr = self.server_socket.accept()
            self.log_writer.info(f"Got a connection from: {addr}")
            self.send_message(conn, addr)


    def send_message(self, conn, addr):
        try:
            init_msg = "This is the first message\n"
            conn.send(init_msg.encode())
            self.log_writer.info(f"Sent initial message: {init_msg}")
            
            while True:
                if self.object_detected.value:
                    msg = f"Detected object\n"
                    self.log_writer.info(f"Sending message: {msg}")
                    conn.send(msg.encode())
                    self.log_writer.info(f"Sent message: {msg} to {addr}")
                else:
                    msg = f"No detected object\n"
                    self.log_writer.info(f"Sending message: {msg}")
                    conn.send(msg.encode())
                    self.log_writer.info(f"Sent message: {msg} to {addr}")

        except Exception as e:
            # conn.close()
            self.log_writer.error(f"Error sending message: {e}")


if __name__ == "__main__":
    server = AppCommunicationServer()
    server.connect_to_app()
    print(server.host)
    print(server.port)
    print(server.socket_type)

