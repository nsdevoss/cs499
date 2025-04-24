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
    def __init__(self, object_detect_queue, host="0.0.0.0", port=9001, socket_type="TCP"):
        super().__init__(host, port, socket_type)
        self.log_writer = server_logger.get_logger()
        self.object_detect_queue = object_detect_queue

    def connect_to_app(self):
        while True:
            self.log_writer.info(f"Waiting for a connection on {self.host}:{self.port}")
            conn, addr = self.server_socket.accept()
            self.object_detect_queue.empty()
            self.log_writer.info(f"Got a connection from: {addr}")
            self.send_message(conn, addr)

    def send_message(self, conn, addr):
        try:
            init_msg = "This is the first message\n"
            conn.send(init_msg.encode())
            self.log_writer.info(f"Sent initial message: {init_msg}")
            
            while True:
                if self.object_detect_queue is not None and not self.object_detect_queue.empty():
                    entry = self.object_detect_queue.get()
                    # json_entry = json.dumps(str(entry))
                    msg = f"Object at distance: {entry.get('distance'):.2f}m, center: {entry.get('center')}, position: {entry.get("position")}, persistence: {entry.get('persistence')}\n"
                    conn.send(msg.encode())
                    # conn.send(json_entry.encode())
                    self.log_writer.info(f"Sent message {msg} to {addr}")
                else:
                    continue

        except (ConnectionResetError, BrokenPipeError) as e:
            server_logger.get_logger().info(f"{e}, APP CLIENT disconnected, Waiting for a new connection...")
        except Exception as e:
            conn.close()
            self.log_writer.error(f"Error sending message: {e}")


if __name__ == "__main__":
    server = AppCommunicationServer()
    server.connect_to_app()
    print(server.host)
    print(server.port)
    print(server.socket_type)

