import socket
from src.server.logger import server_logger


"""
SocketServer Class

This is the base class that just gives some basic socket connection stuff

Params:
:param host: This is who we are looking for, 0.0.0.0 just means that we are looking for any IP address on the network. You could change this to the client's IP address to directly look for them. (Plz don't do, default is fine)
:param port: The port we are opening up on this instance.
:param socket_type: This determines if the server will be TCP or UDP. You can look up the difference but the StreamCameraServer uses TCP
"""


class SocketServer:
    def __init__(self, host="0.0.0.0", port=9000, socket_type="TCP"):
        self.host = host
        self.port = port
        self.socket_type = socket_type.upper()
        self.is_shutdown = False
        socket_type = None
        try:
            assert self.socket_type == "TCP" or self.socket_type == "UDP", f'Socket type must be "TCP" or "UDP", got: {self.socket_type}'
            socket_type = socket.SOCK_STREAM if self.socket_type == "TCP" else socket.SOCK_DGRAM
        except AssertionError as e:
            server_logger.get_logger().error(e)
        self.server_socket = socket.socket(socket.AF_INET, socket_type)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        if self.socket_type == "TCP":
            self.server_socket.listen(1)
        server_logger.get_logger().info(f"Listening on {self.host}:{self.port}")

    def shutdown(self):
        if not self.is_shutdown:
            server_logger.get_logger().info(f"Shutting down server: {self.host}:{self.port}")
            self.server_socket.close()
            self.is_shutdown = True
            server_logger.get_logger().info(f"Successfully shut down server: {self.host}:{self.port}")
