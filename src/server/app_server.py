import socket
import cv2
import numpy
import pickle
from src.server.server import SocketServer


class AppCommunicationServer(SocketServer):
    def __init__(self, host="0.0.0.0", port=9001, socket_type="TCP"):
        super().__init__(host, port, socket_type)


if __name__ == "__main__":
    server = AppCommunicationServer()
    print(server.host)
    print(server.port)
    print(server.socket_type)

