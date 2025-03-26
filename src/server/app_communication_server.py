import socket
import cv2
import numpy
import pickle
from src.server.server import SocketServer


class AppCommunicationServer(SocketServer):
    def __init__(self):
        super().__init__()


if __name__ == "__main__":
    server = AppCommunicationServer()
    print(server.host)
    print(server.port)
    print(server.socket_type)

