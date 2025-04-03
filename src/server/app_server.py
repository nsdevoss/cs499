import socket
import cv2
import numpy
import pickle
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
    def __init__(self, host="0.0.0.0", port=9001, socket_type="TCP"):
        super().__init__(host, port, socket_type)


if __name__ == "__main__":
    server = AppCommunicationServer()
    print(server.host)
    print(server.port)
    print(server.socket_type)

