import socket
import time

class RaspberryPiClient:
    def __init__(self, server_ip, server_port):
        self.server_ip = server_ip
        self.server_port = server_port

    def send_data(self, message):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.connect((self.server_ip, self.server_port))
                print(f"Connected to server {self.server_ip}:{self.server_port}")
                sock.sendall(message.encode('utf-8'))
                print(f"Sent: {message}")
            except Exception as e:
                print(f"Client error: {e}")
            finally:
                print("Closing client connection.")

if __name__ == "__main__":
    SERVER_IP = "192.168.1.69" # This needs to be the IP of the laptop
    SERVER_PORT = 9000

    client = RaspberryPiClient(server_ip=SERVER_IP, server_port=SERVER_PORT)
    
    try:
        while True:
            client.send_data("Hello from Raspberry Pi!")
            time.sleep(2)  # Send data every 2 seconds
    except KeyboardInterrupt:
        print("Client stopped.")
