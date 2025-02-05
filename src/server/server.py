import socket
import cv2
import pickle
import struct


class StreamCameraServer:
    def __init__(self, host="0.0.0.0", port=9000):
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        print(f"Server listening on {self.host}:{self.port}")

    def receive_video_stream(self):
        while True:
            print("Waiting for a connection...")
            conn, addr = self.server_socket.accept()
            print(f"Connection from {addr}")

            data = b""
            payload_size = struct.calcsize("Q")

            try:
                while True:
                    while len(data) < payload_size:
                        packet = conn.recv(4096)
                        if not packet:
                            raise ConnectionResetError("Client disconnected")
                        data += packet

                    packed_msg_size = data[:payload_size]
                    data = data[payload_size:]
                    msg_size = struct.unpack("Q", packed_msg_size)[0]

                    while len(data) < msg_size:
                        packet = conn.recv(4096)
                        if not packet:
                            raise ConnectionResetError("Client disconnected")
                        data += packet

                    frame_data = data[:msg_size]
                    data = data[msg_size:]

                    # Decode frame
                    frame = pickle.loads(frame_data)
                    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

                    if frame is None:
                        print("Warning: Received an empty or corrupted frame.")
                        continue

                    cv2.imshow("Raspberry Pi Camera Stream", frame)
                    if cv2.waitKey(1) == ord("q"):
                        break

            except (ConnectionResetError, BrokenPipeError) as e:
                print(f"Error: {e}. Client disconnected. Waiting for a new connection...")

            except Exception as e:
                print(f"Server error: {e}")

            finally:
                conn.close()
                cv2.destroyAllWindows()
                print("Connection closed.")


if __name__ == "__main__":
    server = StreamCameraServer()
    server.receive_video_stream()
