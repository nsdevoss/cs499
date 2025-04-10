import pickle
import time
import socket
import numpy as np
import open3d as o3d


class VisualizationClient:
    def __init__(self, server_ip, server_port):
        self.server_ip = server_ip
        self.server_port = server_port
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.latest_data = []
        self.connect_to_server()

    def connect_to_server(self):
        while True:
            try:
                print(f"Trying address: {self.server_ip}:{self.server_port}")
                assert isinstance(self.server_port, int), f"Port must be an integer, got {type(self.server_port)}"
                assert 1 <= self.server_port <= 65535, f"Port {self.server_port} is out of range"

                while not is_port_open(self.server_ip, self.server_port):
                    print(f"Port {self.server_port} is not ope yet, waiting 2 seconds")
                    time.sleep(2)
                self.client_socket.connect((self.server_ip, self.server_port))
                print(f"Successfully connected to: {self.server_ip}:{self.server_port}")
                self.receive_data_loop()

            except ConnectionRefusedError:
                print("Connection refused, retrying...")
                time.sleep(3)

            except OSError as e:
                print(f"OSError: {e}, retrying...")
                time.sleep(3)

    def receive_data(self):
        try:
            received_data = self.client_socket.recv(4096 * 2)
            if not received_data:
                raise ConnectionError("No data received or connection closed by the client.")

            try:
                data = pickle.loads(received_data)
            except pickle.UnpicklingError as e:
                raise ValueError("Failed to deserialize data")

            print(len(data))
            frame, points_3d, valid_mask = data

            self.latest_data = (frame, points_3d, valid_mask)
            print(f"Got latest data successfully")

        except Exception as e:
            print(f"Error receiving data: {e}")

    def receive_data_loop(self):
        while True:
            self.receive_data()
            if self.latest_data:
                self.display_visualization()
            time.sleep(0.1)

    def display_visualization(self):
        frame = self.latest_data[0]
        points_3d = self.latest_data[1]
        valid_mask = self.latest_data[2]

        try:
            valid_points = points_3d[valid_mask].reshape(-1, 3)
            valid_colors = frame[valid_mask][:, [2, 1, 0]].reshape(-1, 3) / 255.0

            valid_points[:, 2] = -valid_points[:, 2]
            valid_points[:, 1] = -valid_points[:, 1]

            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(valid_points.astype(np.float64))
            point_cloud.colors = o3d.utility.Vector3dVector(valid_colors.astype(np.float64))

            # Filtering points
            _, good_indices = point_cloud.remove_statistical_outlier(nb_neighbors=80, std_ratio=2.0)
            filtered_cloud = point_cloud.select_by_index(good_indices)
            o3d.visualization.draw_geometries([filtered_cloud])

        except Exception as e:
            print(f"Error visualizing: {e}")


def is_port_open(ip, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(2)
        return sock.connect_ex((ip, port)) == 0


if __name__ == "__main__":
    server_ip = "172.24.28.216"
    server_port = 9002

    client = VisualizationClient(server_ip=server_ip, server_port=server_port)
    client.connect_to_server()
