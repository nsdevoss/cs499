import pickle
import time
import socket
import struct
import select
import numpy as np
import open3d as o3d

############## This needs to be run as a singular file, it is not part of the program!!!!!!!! ######################

class VisualizationClient:
    def __init__(self, server_ip, server_port):
        self.server_ip = server_ip
        self.server_port = server_port
        self.client_socket = None
        self.visualizer = None
        self.point_cloud = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]  # This is just a dummy value
        self.point_cloud_added = False
        self.last_data_time = time.time()
        self.setup_visualization()

    def setup_visualization(self):
        self.visualizer = o3d.visualization.Visualizer()
        self.visualizer.create_window("Point Cloud Visualization", width=1280, height=720)

        self.point_cloud = o3d.geometry.PointCloud()

        self.visualizer.add_geometry(self.point_cloud)

        view_control = self.visualizer.get_view_control()
        view_control.set_front([0, 0, -1])
        view_control.set_up([0, -1, 0])

        view_control.set_zoom(0.5)

        opt = self.visualizer.get_render_option()
        opt.point_size = 5.0
        opt.background_color = np.array([0.1, 0.1, 0.1])

    def connect_to_server(self):
        print(f"Connecting to {self.server_ip}:{self.server_port}...")
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.server_ip, self.server_port))
        print("Connected to server")

    def get_right_length(self, length):
        data = bytearray()
        while len(data) < length:
            packet = self.client_socket.recv(length - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data

    def receive_data(self):
        ready_to_read, _, _ = select.select([self.client_socket], [], [], 0)

        if not ready_to_read:
            return None

        try:
            msg_len_bytes = self.get_right_length(4)
            if not msg_len_bytes:
                print("Connection closed by server")
                return None

            msg_len = struct.unpack('>I', msg_len_bytes)[0]

            serialized_data = self.get_right_length(msg_len)
            if not serialized_data:
                print("Connection closed by server during data transfer")
                return None

            data = pickle.loads(serialized_data)
            now = time.time()
            print(f"Received packet from server after: {(now - self.last_data_time):.2f}s")
            return data

        except Exception as e:
            print(f"Error receiving data: {e}")
            return None

    def update_visualization(self, data):
        try:
            frame, points_3d, valid_mask = data

            valid_points = points_3d[valid_mask].reshape(-1, 3)
            valid_colors = frame[valid_mask][:, [2, 1, 0]].reshape(-1, 3) / 255.0

            valid_points[:, 2] = -valid_points[:, 2]
            valid_points[:, 1] = -valid_points[:, 1]

            if len(valid_points) > 0:
                depth_threshold = np.percentile(np.abs(valid_points[:, 2]), 95)
                depth_mask = np.abs(valid_points[:, 2]) < depth_threshold
                valid_points = valid_points[depth_mask]
                valid_colors = valid_colors[depth_mask]

            if len(valid_points) > 0:
                min_z = np.min(valid_points[:, 2])
                if min_z <= 0:
                    valid_points[:, 2] = valid_points[:, 2] - min_z + 0.1

                valid_points[:, 2] = (valid_points[:, 2] - np.min(valid_points[:, 2])) / (np.max(valid_points[:, 2]) - np.min(valid_points[:, 2]))

            self.point_cloud.points = o3d.utility.Vector3dVector(valid_points)
            self.point_cloud.colors = o3d.utility.Vector3dVector(valid_colors)

            _, indices = self.point_cloud.remove_statistical_outlier(nb_neighbors=40, std_ratio=2.0)
            filtered_cloud = self.point_cloud.select_by_index(indices)

            self.point_cloud.points = filtered_cloud.points
            self.point_cloud.colors = filtered_cloud.colors

            if not self.point_cloud_added:
                self.visualizer.add_geometry(self.point_cloud)
                self.point_cloud_added = True

            self.visualizer.update_geometry(self.point_cloud)
            self.visualizer.reset_view_point(True)
            view_control = self.visualizer.get_view_control()

            view_control.set_zoom(0.35)
            self.last_data_time = time.time()
        except Exception as e:
            print(f"Error updating visualization: {e}")

    def run(self):
        try:
            self.connect_to_server()

            while True:
                self.visualizer.poll_events()
                self.visualizer.update_renderer()

                data = self.receive_data()
                if data:
                    self.update_visualization(data)

                time.sleep(0.01)

        except Exception as e:
            print(f"Error: {e}")
        finally:
            if self.client_socket:
                self.client_socket.close()
            if self.visualizer:
                self.visualizer.destroy_window()


if __name__ == "__main__":
    server_ip = "172.24.28.216"
    server_port = 9002

    client = VisualizationClient(server_ip, server_port)
    client.run()