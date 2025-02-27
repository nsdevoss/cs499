import argparse
import multiprocessing
from src.server import server
from src.server.config import Config
from src.pi import emulator
from src.utils import utils
from src.vision.vision import Vision
from src.server.logger import Logger
from datetime import datetime

MAX_QUEUE_SIZE = 10
processes = []


def start_server(port, frame_queue, display, server_logger):
    local_server = server.StreamCameraServer(port=port, frame_queue=frame_queue, display=display, logger=server_logger)
    local_server.receive_video_stream()


def start_emulator(ip_addr, video, port, client_logger):
    client = emulator.Emulator(ip_addr, video, port, logger=client_logger)
    client.send_video()


def start_vision_process(frame_queue, vision_arguments, server_logger):
    vision = Vision(frame_queue=frame_queue, action_arguments=vision_arguments, server_logger=server_logger)
    vision.start()


def main(use_emulator: bool, stitch: bool, compute_depth: bool, video_names: list, display: bool, server_ports: list, fps: int):
    global processes
    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    server_logger = Logger(name="ServerLogger", log_file="server.log")

    logs_to_zip = ["server.log"]
    ip_addr = utils.get_ip_address()

    vision_arguments = {"stitching": stitch, "depth_perception": compute_depth}
    for argument in vision_arguments:
        server_logger.get_logger().info(f"Vision action {argument}: {vision_arguments[argument]}")
    server_logger.get_logger().info(f"Got IP Address: {ip_addr}")

    frame_queue = multiprocessing.Queue()

    vision_process = multiprocessing.Process(target=start_vision_process, args=(frame_queue, vision_arguments, server_logger))
    vision_process.start()
    processes.append(vision_process)

    for port in server_ports:
        server_logger.get_logger().info(f"Starting server on port: {port}")
        process = multiprocessing.Process(target=start_server, args=(port, frame_queue, display, server_logger),
                                          name=f"Server Process: {port}")
        process.start()
        processes.append(process)

    if use_emulator:
        client_logger = Logger(name="EmulatorLogger", log_file="emulator.log")
        logs_to_zip.append("emulator.log")
        server_logger.get_logger().info("Running Emulated Client...")

        for idx, port in enumerate(server_ports):
            video = video_names[min(idx, len(video_names) - 1)]
            client_logger.get_logger().info(f"Starting emulator on: {port} with video: {video}")
            emulator_process = multiprocessing.Process(target=start_emulator,
                                                       args=(ip_addr, video, port, client_logger))
            emulator_process.start()
            processes.append(emulator_process)

    utils.create_killer(start_time=start_time, logs=logs_to_zip)

    for process in processes:
        process.join()
    server_logger.get_logger().info(f"Joined vision process: {vision_process.pid}")


if __name__ == "__main__":
    server_logger = Logger(name="ServerLogger", log_file="server.log")
    Config.set_logger(server_logger)
    Config.load_config()

    use_emulator = Config.get("use_emulator", False)
    stitch = Config.get("vision_arguments.stitch", False)
    compute_depth = Config.get("vision_arguments.compute_depth", False)
    video_names = Config.get("video_arguments.video_names", ["zoom_out"])
    display = Config.get("video_arguments.display", False)
    server_ports = Config.get("server_ports", [9000, 9001])
    fps = Config.get("video_arguments.fps", 30)

    main(use_emulator, stitch, compute_depth, video_names, display, server_ports, fps)
