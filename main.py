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


def main(use_emulator: bool, stitch: bool, compute_depth: bool, video_name: str, display: bool, server_port: int, fps: int):
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

    server_logger.get_logger().info(f"Starting server on port: {server_port}")
    process = multiprocessing.Process(target=start_server, args=(server_port, frame_queue, display, server_logger), name=f"Server Process: {server_port}")

    process.start()
    processes.append(process)
    if use_emulator:
        client_logger = Logger(name="EmulatorLogger", log_file="emulator.log")
        logs_to_zip.append("emulator.log")
        server_logger.get_logger().info("Running Emulated Client...")

        emu_process = multiprocessing.Process(target=start_emulator, args=(ip_addr, video_name, server_port, client_logger))
        emu_process.start()
        processes.append(emu_process)

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
    video_name = Config.get("video_arguments.video_names", "zoom_out")
    display = Config.get("video_arguments.display", False)
    server_port = Config.get("server_ports", 9000)
    fps = Config.get("video_arguments.fps", 30)

    main(use_emulator, stitch, compute_depth, video_name, display, server_port, fps)
