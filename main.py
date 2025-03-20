import multiprocessing
from src.server import server
from src.server.config import Config
from src.pi import emulator
from src.utils import utils
from src.vision.vision import Vision
from src.server.logger import Logger
from datetime import datetime
# import platform

MAX_QUEUE_SIZE = 10
processes = []


def start_server(port, frame_queue, display, fps, server_logger):
    local_server = server.StreamCameraServer(port=port, frame_queue=frame_queue, display=display, fps=fps, logger=server_logger)
    local_server.receive_video_stream()

def start_emulator(ip_addr, video, stream_enabled, port, client_logger):
    client_logger.get_logger().info("start_emulator")
    client = emulator.Emulator(ip_addr, video, stream_enabled, port, client_logger)
    if stream_enabled:
        client.send_video_stream()
    else:
        client.send_video()

def start_vision_process(frame_queue, vision_arguments, server_logger):
    vision = Vision(frame_queue=frame_queue, action_arguments=vision_arguments, calibration_file=vision_arguments.get("calibration_file"), server_logger=server_logger)
    vision.start()


def main(server_port, emulator_args, vision_args, video_args, server_logger, client_logger):
    global processes, ip_addr
    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    logs_to_zip = ["server.log"]

    # ask if Windows or Mac machine
    ask_windows_or_mac = input("Enter Windows or Mac: \n")
    # system = platform.system()

    if ask_windows_or_mac == "Windows" or ask_windows_or_mac == "windows":
    # if system == "Windows":
        ip_addr = utils.windows_get_ip_address()
    elif ask_windows_or_mac == "mac" or ask_windows_or_mac == "Mac":
    # elif system == "iOS":
        ip_addr = utils.get_ip_address()
    else:
        print("Invalid Entry. Terminating...")
        return 0


    server_logger.get_logger().info(f"Got IP Address: {ip_addr}")

    frame_queue = multiprocessing.Queue()

    vision_process = multiprocessing.Process(target=start_vision_process, args=(frame_queue, vision_args, server_logger))
    vision_process.start()
    processes.append(vision_process)

    server_logger.get_logger().info(f"Starting server on port: {server_port}")
    process = multiprocessing.Process(target=start_server, args=(server_port, frame_queue, video_args.get("display"), video_args.get("fps"), server_logger), name=f"Server Process: {server_port}")

    process.start()
    processes.append(process)

    if emulator_args.get("enabled"):
        logs_to_zip.append("emulator.log")
        server_logger.get_logger().info("Running Emulated Client...")
        emu_process = multiprocessing.Process(target=start_emulator, args=(ip_addr, emulator_args.get("video_name"), emulator_args.get("stream_enabled"), server_port, client_logger), name=f"Emulator Process: {server_port}")
        emu_process.start()
        processes.append(emu_process)

    utils.create_killer(start_time=start_time, processes=processes, logs=logs_to_zip)

    for process in processes:
        process.join()
    server_logger.get_logger().info(f"Joined vision process: {vision_process.pid}")

if __name__ == "__main__":
    server_logger = Logger(name="ServerLogger", log_file="server.log")
    Config.set_logger(server_logger)
    Config.load_config()

    emulator_args = Config.get("emulator_arguments")
    video_args = Config.get("video_arguments")
    vision_args = Config.get("vision_arguments")
    server_port = Config.get("server_port", 9000)

    client_log_name = "client.log"
    if emulator_args.get("enabled"):
        client_log_name = "emulator.log"
    client_logger = Logger(name="ClientLogger", log_file=client_log_name)

    main(server_port, emulator_args, vision_args, video_args, server_logger, client_logger)

