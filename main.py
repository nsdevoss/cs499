import multiprocessing
import platform
from src.server import server
from src.server.config import Config
from src.pi import emulator
from src.utils import utils
from src.vision.vision import Vision
from datetime import datetime
from src.WebServer.webserver import MyServer
from http.server import BaseHTTPRequestHandler, HTTPServer
from src.server.logger import server_logger, client_logger


MAX_QUEUE_SIZE = 10
processes = []


def start_server(port, frame_queue, display, fps):
    local_server = server.StreamCameraServer(port=port, frame_queue=frame_queue, display=display, fps=fps)
    local_server.receive_video_stream()

def start_emulator(ip_addr, video, stream_enabled, resolution, port):
    client = emulator.Emulator(ip_addr, video, stream_enabled, resolution, port)
    if stream_enabled:
        client.send_video_stream()
    else:
        client.send_video()

def start_vision_process(frame_queue, vision_args):
    vision = Vision(frame_queue=frame_queue, vision_args=vision_args)
    vision.start()

def start_webserver(logger):
    hostName = "0.0.0.0"
    serverPort = 8080

    MyServer.logger = logger
    webServer = HTTPServer((hostName, serverPort), MyServer)
    webServer.serve_forever()


def main(server_port, emulator_args, vision_args, video_args):
    global processes, ip_addr
    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    logs_to_zip = ["server.log"]

    system = platform.system()

    # Check if System is Windows or MacOS(Darwin)
    if system == "Windows":
        ip_addr = utils.windows_get_ip_address()
    elif system == "Darwin":
        ip_addr = utils.get_ip_address()

    frame_queue = multiprocessing.Queue()

    webserver_process = multiprocessing.Process(target=start_webserver, args=(server_logger,))
    webserver_process.start()
    processes.append(webserver_process)

    vision_process = multiprocessing.Process(target=start_vision_process, args=(frame_queue, vision_args))
    vision_process.start()
    processes.append(vision_process)

    server_logger.get_logger().info(f"Starting server on port: {server_port}")
    process = multiprocessing.Process(target=start_server, args=(server_port, frame_queue, video_args.get("display"), video_args.get("fps")), name=f"Server Process: {server_port}")

    process.start()
    processes.append(process)

    if emulator_args.get("enabled"):
        logs_to_zip.append("client.log")
        server_logger.get_logger().info("Running Emulated Client...")
        emu_process = multiprocessing.Process(target=start_emulator, args=(ip_addr, emulator_args.get("video_name"), emulator_args.get("stream_enabled"), server_port, video_args.get("resolution")), name=f"Emulator Process: {server_port}")
        emu_process.start()
        processes.append(emu_process)

    utils.create_killer(start_time=start_time, processes=processes, logs=logs_to_zip)

    for process in processes:
        process.join()


if __name__ == "__main__":
    Config.load_config()

    emulator_args = Config.get("emulator_arguments")
    video_args = Config.get("video_arguments")
    vision_args = Config.get("vision_arguments")
    server_port = Config.get("server_port", 9000)

    main(server_port, emulator_args, vision_args, video_args)

