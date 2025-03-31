import multiprocessing
import platform
from src.server.camera_server import StreamCameraServer
from src.server.config import Config
from src.pi import emulator
from src.utils import utils
from src.vision.vision import Vision
from datetime import datetime
from src.WebServer.webserver import MyServer
from http.server import BaseHTTPRequestHandler, HTTPServer
from src.server.logger import server_logger, client_logger
import cv2


MAX_QUEUE_SIZE = 10
processes = []


def play(display_queue):
    while True:
        if display_queue is not None and not display_queue.empty():
            frame, disp = display_queue.get()
            print("Frame taken from display queue: DISPLAY")
            if frame is not None:
                cv2.imshow("Display", disp)
            else:
                print("Display frame not found")

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()



def start_server(server_arguments,vision_queue, fps):
    host = server_arguments.get("host")
    port = server_arguments.get("port")
    socket_type = server_arguments.get("socket_type")

    local_server = StreamCameraServer(host=host, port=port, socket_type=socket_type, vision_queue=vision_queue, fps=fps)
    local_server.receive_video_stream()


def start_emulator(ip_addr, video, stream_enabled, socket_type, encode_quality, resolution, port):
    assert 0 <= resolution <= 100
    client = emulator.Emulator(ip_addr, video, stream_enabled, socket_type, encode_quality, resolution, port)
    if stream_enabled:
        client.send_video_stream()
    else:
        client.send_video()


def start_vision_process(vision_queue, display_queue, vision_args):
    vision = Vision(frame_queue=vision_queue, display_queue=display_queue, vision_args=vision_args)
    vision.start()


def start_webserver():
    hostName = "0.0.0.0"
    serverPort = 8080

    webServer = HTTPServer((hostName, serverPort), MyServer)
    webServer.serve_forever()


def main(server_arguments, emulator_args, vision_args, video_args):
    global processes, ip_addr
    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    logs_to_zip = ["server.log", "webserver.log"]

    system = platform.system()

    # Check if System is Windows or MacOS(Darwin)
    if system == "Windows":
        ip_addr = utils.windows_get_ip_address()
    elif system == "Darwin":
        ip_addr = utils.get_ip_address()

    # I am going to assume we will always run vision and not just display the frame so no need for 3 queues only 2
    display_queue = None
    vision_queue = None

    if vision_args.get("enabled"):
        vision_queue = multiprocessing.Queue()
    if video_args.get("display") and vision_args.get("enabled"):
        display_queue = multiprocessing.Queue()


    webserver_process = multiprocessing.Process(target=start_webserver)
    webserver_process.start()
    processes.append(webserver_process)

    vision_process = multiprocessing.Process(target=start_vision_process, args=(vision_queue, display_queue, vision_args))
    vision_process.start()
    server_logger.get_logger().info(f"Started vision process with pid: {vision_process.pid}")
    processes.append(vision_process)

    if video_args.get("display"):
        play_process = multiprocessing.Process(target=play, args=(display_queue,))
        play_process.start()
        server_logger.get_logger().info(f"Started display process with pid: {play_process.pid}")
        processes.append(play_process)

    server_logger.get_logger().info(f"Starting server on port: {server_arguments.get("port")}")
    server_process = multiprocessing.Process(target=start_server, args=(server_arguments, vision_queue, video_args.get("fps")), name=f"Server Process: {server_arguments.get("port")}")
    server_process.start()
    server_logger.get_logger().info(f"Started server process with pid: {server_process.pid}")
    processes.append(server_process)

    if emulator_args.get("enabled"):
        logs_to_zip.append("client.log")

        server_logger.get_logger().info(f"Starting emulated client looking at: {ip_addr}:{server_arguments.get("port")}.")
        emu_process = multiprocessing.Process(target=start_emulator, args=(ip_addr, emulator_args.get("video_name"), emulator_args.get("stream_enabled"), server_arguments.get("port"), server_arguments.get("socket_type"), emulator_args.get("encode_quality"),video_args.get("resolution")), name="Emulator Process")
        emu_process.start()
        server_logger.get_logger().info(f"Started emulator process with pid: {emu_process.pid}")
        processes.append(emu_process)

    utils.create_killer(start_time=start_time, processes=processes, logs=logs_to_zip)

    for process in processes:
        process.join()


if __name__ == "__main__":
    Config.load_config()

    emulator_args = Config.get("emulator_arguments")
    video_args = Config.get("video_arguments")
    vision_args = Config.get("vision_arguments")
    server_arguments = Config.get("server_arguments")

    main(server_arguments, emulator_args, vision_args, video_args)

