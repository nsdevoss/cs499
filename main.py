import multiprocessing
from src.server.camera_server import StreamCameraServer
from src.pi import emulator
from src.utils import utils
from src.vision.vision import Vision
from src.vision.vision import create_3d_map
from datetime import datetime
from src.WebServer.app import WebServerDisplay
from http.server import HTTPServer
from src.server.logger import server_logger
from src.server.app_server import AppCommunicationServer
import cv2
import time
import src.LocalCommon as lc
from src.utils.config import Config


processes = []


def play(display_queue, depth_map_enabled):
    points_3d = None
    valid_mask = None
    frame = None
    frame_count = 0
    start_time = time.time()

    while True:
        if display_queue is not None and not display_queue.empty():
            disp, frame, points_3d, valid_mask = display_queue.get()
            if disp is not None:
                cv2.imshow("Display", disp)
            else:
                print("Display frame not found")

            frame_count += 1

            now = time.time()
            if now - start_time >= 60.0:
                fps = frame_count / (now - start_time)
                server_logger.get_logger().info(f"DISPLAY FPS: {fps:.2f}")
                frame_count = 0
                start_time = now

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        if key == ord('c') and depth_map_enabled:
            if not [x for x in (points_3d, valid_mask, frame) if x is None]:
                create_3d_map(points_3d=points_3d, valid_mask=valid_mask, frame=frame)
            else:
                server_logger.get_logger().error(
                    f"Error generating 3D visualization, points_3d: {points_3d}, valid_mask: {valid_mask}")

    cv2.destroyAllWindows()


def start_camera_server(vision_queue, camera_server_args):
    host = camera_server_args.get("host")
    port = camera_server_args.get("port")
    socket_type = camera_server_args.get("socket_type")
    fps = camera_server_args.get("fps")

    local_server = StreamCameraServer(host=host, port=port, socket_type=socket_type, vision_queue=vision_queue, fps=fps)
    local_server.receive_video_stream()


def start_app_server():
    app_server = AppCommunicationServer()
    app_server.connect_to_app()

def start_webserver(display_queue):
    web_server = WebServerDisplay(display_queue=display_queue)
    web_server.run()

def start_emulator(ip_addr, emulator_args, camera_server_args):
    video = emulator_args.get("video_name")
    stream_enabled = emulator_args.get("stream_enabled")
    encode_quality = emulator_args.get("encode_quality")
    port = camera_server_args.get("port")
    socket_type = camera_server_args.get("socket_type")
    scale = camera_server_args.get("scale")

    client = emulator.Emulator(server_ip=ip_addr, video=video, stream_enabled=stream_enabled, server_port=port, socket_type=socket_type, encode_quality=encode_quality, scale=scale)
    if stream_enabled:
        client.send_video_stream()
    else:
        client.send_video()


def start_vision_process(vision_queue, display_queue, vision_args, scale):
    vision = Vision(frame_queue=vision_queue, display_queue=display_queue, vision_args=vision_args, scale=scale)
    vision.start()


def main(camera_server_args, emulator_args, vision_args):
    global processes, ip_addr
    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    logs_to_zip = ["server.log", "webserver.log"]
    ip_addr = utils.get_ip_address()
    print(ip_addr)

    # I am going to assume we will always run vision and never display only the raw frame
    display_queue = None  # Use for play process for debug now, SHOULD be used by the Webserver on release
    vision_queue = None  # Use for Vision class

    if vision_args.get("enabled"):
        vision_queue = multiprocessing.Queue()
        display_queue = multiprocessing.Queue()

    ###### Web Server start process ######
    webserver_process = multiprocessing.Process(target=start_webserver, args=(display_queue,))
    webserver_process.start()
    processes.append(webserver_process)

    ###### Vision start process ######
    vision_process = multiprocessing.Process(target=start_vision_process, args=(vision_queue, display_queue, vision_args, camera_server_args.get("scale")))
    vision_process.start()
    server_logger.get_logger().info(f"Started vision process with pid: {vision_process.pid}")
    processes.append(vision_process)

    ###### Display start process ######
    # play_process = multiprocessing.Process(target=play, args=(display_queue, vision_args.get("depth_map_capture")))
    # play_process.start()
    # server_logger.get_logger().info(f"Started display process with pid: {play_process.pid}")
    # processes.append(play_process)

    ###### Camera Server start process ######
    server_logger.get_logger().info(f"Starting server on port: {camera_server_args.get('port')}")
    camera_server_process = multiprocessing.Process(target=start_camera_server,args=(vision_queue, camera_server_args), name=f"Server Process: {camera_server_args.get('port')}")
    camera_server_process.start()
    server_logger.get_logger().info(f"Started server process with pid: {camera_server_process.pid}")
    processes.append(camera_server_process)

    ###### App Server start process
    # server_logger.get_logger().info(f"Starting app server on port: 9001")
    # app_server_process = multiprocessing.Process(target=start_app_server)
    # app_server_process.start()
    # server_logger.get_logger().info(f"Started app server process with pid: {app_server_process.pid}")
    # processes.append(app_server_process)

    ###### Emulator start process ######
    if emulator_args.get("enabled"):
        logs_to_zip.append("client.log")

        server_logger.get_logger().info(f"Starting emulated client looking at: {ip_addr}:{camera_server_args.get('port')}.")
        emu_process = multiprocessing.Process(target=start_emulator, args=(ip_addr,emulator_args, camera_server_args), name="Emulator Process")
        emu_process.start()
        server_logger.get_logger().info(f"Started emulator process with pid: {emu_process.pid}")
        processes.append(emu_process)

    ###### Create kill button ######
    utils.create_killer(start_time=start_time, processes=processes, logs=logs_to_zip)

    for process in processes:
        process.join()


if __name__ == "__main__":
    Config.load_config()

    camera_server_args = Config.get("camera_server_arguments")
    emulator_args = Config.get("emulator_arguments")
    vision_args = Config.get("vision_arguments")

    main(camera_server_args, emulator_args, vision_args)
