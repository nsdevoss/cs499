import multiprocessing
from src.pi import emulator
from src.utils import utils
from datetime import datetime
from src.utils.config import Config
from src.utils.utils import execute_pi_client
from src.server.camera_server import StreamCameraServer
from src.server.visualization_server import VisualizationServer
from src.server.app_server import AppCommunicationServer
from src.WebServer.web_server import WebServerDisplay
from src.server.logger import server_logger
from src.vision.vision import Vision


processes = []


def start_camera_server(vision_queue, camera_server_args):
    host = camera_server_args.get("host")
    port = camera_server_args.get("port")
    socket_type = camera_server_args.get("socket_type")
    fps = camera_server_args.get("fps")

    local_server = StreamCameraServer(host=host, port=port, socket_type=socket_type, vision_queue=vision_queue, fps=fps)
    local_server.receive_video_stream()


def start_app_server(object_detected):
    app_server = AppCommunicationServer(object_detected)
    app_server.connect_to_app()

def start_webserver(display_queue):
    web_server = WebServerDisplay(display_queue=display_queue)
    web_server.run()

def start_emulator(ip_addr, emulator_args, camera_server_args):
    video = emulator_args.get("video_name")
    stream_enabled = emulator_args.get("stream_enabled")
    encode_quality = emulator_args.get("encode_quality")
    port = camera_server_args.get("port")
    scale = camera_server_args.get("scale")
    socket_type = camera_server_args.get("socket_type")

    client = emulator.Emulator(server_ip=ip_addr, video=video, stream_enabled=stream_enabled, server_port=port, socket_type=socket_type, encode_quality=encode_quality, scale=scale)
    if stream_enabled:
        client.send_video_stream()
    else:
        client.send_video()


def start_vision_process(vision_queue, display_queue, object_detect_queue, info_queue, vision_args, scale, object_detected):
    vision = Vision(frame_queue=vision_queue, display_queue=display_queue, info_queue=info_queue, object_detect_queue=object_detect_queue,vision_args=vision_args, scale=scale, object_detected=object_detected)
    vision.start()

def start_visualization_process(info_queue):
    visualization_server = VisualizationServer(info_queue)
    visualization_server.connect()


def main(camera_server_args, pi_args, emulator_args, vision_args, object_detected):
    global processes, ip_addr
    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    logs_to_zip = ["server.log", "webserver.log"]
    ip_addr = utils.get_ip_address()
    print(ip_addr)

    # I am going to assume we will always run vision and never display only the raw frame
    display_queue = None  # Use for play process for debug now, SHOULD be used by the Webserver on release
    vision_queue = None  # Use for Vision class
    info_queue = None  # This is for the other computer to use
    object_detect_queue = None

    if vision_args.get("enabled"):
        vision_queue = multiprocessing.Queue()
        display_queue = multiprocessing.Queue()
        object_detect_queue = multiprocessing.Queue()
        if vision_args.get("depth_map_capture"):
            info_queue = multiprocessing.Queue()

    ###### Start Visualization Server ######
    if vision_args.get("depth_map_capture"):
        server_logger.get_logger().info(f"Starting visualization server on port: 9002")
        visualization_server_process = multiprocessing.Process(target=start_visualization_process, args=(info_queue,))
        visualization_server_process.start()
        server_logger.get_logger().info(f"Started visualization server process with pid: {visualization_server_process.pid}")
        processes.append(visualization_server_process)

    ###### Web Server start process ######
    webserver_process = multiprocessing.Process(target=start_webserver, args=(display_queue,))
    webserver_process.start()
    processes.append(webserver_process)

    ###### Vision start process ######
    vision_process = multiprocessing.Process(target=start_vision_process, args=(vision_queue, display_queue, object_detect_queue, info_queue,vision_args, camera_server_args.get("scale"), object_detected))
    vision_process.start()
    server_logger.get_logger().info(f"Started vision process with pid: {vision_process.pid}")
    processes.append(vision_process)

    ###### Camera Server start process ######
    server_logger.get_logger().info(f"Starting server on port: {camera_server_args.get('port')}")
    camera_server_process = multiprocessing.Process(target=start_camera_server,args=(vision_queue, camera_server_args), name=f"Server Process: {camera_server_args.get('port')}")
    camera_server_process.start()
    server_logger.get_logger().info(f"Started server process with pid: {camera_server_process.pid}")
    processes.append(camera_server_process)

    ###### App Server start process ######
    server_logger.get_logger().info(f"Starting app server on port: 9001")
    app_server_process = multiprocessing.Process(target=start_app_server, args=(object_detected,))
    app_server_process.start()
    server_logger.get_logger().info(f"Started app server process with pid: {app_server_process.pid}")
    processes.append(app_server_process)


    ###### Emulator start process ######
    if emulator_args.get("enabled"):
        logs_to_zip.append("client.log")

        server_logger.get_logger().info(f"Starting emulated client looking at: {ip_addr}:{camera_server_args.get('port')}.")
        emu_process = multiprocessing.Process(target=start_emulator, args=(ip_addr,emulator_args, camera_server_args), name="Emulator Process")
        emu_process.start()
        server_logger.get_logger().info(f"Started emulator process with pid: {emu_process.pid}")
        processes.append(emu_process)

    else:
        pi_client_process = multiprocessing.Process(
            target=execute_pi_client,
            args=(
                pi_args,
                ip_addr,
                camera_server_args.get("port"),
                camera_server_args.get("socket_type"),
                camera_server_args.get("scale"),
            )
        )
        pi_client_process.start()
        server_logger.get_logger().info(f"Started Pi client process with pid: {pi_client_process.pid}")
        processes.append(pi_client_process)

    ###### Create kill button ######
    utils.create_killer(start_time=start_time, processes=processes, logs=logs_to_zip)

    for process in processes:
        process.join()


if __name__ == "__main__":
    Config.load_config()

    camera_server_args = Config.get("camera_server_arguments")
    emulator_args = Config.get("emulator_arguments")
    pi_arguments = Config.get("pi_arguments")
    vision_args = Config.get("vision_arguments")

    object_detected = multiprocessing.Value('b', False)

    main(camera_server_args, pi_arguments, emulator_args, vision_args, object_detected)
