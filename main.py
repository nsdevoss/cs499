import multiprocessing
from src.pi import emulator
from src.utils import utils
from datetime import datetime
from multiprocessing import Manager
from src.utils.config import Config
from src.vision.vision import Vision
from src.server.logger import server_logger
from src.utils.utils import execute_pi_client
from src.vision.object_detect import ObjectDetector
from src.WebServer.web_server import WebServerDisplay
from src.server.camera_server import StreamCameraServer
from src.server.app_server import AppCommunicationServer
from src.server.visualization_server import VisualizationServer
from src.server.visualization_client import VisualizationClient


processes = []


def start_camera_server(vision_queue, camera_server_args, emulator_enabled):
    host = camera_server_args.get("host")
    port = camera_server_args.get("port")
    socket_type = camera_server_args.get("socket_type")
    fps = camera_server_args.get("fps")

    local_server = StreamCameraServer(host=host, port=port, socket_type=socket_type, vision_queue=vision_queue, emulator_enabled=emulator_enabled,fps=fps)
    local_server.receive_video_stream()


def start_app_server(detected_dist_object_queue):
    app_server = AppCommunicationServer(detected_dist_object_queue)
    app_server.connect_to_app()

def start_webserver(display_queue, shared_data, shared_fps):
    web_server = WebServerDisplay(display_queue=display_queue, shared_data=shared_data, shared_fps=shared_fps)
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


def start_vision_process(vision_queue, display_queue, detected_dist_object_queue, info_queue, yolo_input_queue, vision_args, scale, shared_fps):
    vision = Vision(frame_queue=vision_queue, display_queue=display_queue, info_queue=info_queue, object_detect_queue=detected_dist_object_queue,yolo_input_queue=yolo_input_queue,vision_args=vision_args, scale=scale, shared_fps=shared_fps)
    vision.start()

def start_visualization_process(info_queue):
    visualization_server = VisualizationServer(info_queue)
    visualization_server.connect()

def start_visualization_client(server_ip, port):
    client = VisualizationClient(server_ip, port)
    client.run()

def start_yolo_process(yolo_args, input_queue, shared_data):
    yolo_detector = ObjectDetector(yolo_args, input_queue, shared_data)
    yolo_detector.start()

def main(camera_server_args, pi_args, emulator_args, vision_args):
    global processes, ip_addr
    shared_fps = multiprocessing.Value('i', 0)
    manager = Manager()
    shared_data = manager.dict()
    shared_data["frame"] = None
    shared_data["boxes"] = None
    shared_data["scores"] = None
    shared_data["labels"] = None
    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    logs_to_zip = ["server.log", "webserver.log"]
    ip_addr = utils.get_ip_address()
    print(ip_addr)

    display_queue = None  # Vision processing -> Webserver
    vision_queue = None  # Raw server frame -> Vision processing
    info_queue = None  # Vision processing -> 3D visualization
    detected_dist_object_queue = None  # Vision dist processing -> App Server

    detected_object_frame_queue = None  # Vision processing -> Object Detector
    yolo_input_frame = None  # Object_detector -> Webserver

    vision_queue = multiprocessing.Queue()
    display_queue = multiprocessing.Queue()
    detected_dist_object_queue = multiprocessing.Queue()
    if vision_args.get("3d_render_args").get("enabled"):
        info_queue = multiprocessing.Queue()
    if vision_args.get("yolo_arguments").get("enabled"):
        yolo_input_frame = multiprocessing.Queue()

        ###### Start Yolo Detector ######
        server_logger.get_logger().info(f"Starting object detector process")
        yolo_detector_process = multiprocessing.Process(target=start_yolo_process, args=(vision_args.get("yolo_arguments"), yolo_input_frame, shared_data))
        yolo_detector_process.start()
        server_logger.get_logger().info(f"Started yolo detector process with pid: {yolo_detector_process.pid}")
        processes.append(yolo_detector_process)

    ###### Start Visualization Server ######
    if vision_args.get("3d_render_args").get("enabled"):
        server_logger.get_logger().info(f"Starting visualization server on port: 9002")
        visualization_server_process = multiprocessing.Process(target=start_visualization_process, args=(info_queue,))
        visualization_server_process.start()
        server_logger.get_logger().info(f"Started visualization server process with pid: {visualization_server_process.pid}")
        processes.append(visualization_server_process)
        if vision_args.get("3d_render_args").get("run_locally"):
            server_logger.get_logger().info(f"Starting local visualization client")
            visualization_client_process = multiprocessing.Process(target=start_visualization_client, args=(ip_addr, 9002))
            visualization_client_process.start()
            server_logger.get_logger().info(f"Started local visualization client with pid: {visualization_client_process.pid}")
            processes.append(visualization_client_process)


    ###### Web Server start process ######
    webserver_process = multiprocessing.Process(target=start_webserver, args=(display_queue,shared_data, shared_fps))
    webserver_process.start()
    processes.append(webserver_process)

    ###### Vision start process ######
    vision_process = multiprocessing.Process(target=start_vision_process, args=(vision_queue, display_queue, detected_dist_object_queue, info_queue, yolo_input_frame ,vision_args, camera_server_args.get("scale"), shared_fps))
    vision_process.start()
    server_logger.get_logger().info(f"Started vision process with pid: {vision_process.pid}")
    processes.append(vision_process)

    ###### Camera Server start process ######
    server_logger.get_logger().info(f"Starting server on port: {camera_server_args.get('port')}")
    camera_server_process = multiprocessing.Process(target=start_camera_server,args=(vision_queue, camera_server_args, emulator_args.get("enabled")), name=f"Server Process: {camera_server_args.get('port')}")
    camera_server_process.start()
    server_logger.get_logger().info(f"Started server process with pid: {camera_server_process.pid}")
    processes.append(camera_server_process)

    ###### App Server start process ######
    server_logger.get_logger().info(f"Starting app server on port: 9001")
    app_server_process = multiprocessing.Process(target=start_app_server, args=(detected_dist_object_queue,))
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

    main(camera_server_args, pi_arguments, emulator_args, vision_args)
