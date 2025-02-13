import argparse
import multiprocessing
from src.server import server
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


"""
This is the big dog function where everything goes down.

How it all goes down:
- Each server is ran on a different port AND on a different process.
- We want to do this rather than a different thread because we need each server to process things individually (and because its easier).
- Also each Client/Emulator is run on their own processes which makes more sense because we have two camera streams.
- This function takes command line arguments to run.

Params:
***** All of the parameters here are from the command line arguments *****
@use_emulator: This tells us if we are going to run the emulator or not, default false
@stitch: If we want to run the stitch function on the frames, default false
@video_names -> list[str]: Holds the names of the videos the emulator will play, default ["zoom_in"]
"""
def main(use_emulator: bool, stitch: bool, video_names: list, display: bool):
    global processes
    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Set up loggers and stuff
    server_logger = Logger(name="ServerLogger", log_file="server.log")
    client_logger = Logger(name="EmulatorLogger", log_file="emu_client.log")
    logs_to_zip = ["server.log"]
    ip_addr = utils.get_ip_address()
    vision_arguments = {"stitching": stitch}
    for argument in vision_arguments:
        server_logger.get_logger().info(f"Vision action {argument}: {vision_arguments[argument]}")
    server_logger.get_logger().info(f"Got IP Address: {ip_addr}")

    """
    frame_queue:
    This is the thing we are using to extract each frame from the servers and be able to use them on different processes.
    The queue is seen by both servers even though they run on different processes, they each individually put their frames in this queue.
    Here is an example of what the queue looks like when the servers send their frames:
            (9000, frame1),
            (9001, frame1),
            (9000, frame2),
            (9001, frame2),
            (9001, frame3),
            (9000, frame3)
    Sometimes one server might send than the other for a few frames but it is not important.
    
    You can pass this queue into any function and extract the frames using their ports as indicators for where they came from.
    ***** See stitching.frame_stitcher to see a good example on how to extract the frames from the queue. *****
    """
    frame_queue = multiprocessing.Queue()

    server_ports = [9000, 9001]
    vision = Vision(frame_queue=frame_queue, action_arguments=vision_arguments,
                    server_logger=server_logger)  # This is the vision object, responsible for the calculations

    # HUGE memory optimization BUT it breaks stitching, so it will break anything reading from the frame_queue
    # queue_manager_process = multiprocessing.Process(target=manage_queue, args=(frame_queue,))
    # queue_manager_process.start()

    for port in server_ports:
        server_logger.get_logger().info("Starting server on port: {port}")
        process = multiprocessing.Process(target=start_server, args=(port, frame_queue, display, server_logger),
                                          name=f"Server Process: {port}")
        process.start()
        processes.append(process)

    vision_process = multiprocessing.Process(target=vision.start)
    vision_process.start()
    processes.append(vision_process)

    if use_emulator:
        logs_to_zip.append("emu_client.log")
        server_logger.get_logger().info("Running Emulated Client...")
        for idx, port in enumerate(server_ports):
            video = video_names[min(idx, len(video_names) - 1)]
            client_logger.get_logger().info(f"Starting emulator on: {port} with video: {video}")
            emulator_process = multiprocessing.Process(target=start_emulator,
                                                       args=(ip_addr, video, port, client_logger))
            emulator_process.start()
            processes.append(emulator_process)

    utils.create_killer(start_time=start_time, logs=logs_to_zip)  # Kill button that KILLS every process

    for process in processes:
        process.join()
    server_logger.get_logger().info(f"Joined vision process: {vision_process.pid}")
    # queue_manager_process.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--emulator", action="store_true",
        help="Run the emulated client instead of the Raspberry Pi Client if you have one connected."
    )
    parser.add_argument(
        "-d", "--display", action="store_false",
        help="This will disable display from the two cameras"
    )
    parser.add_argument(
        "-s", "--stitch", action="store_true",
        help="Will run the frame stitcher"
    )
    parser.add_argument(
        "-v", "--video", nargs="+", type=str, default=["zoom_out"],
        choices=["crystal", "cube", "emu", "fade_in", "heart", "rotate", "slide_down", "slide_right", "slide_up", "waltzer", "zoom_in", "zoom_out", "left", "right", "left_emu", "right_emu"],
        help="Choose what video will be played on the emulator"
    )
    args = parser.parse_args()
    main(args.emulator, args.stitch, args.video, args.display)  # We actually call the function with the passed in cli
