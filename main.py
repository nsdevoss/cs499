import argparse
import multiprocessing
from src.server import server
from src.pi import emulator
from src.utils import utils
from src.vision.stitching import frame_stitcher


def start_server(port, frame_queue, display):
    local_server = server.StreamCameraServer(port=port, frame_queue=frame_queue, display=display)
    local_server.receive_video_stream()


def start_emulator(ip_addr, video, port=9000):
    client = emulator.Emulator(ip_addr, video, port)
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
    ip_addr = utils.get_ip_address()
    print(f"IP Address: {ip_addr}")

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
    server_processes = []
    stitcher_process = None
    server_ports = [9000, 9001]

    # Start the servers on their own processes
    for port in server_ports:
        process = multiprocessing.Process(target=start_server, args=(port, frame_queue, display))
        process.start()
        server_processes.append(process)

    # Example on how we can pass the frame queue into a function for extraction
    if stitch:
        stitcher_process = multiprocessing.Process(target=frame_stitcher, args=(frame_queue,))
        stitcher_process.start()

    print("Server started...")

    # Start the emulators on their own processes
    if use_emulator:
        print("Running Emulated Client...")
        emulator_processes = []
        name_index = 0
        for port in server_ports:
            if len(video_names) == 1:
                emulator_process = multiprocessing.Process(target=start_emulator, args=(ip_addr, video_names[0], port))
                emulator_process.start()
                emulator_processes.append(emulator_process)
            else:
                emulator_process = multiprocessing.Process(target=start_emulator, args=(ip_addr, video_names[name_index], port))
                emulator_process.start()
                emulator_processes.append(emulator_process)
                name_index += 1
        for process in emulator_processes:
            process.join()

    for process in server_processes:
        process.join()

    if stitch:
        stitcher_process.join()


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
    main(args.emulator, args.stitch, args.video, args.display)  # We actually call the function with the passed in command line arguments
