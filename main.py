import argparse
import multiprocessing
from src.server import server
from src.pi import emulator
from src.utils import utils
from src.vision.stitching import frame_stitcher


def start_server(port, frame_queue):
    local_server = server.StreamCameraServer(port=port, frame_queue=frame_queue)
    local_server.receive_video_stream()


def start_emulator(ip_addr, video, port=9000):
    client = emulator.Emulator(ip_addr, video, port)
    client.send_video()


def main(use_emulator: bool, stitch: bool, video_names: list):
    ip_addr = utils.get_ip_address()
    print(f"IP Address: {ip_addr}")

    frame_queue = multiprocessing.Queue()
    server_processes = []
    stitcher_process = None
    server_ports = [9000, 9001]

    for port in server_ports:
        process = multiprocessing.Process(target=start_server, args=(port,frame_queue))
        process.start()
        server_processes.append(process)

    if stitch:
        stitcher_process = multiprocessing.Process(target=frame_stitcher, args=(frame_queue,))
        stitcher_process.start()

    print("Server started...")

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
    parser = argparse.ArgumentParser(description="Run Emulated Client.")
    parser.add_argument(
        "-e", "--emulator", action="store_true",
        help="Run the emulated client instead of the Raspberry Pi Client if you have one connected."
    )
    parser.add_argument(
        "-s", "--stitch", action="store_true",
        help="Will run the frame stitcher"
    )
    parser.add_argument(
        "-v", "--video", nargs="+", type=str, default="zoom_out",
        choices=["crystal", "cube", "emu", "fade_in", "heart", "rotate", "slide_down", "slide_right", "slide_up", "waltzer", "zoom_in", "zoom_out", "left", "right", "left_emu", "right_emu"],
        help="Choose what video will be played on the emulator"
    )
    args = parser.parse_args()
    main(args.emulator, args.stitch, args.video)
