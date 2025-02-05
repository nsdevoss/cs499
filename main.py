import argparse
import multiprocessing
from src.server import server
from src.pi import emulator
from src.utils import utils


def start_server():
    local_server = server.StreamCameraServer()
    local_server.receive_video_stream()


def start_emulator(ip_addr, video, port=9000):
    client = emulator.Emulator(ip_addr, video, port)
    client.send_video()


def main(use_emulator: bool, video_name: str):
    ip_addr = utils.get_ip_address()
    print(ip_addr)
    server_port = 9000

    server_process = multiprocessing.Process(target=start_server)
    server_process.start()
    print("Server started...")
    if use_emulator:
        print("Running Emulated Client...")
        emulator_process = multiprocessing.Process(target=start_emulator, args=(ip_addr, video_name, server_port))
        emulator_process.start()
        emulator_process.join()

    server_process.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Emulated Client.")
    parser.add_argument(
        "-e", "--emulator", action="store_true",
        help="Run the emulated client instead of the Raspberry Pi Client if you have one connected."
    )
    parser.add_argument(
        "-v", "--video", type=str, default="zoom_out",
        choices=["crystal", "cube", "emu", "fade_in", "heart", "rotate", "slide_down", "slide_right", "slide_up", "waltzer", "zoom_in", "zoom_out"],
        help="Choose what video will be played on the emulator"
    )
    args = parser.parse_args()
    main(args.emulator, args.video)
