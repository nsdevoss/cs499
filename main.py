import argparse
import multiprocessing
from src.server import server
from src.pi import emulator
from src.utils import utils


def start_server(port):
    local_server = server.StreamCameraServer(port=port)
    local_server.receive_video_stream()


def start_emulator(ip_addr, video, port=9000):
    client = emulator.Emulator(ip_addr, video, port)
    client.send_video()


def main(use_emulator: bool, single_server: bool, video_name: str):
    ip_addr = utils.get_ip_address()
    print(f"IP Address: {ip_addr}")

    server_processes = []
    server_ports = [9000] if single_server else [9000, 9001]

    for port in server_ports:
        process = multiprocessing.Process(target=start_server, args=(port,))
        process.start()
        server_processes.append(process)

    print("Server started...")

    if use_emulator:
        print("Running Emulated Client...")
        emulator_processes = []
        for port in server_ports:
            emulator_process = multiprocessing.Process(target=start_emulator, args=(ip_addr, video_name, port))
            emulator_process.start()
            emulator_processes.append(emulator_process)
        for process in emulator_processes:
            process.join()

    for process in server_processes:
        process.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Emulated Client.")
    parser.add_argument(
        "-e", "--emulator", action="store_true",
        help="Run the emulated client instead of the Raspberry Pi Client if you have one connected."
    )
    parser.add_argument(
        "-s", "--single", action="store_true",
        help="Run only one server on port 9000. If not set, two servers will run on 9000 and 9001"
    )
    parser.add_argument(
        "-v", "--video", type=str, default="zoom_out",
        choices=["crystal", "cube", "emu", "fade_in", "heart", "rotate", "slide_down", "slide_right", "slide_up", "waltzer", "zoom_in", "zoom_out"],
        help="Choose what video will be played on the emulator"
    )
    args = parser.parse_args()
    main(args.emulator, args.single, args.video)
