import subprocess
import tkinter as tk
import gc
import os
import zipfile
import socket

MAX_QUEUE_SIZE = 10


def get_ip_address():
    try:
        # This is for Mac only, idk what it is on Windows
        result = subprocess.run(
            "ifconfig | grep 'inet ' | grep -v 127.0.0.1",
            shell=True, capture_output=True, text=True
        )

        lines = result.stdout.strip().split("\n")
        if lines:
            print(f"Found IP Address: {lines[0].split()[1]}")
            return lines[0].split()[1]

    except Exception as e:
        print(f"Error retrieving IP: {e}")
    print("No IP Address Found")
    return "Unknown"

# Windows get_ip_address() method
def windows_get_ip_address():
    try:
        host_name = socket.gethostname()
        windows_ip = socket.gethostbyname(host_name)
        print("Your Windows Computer name is: ")
        print("Your Windows Computer IP Address is:" + windows_ip)
        if windows_ip:
            print(f"Found IP Address: " + windows_ip)
            return windows_ip
    except Exception as e:
        print(f"Error Retrieving Windows IP: {e}")
    print("No IP Address Found")
    return "Unknown"


# Zips all the current logs, we get the logs from main.py in logs_to_zip
def zip_logs(start_time, logs: list):
    if not logs:
        return

    if not os.path.isdir("logs"):
        os.makedirs("logs")

    zip_filename = os.path.join("logs", f"{start_time}.zip")
    try:
        with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
            for log_file in logs:
                log_path = os.path.join("logs", log_file)
                if os.path.exists(log_path):
                    base_name = os.path.splitext(log_file)[0]

                    zipf.write(log_path, os.path.basename(log_file))
                    os.rename(log_path, f"ogs/{base_name}_latest.log")
        print(f"Zipped logs: {logs}")

    except Exception as e:
        print(f"Error zipping: {e}")


# Kills all the processes in main loop
def shutdown(start_time, processes, logs: list, root=None):
    print("Killing all processes...")
    zip_logs(start_time, logs)
    for process in processes:
        obj = getattr(process, 'obj', None)
        if obj and hasattr(obj, 'shutdown'):
            obj.shutdown()
            print(f"Shutdown called for process {process.pid}")

        pid = process.pid
        print(f"Killed process: {pid}")
        process.terminate()
    print("Zipped all current logs")

    if root:
        print("Closing Tkinter window...")
        root.destroy()


# Makes a little Tkinter widget to kill everything
def create_killer(start_time, processes, logs: list):
    root = tk.Tk()
    root.title("Kill Button")

    kill_button = tk.Button(root, text="Kill Everything", command=lambda: shutdown(start_time, processes, logs=logs, root=root), bg="red",
                            fg="white")
    kill_button.pack(pady=20, padx=20)

    root.mainloop()


def split_frame(frame):
    print(frame.shape)
    height, width = frame.shape
    half_width = width // 2
    left_frame = frame[:, :half_width]
    right_frame = frame[:, half_width:]
    return left_frame, right_frame


################ Experimental, you can try to make it better ################

def manage_queue(frame_queue):
    """
    This function continuously reads frames from the shared queue
    and removes old frames when the queue reaches MAX_QUEUE_SIZE.
    """
    processed_frames = {}

    while True:
        port, frame = frame_queue.get()

        processed_frames[port] = frame

        if len(processed_frames) > MAX_QUEUE_SIZE:
            oldest_port = list(processed_frames.keys())[0]
            del processed_frames[oldest_port]
            gc.collect()
