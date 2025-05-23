import subprocess
import os
import zipfile
import socket
import platform
import paramiko
import tkinter as tk
from src.server.logger import server_logger, client_logger


def get_ip_address():
    system = platform.system()
    if system == "Darwin":
        server_logger.get_logger().info(f"Program will be run on {platform.platform(terse=True)}")
        try:
            result = subprocess.run(
                "ifconfig | grep 'inet ' | grep -v 127.0.0.1",
                shell=True, capture_output=True, text=True
            )

            lines = result.stdout.strip().split("\n")
            if lines:
                server_logger.get_logger().info(f"LAN WIFI IPv4 Address: {lines[0].split()[1]}")
                return lines[0].split()[1]

        except Exception as e:
            server_logger.get_logger().error(f"Error retrieving IP: {e}")
        server_logger.get_logger().error("No IP Address Found")
        return "Unknown"
    elif system == "Windows":
        server_logger.get_logger().info(f"Program will be run on {platform.platform(terse=True)}")
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
    else:
        print("Platform is unknown")
        server_logger.get_logger().error("Unsupported platform, returning 'Unknown'")
        return "Unknown"


# Zips all the current logs, we get the logs from main.py in logs_to_zip, still some bugs with it
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
                    os.rename(log_path, f"logs/{base_name}_latest.log")
                    print(log_path)
        server_logger.get_logger().info(f"Zipped logs: {logs}")

    except Exception as e:
        server_logger.get_logger().error(f"Error zipping: {e}")


# Kills all the processes in main loop
def shutdown(start_time, processes, logs: list, root=None):
    server_logger.get_logger().info("Starting shutdown, killing all processes...")
    zip_logs(start_time, logs)
    for process in processes:
        obj = getattr(process, 'obj', None)
        if obj and hasattr(obj, 'shutdown'):
            obj.shutdown()

        pid = process.pid
        server_logger.get_logger().info(f"Killed process: {pid}")
        process.terminate()

    if root:
        server_logger.get_logger().info("Closing Tkinter window...")
        root.destroy()
        server_logger.get_logger().info("Successfully terminated program!")


# Makes a little Tkinter widget to kill everything
def create_killer(start_time, processes, logs: list):
    root = tk.Tk()
    root.title("Kill Button")

    kill_button = tk.Button(root, text="Kill Everything", command=lambda: shutdown(start_time, processes, logs=logs, root=root), bg="red",
                            fg="white")
    kill_button.pack(pady=20, padx=20)

    root.mainloop()


def split_frame(frame):
    if len(frame.shape) == 2:
        height, width = frame.shape
        half_width = width // 2
        left_frame = frame[:, :half_width]
        right_frame = frame[:, half_width:]
    else:
        height, width, channels = frame.shape
        half_width = width // 2
        left_frame = frame[:, :half_width, :]
        right_frame = frame[:, half_width:, :]

    return left_frame, right_frame


#############################  SSH For the Raspberry PI #####################################

def execute_pi_client(pi_arguments, server_ip, server_port, socket_type, scale):
    hostname = pi_arguments.get("hostname")
    port = pi_arguments.get("port")
    username = pi_arguments.get("username")
    password = pi_arguments.get("password")
    fps = pi_arguments.get("fps")

    encode_quality = pi_arguments.get("encode_quality")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(hostname, port=port, username=username, password=password)

        commands = f"""
        cd Desktop
        pwd
        source .venv/bin/activate
        python camera_client.py --server_ip {server_ip} --server_port {server_port} --socket_type {socket_type} --encode_quality {encode_quality} --scale {scale} --fps {fps}
        """
        stdin, stdout, stderr = client.exec_command(commands)
        output = stdout.read().decode()
        errors = stderr.read().decode()

        print("Output:", output)
        if errors:
            print(f"Errors: {errors}")
    finally:
        client.close()


if __name__ == "__main__":
    execute_pi_client("192.168.1.159", 22, "uab.edu", "pi")
