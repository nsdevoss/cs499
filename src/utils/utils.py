import subprocess
from multiprocessing.queues import Queue

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
