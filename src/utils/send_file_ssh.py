import paramiko
import os
import src.LocalCommon as lc

def send_file_to_pi(local_path, remote_path, pi_ip, pi_user, pi_password, port=22):
    if not os.path.isfile(local_path):
        print(f"Local file '{local_path}' not found.")
        return

    try:
        print(f"Connecting to {pi_ip}:{port} as {pi_user}")
        transport = paramiko.Transport((pi_ip, port))
        transport.connect(username=pi_user, password=pi_password)

        sftp = paramiko.SFTPClient.from_transport(transport)
        print(f"Transferring {local_path} to {remote_path}")
        sftp.put(local_path, remote_path)
        print("File transfer complete.")

        sftp.close()
        transport.close()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    send_file_to_pi(
        local_path=f"{lc.ROOT_DIR}/src/pi/pi_client.py",
        remote_path="/home/uab.edu/Desktop/camera_client.py",
        pi_ip="192.168.1.159",
        pi_user="uab.edu",
        pi_password="pi"
    )
