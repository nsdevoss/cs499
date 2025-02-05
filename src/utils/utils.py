import subprocess
import os
import moviepy.video.io.ImageSequenceClip


def create_video(image_dir: str, video_name: str, fps=1):
    images = [os.path.join(image_dir,img) for img in os.listdir(image_dir)]
    images.sort()
    for img in images:
        print(f"{img}\n")
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(images, fps=fps)
    clip.write_videofile(video_name)


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
