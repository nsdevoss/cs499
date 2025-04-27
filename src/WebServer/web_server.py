import time
import cv2
import threading
from src.server.logger import webserver_logger
from flask import Flask, render_template, Response

app = Flask(__name__)

latest_frame = None
lock = threading.Lock()

def frame_producer(display_queue):
    global latest_frame

    # Setup video writer
    frame_width, frame_height = 640, 480  # Make sure this matches the actual frame dimensions
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Or 'mp4v' for .mp4
    out = cv2.VideoWriter('output.avi', fourcc, fps, (frame_width, frame_height))

    while True:
        disp = display_queue.get()
        if disp is None:
            break  # Allows clean shutdown if needed

        # Write frame to video
        out.write(disp)

        # Prepare frame for web stream
        ret, buffer = cv2.imencode('.jpg', disp)
        with lock:
            latest_frame = buffer.tobytes()

        time.sleep(0.01)

    out.release()  # Clean up when done


class WebServerDisplay:
    def __init__(self, display_queue, shared_data, shared_fps, host="0.0.0.0", port=8080):
        self.display_queue = display_queue
        self.shared_data = shared_data
        self.shared_fps = shared_fps

        while True:
            frame = display_queue.get()
            if frame is not None:
                break
        self.frame_dimensions = str(f"{int(frame.shape[1])}x{int(frame.shape[0])}")
        self.logger = webserver_logger.get_logger()
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self.logger.info(f"Starting webserver on {self.host}:{self.port}")

        self.app.add_url_rule('/', 'index', self.index)
        self.app.add_url_rule('/video', 'video', self.video)
        self.logger.info(f"Successfully initialized webserver")

    def index(self):
        return render_template('index.html', dimensions=self.frame_dimensions)

    def video(self):
        return Response(self.stream_latest_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def stream_latest_frame(self):
        while True:
            with lock:
                if latest_frame is not None:
                    frame = latest_frame
                else:
                    self.logger.error(f"Latest frame could not be read")
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    def run(self):
        threading.Thread(target=frame_producer, args=(self.display_queue,), daemon=True).start()
        self.app.run(host=self.host, port=self.port, threaded=True)

