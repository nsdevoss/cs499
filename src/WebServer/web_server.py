import time
import cv2
from flask import Flask, render_template, Response
import threading

app = Flask(__name__)

latest_frame = None
lock = threading.Lock()

def frame_producer(display_queue):
    global latest_frame
    while True:
        disp, frame, points_3d, valid_mask = display_queue.get()
        ret, buffer = cv2.imencode('.jpg', disp)
        with lock:
            latest_frame = buffer.tobytes()
        time.sleep(0.01)

class WebServerDisplay:
    def __init__(self, display_queue, host="0.0.0.0", port=8080):
        self.display_queue = display_queue
        self.host = host
        self.port = port
        self.app = Flask(__name__)

        self.app.add_url_rule('/', 'index', self.index)
        self.app.add_url_rule('/video', 'video', self.video)

    def index(self):
        return render_template('index.html')

    def video(self):
        return Response(self.stream_latest_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def stream_latest_frame(self):
        while True:
            with lock:
                if latest_frame is not None:
                    frame = latest_frame
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    def run(self):
        threading.Thread(target=frame_producer, args=(self.display_queue,), daemon=True).start()
        self.app.run(host=self.host, port=self.port, threaded=True)

