import time

import cv2
from flask import Flask,render_template,Response

app=Flask(__name__)

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
        return Response(self.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def generate_frames(self):
        while True:
            # if self.display_queue is not None and not self.display_queue.empty():
                disp, frame, points_3d, valid_mask = self.display_queue.get()
                ret, buffer = cv2.imencode('.jpg', disp)
                frame = buffer.tobytes()

                yield(b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            # else:
            #     time.sleep(0.01)

    def run(self):
        self.app.run(host=self.host, port=self.port, threaded=True)
