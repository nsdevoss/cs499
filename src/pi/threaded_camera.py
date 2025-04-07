import os.path
import threading
import cv2
import time

############################ EXPERIMENTAL ##################################


class ThreadedCamera:
    def __init__(self, src=0, scale=1.0):
        self.src = src
        self.scale = scale

        if isinstance(src, str) and os.path.exists(src):
            self.cap = cv2.VideoCapture(src)
        else:
            try:
                self.cap = cv2.VideoCapture(src, cv2.CAP_ANY)
            except:
                self.cap = cv2.VideoCapture(src)

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        self.ret, self.frame = self.cap.read()
        if self.ret and self.scale != 1.0:
            self.frame = cv2.resize(self.frame, (int(self.frame.shape[1] * self.scale),
                                                 int(self.frame.shape[0] * self.scale)))

        self.lock = threading.Lock()
        self.stopped = False

        self.frame_count = 0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if isinstance(src, str) else 0

        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while not self.stopped:
            if not self.cap.isOpened():
                self.stopped = True
                break

            for _ in range(5):
                self.cap.grab()

            ret, frame = self.cap.read()

            # This loops the video like in the emulator
            if not ret and isinstance(self.src, str) and self.total_frames > 0:
                self.frame_count = 0
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()

            if ret:
                if self.scale != 1.0:
                    frame = cv2.resize(frame, (int(frame.shape[1] * self.scale),
                                               int(frame.shape[0] * self.scale)))

                with self.lock:
                    self.ret = ret
                    self.frame = frame
                    self.frame_count += 1

            time.sleep(0.001)

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.ret else None

    def release(self):
        self.stopped = True
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()
