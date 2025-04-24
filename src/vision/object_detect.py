import os
import cv2
import onnxruntime
import numpy as np
from src.server.logger import server_logger
import src.LocalCommon as lc


class ObjectDetector:
    def __init__(self, yolo_args, input_queue, shared_data):
        self.model_path = os.path.join(lc.MODEL_DIR, yolo_args.get("model_file"))
        self.frame_skip = yolo_args.get("frame_skip")
        self.frame_count = 0
        self.conf_threshold = yolo_args.get("conf_threshold")
        self.iou_threshold = yolo_args.get("iou_threshold")
        self.input_size = yolo_args.get("input_size")
        self.class_names = yolo_args.get("class_names")

        self.input_queue = input_queue
        self.shared_data = shared_data

        self.session = None
        self.input_name = None

        self.logger = server_logger.get_logger()

    def start(self):
        try:
            self.session = onnxruntime.InferenceSession(self.model_path)
            self.input_name = self.session.get_inputs()[0].name
            self.logger.info(f"Initialized detector with model: {self.model_path}")
        except Exception as e:
            self.logger.error(f"Error initializing model: {e}")
            return

        self.logger.info("Starting object detection loop...")
        while True:
            try:
                self.process_queue()
            except Exception as e:
                self.logger.error(f"Error during detection loop: {e}")

    def process_queue(self):
        frame = get_latest_frame(self.input_queue)
        if frame is not None:
            self.frame_count += 1
            if self.frame_count <= self.frame_skip:
                return
            self.frame_count = 0

            try:
                detections = self.detect_objects(frame)
                self.update_data(detections)
            except Exception as e:
                self.logger.error(f"Error during detection loop: {e}")

    def update_data(self, detections):
        try:
            self.shared_data.update(detections)
        except AttributeError:
            for key, value in detections.items():
                if key in self.shared_data:
                    self.shared_data[key] = value

    def detect_objects(self, frame):
        input_tensor, orig_h, orig_w, ratio, pad = self.preprocess(frame)

        outputs = self.session.run(None, {self.input_name: input_tensor})
        boxes, scores, class_ids = self.process_output(outputs[0], orig_h, orig_w, ratio, pad)

        result_frame = self.draw_boxes(frame.copy(), boxes, scores, class_ids)

        return {
            'frame': result_frame,
            'boxes': boxes.tolist() if isinstance(boxes, np.ndarray) else boxes,
            'scores': scores.tolist() if isinstance(scores, np.ndarray) else scores,
            'class_ids': class_ids.tolist() if isinstance(class_ids, np.ndarray) else class_ids,
            'class_names': [self.class_names[int(i)] for i in class_ids]
        }

    def preprocess(self, img):
        orig_h, orig_w = img.shape[:2]

        ratio = min(self.input_size[0] / orig_h, self.input_size[1] / orig_w)
        new_w, new_h = int(round(orig_w * ratio)), int(round(orig_h * ratio))

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        dw, dh = self.input_size[1] - new_w, self.input_size[0] - new_h
        dw, dh = dw / 2, dh / 2

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        padded = padded.astype(np.float32) / 255.0
        padded = padded.transpose(2, 0, 1)
        padded = np.expand_dims(padded, 0)

        return padded, orig_h, orig_w, ratio, (dw, dh)

    def process_output(self, output, orig_h, orig_w, ratio, pad):
        if output.shape[1] <= 84 and output.shape[2] >= 1000:
            output = np.transpose(output, (0, 2, 1))

        predictions = output[0]

        boxes = predictions[:, :4]

        boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]  # x2
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]  # y2

        scores = predictions[:, 4:]
        class_scores = np.max(scores, axis=1)
        class_ids = np.argmax(scores, axis=1)

        mask = class_scores >= self.conf_threshold
        boxes = boxes[mask]
        scores = class_scores[mask]
        class_ids = class_ids[mask]

        boxes[:, [0, 2]] -= pad[0]
        boxes[:, [1, 3]] -= pad[1]
        boxes[:, :4] /= ratio

        boxes[:, 0].clip(0, orig_w, out=boxes[:, 0])
        boxes[:, 1].clip(0, orig_h, out=boxes[:, 1])
        boxes[:, 2].clip(0, orig_w, out=boxes[:, 2])
        boxes[:, 3].clip(0, orig_h, out=boxes[:, 3])

        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.conf_threshold, self.iou_threshold)
            return boxes[indices], scores[indices], class_ids[indices]

        return [], [], []

    def draw_boxes(self, img, boxes, scores, class_ids):
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = [int(i) for i in box]

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"{self.class_names[class_id]}: {score:.2f}"
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - label_h - baseline), (x1 + label_w, y1), (0, 255, 0), -1)
            cv2.putText(img, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return img

def get_latest_frame(queue):
    latest_frame = None
    while not queue.empty():
        try:
            latest_frame = queue.get_nowait()
        except Exception:
            break
    return latest_frame
