import cv2
import numpy as np
import onnxruntime as ort
from Alarm_Player import AlarmPlayer

model_path = 'best.onnx'
alarm_path = 'fire_alarm.mp3'

# --- Classes included in the model ---
CLASSES = ['Fire', 'Smoke']  # 0: Fire, 1: Smoke


class YOLOFireDetectorONNX:
    def __init__(self, model_path, conf_threshold = 0.40, nms_threshold = 0.40, input_size = 640):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size

        # Initialize ONNX Runtime session for optimal CPU performance
        self.session = ort.InferenceSession(model_path, providers = ['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(self, frame):
        self.orig_h, self.orig_w = frame.shape[:2]

        # 1. Resize to model size (640x640)
        resized = cv2.resize(frame, (self.input_size, self.input_size))

        # 2. BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # 3. Normalize [0, 255] -> [0.0, 1.0] and Transpose (H, W, C) -> (C, H, W)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = tensor.transpose(2, 0, 1)

        # 4. Add Batch dimension -> Shape: (1, 3, 640, 640)
        blob = np.expand_dims(tensor, axis=0)
        return blob

    def postprocess(self, outputs):
        # Output shape is typically (1, 6, 8400) -> Transpose to (8400, 6)
        predictions = np.squeeze(outputs[0])
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T

        boxes = []
        confidences = []
        class_ids = []

        scale_x = self.orig_w / self.input_size
        scale_y = self.orig_h / self.input_size

        for row in predictions:
            scores = row[4:]
            class_id = np.argmax(scores)
            max_score = scores[class_id]

            if max_score >= self.conf_threshold:
                cx, cy, w, h = row[0:4]

                # Convert coordinates from center format (cx, cy, w, h) to top-left format (x, y, w, h)
                left = int((cx - 0.5 * w) * scale_x)
                top = int((cy - 0.5 * h) * scale_y)
                width = int(w * scale_x)
                height = int(h * scale_y)

                boxes.append([left, top, width, height])
                confidences.append(float(max_score))
                class_ids.append(class_id)

        # Apply Non-Maximum Suppression (NMS) to eliminate duplicate overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)

        final_boxes, final_confs, final_classes = [], [], []
        # Corrected usage of 'indices' variable instead of 'np.indices'
        if len(indices) > 0:
            for i in np.array(indices).flatten():
                final_boxes.append(boxes[i])
                final_confs.append(confidences[i])
                final_classes.append(class_ids[i])

        return final_boxes, final_confs, final_classes

    def detect(self, frame):
        blob = self.preprocess(frame)
        outputs = self.session.run([self.output_name], {self.input_name: blob})
        return self.postprocess(outputs)


def main():
    # 1. Initialize objects
    detector = YOLOFireDetectorONNX(model_path, conf_threshold = 0.40, nms_threshold = 0.40, input_size = 640)
    alarm_player = AlarmPlayer(alarm_path)
    webcam = cv2.VideoCapture(0)

    if not webcam.isOpened():
        print("Error: Unable to open webcam stream.")
        return

    print("Starting live alarm system via YOLO ONNX...")

    # 2. Live capture and processing loop
    while True:
        ret, frame = webcam.read()
        if not ret:
            print("Error: Unable to capture frame from webcam.")
            break

        boxes, confidences, class_ids = detector.detect(frame)

        # 3. Trigger audio alert when any detection is found
        if len(boxes) > 0:
            alarm_player.play()

        # 4. Draw bounding boxes and text labels on frame
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            x, y, w, h = box
            label_name = (CLASSES[cls_id] if cls_id < len(CLASSES) else "Fire/Smoke")

            # Red color for Fire (Class 0), Orange color for Smoke (Class 1)
            color = (0, 0, 255) if cls_id == 0 else (0, 165, 255)

            cv2.rectangle(
                frame, (x, y), (x + w, y + h), color, thickness = 2
            )
            cv2.putText(
                frame,
                f"{label_name} {conf * 100:.1f}%",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        # 5. Display frame and capture exit key
        cv2.imshow("YOLO ONNX Fire & Smoke Detection System", frame)

        # Press 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 6. Release resources and close windows
    webcam.release()
    cv2.destroyAllWindows()
    alarm_player.stop()


if __name__ == '__main__':
    main()