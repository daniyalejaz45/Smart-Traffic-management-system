from ultralytics import YOLO
import cv2
from roboflow import Roboflow


class VehicleDetector:
    def __init__(self, yolov8_path='yolov8n.pt'):
        self.yolo_model = YOLO(yolov8_path)

        # Initialize Roboflow model
        rf = Roboflow(api_key="TjSPeKVOPpdxLx9PULYS")
        project = rf.workspace().project("-bjtxt")
        self.rf_model = project.version("1").model

    def detect(self, frame):
        # --- YOLOv8 Detection ---
        results = self.yolo_model(frame, verbose=False)[0]
        count = 0
        emergency_detected = False

        for box in results.boxes:
            cls = int(box.cls)
            label = self.yolo_model.names[cls]
            if label in ['car', 'truck', 'bus', 'motorbike']:
                count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # --- Roboflow Ambulance Detection ---
        height, width = frame.shape[:2]
        result = self.rf_model.predict(frame, confidence=40, overlap=30).json()

        for prediction in result["predictions"]:
            label = prediction["class"]
            if label.lower() == "ambulance":
                emergency_detected = True
                x, y, w, h = prediction["x"], prediction["y"], prediction["width"], prediction["height"]
                x1, y1 = int(x - w / 2), int(y - h / 2)
                x2, y2 = int(x + w / 2), int(y + h / 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "Ambulance", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return frame, count, emergency_detected
