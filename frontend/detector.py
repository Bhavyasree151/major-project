# detector.py
import cv2
import numpy as np
from depth_estimator import estimate_depth
import torch
from ultralytics import YOLO

# Load models only once
yolo5 = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
yolo8 = YOLO('yolov8n.pt')

def detect_and_estimate(image, model_name="YOLOv8"):
    depth_map = estimate_depth(image)

    if model_name == "YOLOv5":
        results = yolo5(image)
        detections = results.xyxy[0]
        names = yolo5.names

        for det in detections:
            x1, y1, x2, y2 = map(int, det[:4].tolist())
            conf = det[4].item()
            cls = int(det[5].item())

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            distance = depth_map[cy, cx] if 0 <= cx < depth_map.shape[1] and 0 <= cy < depth_map.shape[0] else -1
            label = f"{names[cls]} {conf:.2f} | {distance:.2f} m"

            cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    else:
        results = yolo8(image)[0]
        names = yolo8.names

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            distance = depth_map[cy, cx] if 0 <= cx < depth_map.shape[1] and 0 <= cy < depth_map.shape[0] else -1
            label = f"{names[cls]} {conf:.2f} | {distance:.2f} cm"

            cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    return image
