import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')

def process_frame(frame):
    results = model(frame)[0]

    if results.probs is not None:
        top1_idx = results.probs.top1
        top1_conf = results.probs.top1conf.item() 
        class_name = results.names[top1_idx]
        label = f"{class_name}: {top1_conf:.2f}"
        cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    return frame

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera error")
    exit(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("read error")
        break

    processed_frame = process_frame(frame)
    cv2.imshow('YOLOv8 Classification', processed_frame)

    if cv2.waitKey(1)== ord('q'): 
        break

cap.release() 
cv2.destroyAllWindows()


