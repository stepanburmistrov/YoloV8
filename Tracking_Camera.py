from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO


model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

print("Video file opened successfully.")

track_history = defaultdict(lambda: [])

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        print("End of video or cannot read the frame.")
        break

    results = model.track(frame, persist=True)

    if results[0].boxes is not None and results[0].boxes.id is not None:
        
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        annotated_frame = results[0].plot()


        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))
            if len(track) > 30: 
                track.pop(0)

            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

        cv2.imshow("YOLOv8 Tracking", annotated_frame)
    else:

        cv2.imshow("YOLOv8 Tracking", frame)

    if cv2.waitKey(1)==27:
        break
cap.release()
cv2.destroyAllWindows()
