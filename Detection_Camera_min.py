from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO('yolov8n.pt')

color = (0, 255, 255)
capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()
    results = model(frame)[0]

    for class_id, box in zip(results.boxes.cls.cpu().numpy(),
                             results.boxes.xyxy.cpu().numpy().astype(np.int32)):
        class_name = results.names[int(class_id)]
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame,
                    class_name,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 2)

    cv2.imshow('YOLOv8 Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
