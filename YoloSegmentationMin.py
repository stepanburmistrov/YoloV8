import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')

colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (192, 192, 192), (128, 128, 128), (128, 0, 0), (128, 128, 0),
    (0, 128, 0), (128, 0, 128), (0, 128, 128), (0, 0, 128), (72, 61, 139),
    (47, 79, 79), (47, 79, 47), (0, 206, 209), (148, 0, 211), (255, 20, 147)
]

def process_frame(frame):
    results = model(frame)[0]
    image = results.orig_img
    h, w = image.shape[:2]
    classes = results.boxes.cls.cpu().numpy()
    masks = results.masks.data.cpu().numpy()

    for i, mask in enumerate(masks):
        color = colors[int(classes[i]) % len(colors)]
        resized_mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        color_mask = np.zeros_like(image)
        color_mask[resized_mask > 0] = color
        image = cv2.addWeighted(image, 1.0, color_mask, 0.5, 0)
    return image


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Ошибка: Не удалось открыть камеру")
    exit

while True:
    ret, frame = cap.read()
    if not ret:
        print("Ошибка: Не удалось получить кадр из камеры")
        break

    processed_frame = process_frame(frame)
    cv2.imshow('Segmented Frame', processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
