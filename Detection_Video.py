from ultralytics import YOLO
import cv2
import numpy as np

# Загрузка модели YOLOv8
model = YOLO('yolov8x.pt')

# Словарь цветов для различных классов
colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (192, 192, 192), (128, 128, 128), (128, 0, 0), (128, 128, 0),
    (0, 128, 0), (128, 0, 128), (0, 128, 128), (0, 0, 128), (72, 61, 139),
    (47, 79, 79), (47, 79, 47), (0, 206, 209), (148, 0, 211), (255, 20, 147)
]

# Открытие исходного видеофайла
input_video_path = 'input.mp4'
capture = cv2.VideoCapture(input_video_path)

# Чтение параметров видео
fps = int(capture.get(cv2.CAP_PROP_FPS))
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Настройка выходного файла
output_video_path = 'detect.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

while True:
    # Захват кадра
    ret, frame = capture.read()
    if not ret:
        break

    # Обработка кадра с помощью модели YOLO
    results = model(frame)[0]

    # Получение данных об объектах
    classes_names = results.names
    classes = results.boxes.cls.cpu().numpy()
    boxes = results.boxes.xyxy.cpu().numpy().astype(np.int32)
    # Рисование рамок и подписей на кадре
    for class_id, box, conf in zip(classes, boxes, results.boxes.conf):
        if conf>0.5:
            class_name = classes_names[int(class_id)]
            color = colors[int(class_id) % len(colors)]
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Запись обработанного кадра в выходной файл
    writer.write(frame)

# Освобождение ресурсов и закрытие окон
capture.release()
writer.release()
