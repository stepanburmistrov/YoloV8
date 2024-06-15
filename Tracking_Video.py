from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO

# Загрузка модели YOLOv8
model = YOLO("yolov8x.pt")

# Открытие видео файла
video_path = "input.mp4"
cap = cv2.VideoCapture(video_path)

# Проверка успешного открытия видео
if not cap.isOpened():
    print(f"Ошибка открытия {video_path}")
    exit()


# Получение FPS видео
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Настройка VideoWriter для сохранения выходного видео
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('out.mp4', fourcc, fps, (width, height))

# Создание словаря для хранения истории треков объектов
track_history = defaultdict(lambda: [])

# Цикл для обработки каждого кадра видео
while cap.isOpened():
    # Считывание кадра из видео
    success, frame = cap.read()

    if not success:
        print("Конец видео")
        break

    # Применение YOLOv8 для отслеживания объектов на кадре, с сохранением треков между кадрами
    results = model.track(frame, persist=True)

    # Проверка на наличие объектов
    if results[0].boxes is not None and results[0].boxes.id is not None:
        # Получение координат боксов и идентификаторов треков
        boxes = results[0].boxes.xywh.cpu()  # xywh координаты боксов
        track_ids = results[0].boxes.id.int().cpu().tolist()  # идентификаторы треков

        # Визуализация результатов на кадре
        annotated_frame = results[0].plot()

        # Отрисовка треков
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box  # координаты центра и размеры бокса
            track = track_history[track_id]
            track.append((float(x), float(y)))  # добавление координат центра объекта в историю
            if len(track) > 30:  # ограничение длины истории до 30 кадров
                track.pop(0)

            # Рисование линий трека
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

        # Отображение аннотированного кадра
        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        out.write(annotated_frame)  # запись кадра в выходное видео
    else:
        # Если объекты не обнаружены, просто отображаем кадр
        cv2.imshow("YOLOv8 Tracking", frame)
        out.write(frame)  # запись кадра в выходное видео

    # Прерывание цикла при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Освобождение видеозахвата и закрытие всех окон OpenCV
cap.release()
out.release()  # закрытие выходного видеофайла
cv2.destroyAllWindows()
