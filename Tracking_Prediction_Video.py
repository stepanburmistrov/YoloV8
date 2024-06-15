from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO

# Загрузка модели YOLOv8
model = YOLO("yolov8n.pt")

# Путь к видео
video_path = "input.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error opening file {video_path}")
    exit()

print("start")

# Получение FPS и размеров видео
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Создание VideoWriter для записи выходного видео
output_path = "output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Кодек для записи
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

track_history = defaultdict(lambda: [])

def predict_position(track, future_time, fps):
    if len(track) < 2:
        return track[-1]

    N = min(len(track), 25)
    track = np.array(track[-N:])

    times = np.arange(-N + 1, 1)

    A = np.vstack([times, np.ones(len(times))]).T
    k_x, b_x = np.linalg.lstsq(A, track[:, 0], rcond=None)[0]
    k_y, b_y = np.linalg.lstsq(A, track[:, 1], rcond=None)[0]

    future_frames = future_time * fps
    future_x = k_x * future_frames + b_x
    future_y = k_y * future_frames + b_y

    return future_x, future_y

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        print("reading error")
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
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=3)

            future_time = 1.5  # секунд
            future_x, future_y = predict_position(track, future_time, fps)

            if len(track) > 1:
                last_x, last_y = track[-1]
                cv2.line(annotated_frame, (int(last_x), int(last_y)), (int(future_x), int(future_y)), (0, 255, 255), 2)

            cv2.circle(annotated_frame, (int(future_x), int(future_y)), 5, (0, 255, 0), -1)
            cv2.putText(annotated_frame, 'Predicted', (int(future_x), int(future_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Запись кадра в выходное видео
        out.write(annotated_frame)

        cv2.imshow("YOLOv8 Tracking", annotated_frame)
    else:
        cv2.imshow("YOLOv8 Tracking", frame)

        # Запись кадра в выходное видео
        out.write(frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
