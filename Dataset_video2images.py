import cv2
import os
import time

# Путь к видеофайлу
video_path = '000.mp4'
# Папка для сохранения изображений
output_folder = 'output_images'
# Интервал между кадрами (каждый n-й кадр будет сохранен)
frame_interval = 1  # Можно изменить на 2, 5 и т.д.

os.makedirs(output_folder, exist_ok=True)

# Открытие видеофайла
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Ошибка открытия видеофайла: {video_path}")
    exit()

frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        # Получение текущего времени в виде временной метки
        timestamp = int(time.time() * 1000)  # Используем миллисекунды для большей точности
        output_path = os.path.join(output_folder, f'{timestamp}_frame_{saved_count:05d}.jpg')
        cv2.imwrite(output_path, frame)
        print(f"Сохранено: {output_path}")
        saved_count += 1

    frame_count += 1

cap.release()
print("Разделение видео на фотографии завершено.")
