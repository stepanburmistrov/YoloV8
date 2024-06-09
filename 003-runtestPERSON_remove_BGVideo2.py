import cv2
import numpy as np
from ultralytics import YOLO

# Загрузка модели YOLOv8
model = YOLO('yolov8x-seg.pt')

# Цвет для выделения объектов класса "person"
person_color = (0, 255, 0)  # Зеленый цвет


# Функция для обработки кадра
def process_frame(frame):
    image_orig = frame.copy()
    h_or, w_or = frame.shape[:2]
    image = cv2.resize(frame, (640, 640))
    results = model(image)[0]

    classes = results.boxes.cls.cpu().numpy()
    masks = results.masks.data.cpu().numpy()

    # Создаем зеленый фон
    green_background = np.zeros_like(image_orig)
    green_background[:] = (0, 255, 0)

    # Наложение масок на изображение
    for i, mask in enumerate(masks):
        class_name = results.names[int(classes[i])]
        if class_name == 'person':
            color_mask = np.zeros((640, 640, 3), dtype=np.uint8)
            resized_mask = cv2.resize(mask, (640, 640), interpolation=cv2.INTER_NEAREST)
            color_mask[resized_mask > 0] = person_color

            # Resize color_mask to original image size
            color_mask = cv2.resize(color_mask, (w_or, h_or), interpolation=cv2.INTER_NEAREST)

            # Replace green background with original image where mask is present
            mask_resized = cv2.resize(mask, (w_or, h_or), interpolation=cv2.INTER_NEAREST)
            green_background[mask_resized > 0] = image_orig[mask_resized > 0]

    return green_background


# Основной цикл для обработки видеофайла
def process_video(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть видеофайл")
        return

    # Получение параметров видеофайла
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame(frame)  # Обработка кадра
        out.write(processed_frame)  # Запись обработанного кадра в выходной видеофайл

        cv2.imshow('Processed Frame', processed_frame)  # Отображение обработанного кадра (для отладки)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Выход из цикла по нажатию 'q'
            break

    cap.release()  # Освобождение видеофайла
    out.release()  # Закрытие выходного видеофайла
    cv2.destroyAllWindows()  # Закрытие всех окон


if __name__ == "__main__":
    input_video_path = '20240609_070947.mp4'  # Путь к входному видеофайлу
    output_video_path = 'output_video.mp4'  # Путь к выходному видеофайлу
    process_video(input_video_path, output_video_path)
