from ultralytics import YOLO
import cv2
import numpy as np
import os

# Загрузка модели YOLOv8-Pose
model = YOLO('yolov8n-pose.pt')

# Словарь цветов для различных классов
colors = {
    'white': (255, 255, 255),
    'red': (0, 0, 255),
    'blue': (255, 0, 0)
}

def draw_skeleton(image, keypoints, confs, pairs, color):
    for (start, end) in pairs:
        if confs[start] > 0.5 and confs[end] > 0.5:
            x1, y1 = int(keypoints[start][0]), int(keypoints[start][1])
            x2, y2 = int(keypoints[end][0]), int(keypoints[end][1])
            if (x1, y1) != (0, 0) and (x2, y2) != (0, 0):  # Игнорирование точек в (0, 0)
                cv2.line(image, (x1, y1), (x2, y2), color, 2)

def process_image(image_path):
    # Загрузка изображения
    image = cv2.imread(image_path)
    if image is None:
        print("Ошибка: не удалось загрузить изображение")
        return

    # Обработка изображения с помощью модели
    results = model(image)[0]

    # Проверка на наличие обнаруженных объектов
    if hasattr(results, 'boxes') and hasattr(results.boxes, 'cls') and len(results.boxes.cls) > 0:
        classes_names = results.names
        classes = results.boxes.cls.cpu().numpy()
        boxes = results.boxes.xyxy.cpu().numpy().astype(np.int32)

        # Обработка ключевых точек
        if results.keypoints:
            keypoints = results.keypoints.data.cpu().numpy()
            confs = results.keypoints.conf.cpu().numpy()
            
            for i, (class_id, box, kp, conf) in enumerate(zip(classes, boxes, keypoints, confs)):
                draw_box=False
                if draw_box:
                    class_name = classes_names[int(class_id)]
                    color = colors['white']
                    x1, y1, x2, y2 = box
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Визуализация ключевых точек с номерами
                for j, (point, point_conf) in enumerate(zip(kp, conf)):
                    if point_conf > 0.5:  # Фильтрация по уверенности
                        x, y = int(point[0]), int(point[1])
                        if (x, y) != (0, 0):  # Игнорирование точек в (0, 0)
                            cv2.circle(image, (x, y), 5, colors['blue'], -1)
                            cv2.putText(image, str(j), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['blue'], 2)

                # Рисование скелета
                draw_skeleton(image, kp, conf, [(5, 7), (7, 9), (6, 8), (8, 10)], colors['white']) # Руки
                draw_skeleton(image, kp, conf, [(11, 13), (13, 15), (12, 14), (14, 16)], colors['red']) # Ноги
                draw_skeleton(image, kp, conf, [(5, 11), (6, 12)], colors['blue']) # Тело

    # Сохранение и отображение результатов
    output_path = os.path.splitext(image_path)[0] + "_pose_detected.jpg"
    cv2.imwrite(output_path, image)
    print(f"Сохранено изображение с результатами: {output_path}")

    cv2.imshow('YOLOv8-Pose Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Путь к изображению для обработки
image_path = 'd.jpg'
process_image(image_path)
