import cv2
import numpy as np
from ultralytics import YOLO
import os

model = YOLO('yolov8n-cls.pt')


def process_image(img):
    # Обработка кадра с помощью модели
    results = model(img)[0]

    # Отображение результатов классификации на изображении
    if results.probs is not None:
        # Доступ к вершинам классификации
        top1_idx = results.probs.top1  # Индекс класса с наивысшей вероятностью
        top1_conf = results.probs.top1conf.item()  # Вероятность для класса с наивысшей вероятностью
        class_name = results.names[top1_idx]  # Получаем имя класса по индексу

        # Отображаем класс и вероятность на кадре
        label = f"{class_name}: {top1_conf:.2f}"
        cv2.putText(img, label, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (255, 0, 0), 3)

    return image

image = cv2.imread("dog.jpg")
image = process_image(image)
cv2.imwrite('result.jpg', image)
