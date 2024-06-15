import cv2
import numpy as np
from ultralytics import YOLO
import os

# Загрузка модели YOLOv8
model = YOLO('yolov8x-seg.pt')

colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (192, 192, 192), (128, 128, 128), (128, 0, 0), (128, 128, 0),
    (0, 128, 0), (128, 0, 128), (0, 128, 128), (0, 0, 128), (72, 61, 139),
    (47, 79, 79), (47, 79, 47), (0, 206, 209), (148, 0, 211), (255, 20, 147)
]

def process_image(image_path):
    # Проверка наличия папки для сохранения результатов
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Загрузка изображения
    image = cv2.imread(image_path)
    image_orig = image.copy()
    h_or, w_or = image.shape[:2]
    image = cv2.resize(image, (640, 640))
    results = model(image)[0]
    
    classes_names = results.names
    classes = results.boxes.cls.cpu().numpy()
    masks = results.masks.data.cpu().numpy()

    # Наложение масок на изображение
    for i, mask in enumerate(masks):
        color = colors[int(classes[i]) % len(colors)]
        
        # Изменение размера маски перед созданием цветной маски
        mask_resized = cv2.resize(mask, (w_or, h_or))
        
        # Создание цветной маски
        color_mask = np.zeros((h_or, w_or, 3), dtype=np.uint8)
        color_mask[mask_resized > 0] = color

        # Сохранение маски каждого класса в отдельный файл
        mask_filename = os.path.join('results', f"{classes_names[classes[i]]}_{i}.png")
        cv2.imwrite(mask_filename, color_mask)

        # Наложение маски на исходное изображение
        image_orig = cv2.addWeighted(image_orig, 1.0, color_mask, 0.5, 0)


    # Сохранение измененного изображения
    new_image_path = os.path.join('results', os.path.splitext(os.path.basename(image_path))[0] + '_segmented' + os.path.splitext(image_path)[1])
    cv2.imwrite(new_image_path, image_orig)
    print(f"Segmented image saved to {new_image_path}")

process_image('segmentation_test.png')
