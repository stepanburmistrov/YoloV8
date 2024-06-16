import cv2
import os
import numpy as np

input_folder = 'output_images'
output_folder = 'dataset/train'
output_images_folder = os.path.join(output_folder, 'images')
output_labels_folder = os.path.join(output_folder, 'labels')

os.makedirs(output_images_folder, exist_ok=True)
os.makedirs(output_labels_folder, exist_ok=True)

lower_hsv = np.array([89, 71, 120])
upper_hsv = np.array([180, 255, 255])

def find_mask(image, lower_hsv, upper_hsv):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
    return mask

def find_bounding_rect(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        return x, y, x + w, y + h
    else:
        return None

def normalize_coordinates(x1, y1, x2, y2, img_width, img_height):
    x_center = (x1 + x2) / 2 / img_width
    y_center = (y1 + y2) / 2 / img_height
    width = abs(x2 - x1) / img_width
    height = abs(y2 - y1) / img_height
    return x_center, y_center, width, height

for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        resized_image = cv2.resize(image, (640,640))
        mask = find_mask(resized_image, lower_hsv, upper_hsv)
        bounding_rect = find_bounding_rect(mask)

        if bounding_rect is not None: 
            x1, y1, x2, y2 = bounding_rect
            x_center, y_center, width, height = normalize_coordinates(x1, y1, x2, y2, 640, 640)

            # Сохранение изображения
            output_image_path = os.path.join(output_images_folder, filename)
            cv2.imwrite(output_image_path, resized_image)

            # Сохранение 
            label_filename = os.path.splitext(filename)[0] + '.txt'
            label_file_path = os.path.join(output_labels_folder, label_filename)
            with open(label_file_path, 'w') as f:
                f.write(f"0 {x_center} {y_center} {width} {height}\n")

            print(f"Processed and saved {filename}")

print("Подготовка датасета завершена.")
