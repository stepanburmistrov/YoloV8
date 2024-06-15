import cv2
import numpy as np
from ultralytics import YOLO
import os
from PIL import Image, ImageDraw, ImageFont

model = YOLO('yolov8x-seg.pt')

# Цвет для выделения объектов класса "person"
person_color = (0, 255, 0)  # Зеленый цвет
contour_color = (255, 255, 255)  # Белый цвет
contour_thickness = 30  # Толщина контура

# Текст для надписи
text = "Начинаем!"
font_path = "days.ttf"  # Путь к файлу шрифта
initial_font_size = 200  # Начальный размер шрифта

def process_image(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        print("Ошибка: не удалось загрузить изображение")
        return

    image_orig = frame.copy()
    h_or, w_or = frame.shape[:2]
    image = cv2.resize(frame, (640, 640))
    results = model(image)[0]

    classes = results.boxes.cls.cpu().numpy()
    masks = results.masks.data.cpu().numpy()

    largest_mask = None
    largest_area = 0

    # Находим самого большого человека
    for i, mask in enumerate(masks):
        class_name = results.names[int(classes[i])]
        if class_name == 'person':
            area = np.sum(mask)
            if area > largest_area:
                largest_area = area
                largest_mask = mask

    if largest_mask is None:
        print("Человек не найден")
        return

    # Resizing the largest mask to match the original image size
    largest_mask_resized = cv2.resize(largest_mask, (w_or, h_or), interpolation=cv2.INTER_NEAREST)

    # Convert the mask to a binary image
    binary_mask = (largest_mask_resized > 0).astype(np.uint8)

    # Создаем изображение с прозрачным фоном
    alpha_channel = np.zeros((h_or, w_or), dtype=np.uint8)
    alpha_channel[binary_mask > 0] = 255

    # Наложение маски на изображение
    rgb_image = cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB)
    transparent_background = np.zeros((h_or, w_or, 4), dtype=np.uint8)
    transparent_background[:, :, :3] = rgb_image
    transparent_background[:, :, 3] = alpha_channel

    # Обрезаем пустые области
    x, y, w, h = cv2.boundingRect(binary_mask)
    x = max(x - 20, 0)
    y = max(y - 20, 0)
    w = min(w + 40, w_or - x)
    h = min(h + 40, h_or - y)
    cropped_rgb_image = rgb_image[y:y + h, x:x + w]
    cropped_alpha_channel = alpha_channel[y:y + h, x:x + w]
    cropped_image = transparent_background[y:y + h, x:x + w]

    # Adjust mask coordinates for the cropped image
    cropped_binary_mask = binary_mask[y:y + h, x:x + w]

    # Рисуем контур вокруг человека на RGB изображении
    contours, _ = cv2.findContours(cropped_binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(cropped_rgb_image, contours, -1, contour_color, contour_thickness)

    # Объединяем обратно RGB и Alpha каналы
    cropped_image[:, :, :3] = cropped_rgb_image
    cropped_image[:, :, 3] = cropped_alpha_channel

    # Добавляем текст в нижнюю часть кадра
    pil_image = Image.fromarray(cropped_image)
    draw = ImageDraw.Draw(pil_image)
    font_size = initial_font_size
    font = ImageFont.truetype(font_path, font_size)

    # Подгоняем размер шрифта по ширине кадра
    max_text_width = pil_image.width * 0.8
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]

    while text_width > max_text_width:
        font_size -= 1
        font = ImageFont.truetype(font_path, font_size)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]

    text_position = ((pil_image.width - text_width) // 2, pil_image.height - text_bbox[3] - 10)
    draw.text(text_position, text, font=font, fill=(255, 255, 255, 255))

    # Изменяем масштаб картинки
    max_side = 512
    scale = max_side / max(cropped_image.shape[:2])
    new_size = (int(cropped_image.shape[1] * scale), int(cropped_image.shape[0] * scale))
    resized_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)

    # Сохраняем изображение в формате webp
    base_name, ext = os.path.splitext(image_path)
    output_path = f"{base_name}_processed.webp"
    resized_image.save(output_path, format="webp")
    print(f"Processed image saved to {output_path}")

    # Сохраняем изображение в формате PNG
    output_path_png = f"{base_name}_processed.png"
    resized_image.save(output_path_png, format="png")

# Путь к изображению, которое необходимо обработать
image_path = 'pg.jpg'
process_image(image_path)
