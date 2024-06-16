import cv2
import os

# Папки с изображениями и метками
images_path = 'dataset/train/images'
labels_path = 'dataset/train/labels'

# Папка для сохранения изображений с нарисованными прямоугольниками
output_folder = 'checked_images'
os.makedirs(output_folder, exist_ok=True)

# Цвета для классов (можно добавить больше цветов, если классов больше)
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

# Чтение всех файлов изображений и меток
images = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Функция для преобразования координат из нормализованных значений в пиксели
def denormalize_coordinates(x_center, y_center, width, height, img_width, img_height):
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)
    return x1, y1, x2, y2

# Обработка изображений
for image_file in images:
    image_path = os.path.join(images_path, image_file)
    label_file = os.path.splitext(image_file)[0] + '.txt'
    label_path = os.path.join(labels_path, label_file)

    # Проверка наличия файла меток
    if not os.path.exists(label_path):
        print(f"Label file not found for image: {image_file}")
        continue

    # Загрузка изображения
    image = cv2.imread(image_path)
    img_height, img_width = image.shape[:2]

    # Чтение файла меток и рисование прямоугольников
    with open(label_path, 'r') as f:
        for line in f:
            cls, x_center, y_center, width, height = map(float, line.strip().split())
            x1, y1, x2, y2 = denormalize_coordinates(x_center, y_center, width, height, img_width, img_height)
            color = colors[int(cls) % len(colors)]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f'class {int(cls)}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Сохранение изображения с нарисованными прямоугольниками
    output_path = os.path.join(output_folder, image_file)
    cv2.imwrite(output_path, image)
    print(f"Saved checked image: {output_path}")

print("Проверка разметки завершена.")
