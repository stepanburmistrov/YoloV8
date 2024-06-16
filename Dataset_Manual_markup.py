import cv2
import os
import yaml
import shutil

# Путь к папке с исходными изображениями
full_images_path = 'output_images'
# Путь к папке для сохранения обработанных данных
dataset_path = 'dataset'
train_images_path = os.path.join(dataset_path, 'train', 'images')
train_labels_path = os.path.join(dataset_path, 'train', 'labels')
valid_images_path = os.path.join(dataset_path, 'valid', 'images')
valid_labels_path = os.path.join(dataset_path, 'valid', 'labels')
test_images_path = os.path.join(dataset_path, 'test', 'images')
test_labels_path = os.path.join(dataset_path, 'test', 'labels')
ready_images_path = os.path.join(full_images_path, 'ready')

os.makedirs(train_images_path, exist_ok=True)
os.makedirs(train_labels_path, exist_ok=True)
os.makedirs(valid_images_path, exist_ok=True)
os.makedirs(valid_labels_path, exist_ok=True)
os.makedirs(test_images_path, exist_ok=True)
os.makedirs(test_labels_path, exist_ok=True)
os.makedirs(ready_images_path, exist_ok=True)


window_name = 'Annotation Tool'
current_class = 0
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
annotations = []

# Функция для масштабирования изображения
def resize_image(image, size=(640, 640)):
    return cv2.resize(image, size)

# Обработка событий мыши
drawing = False
ix, iy = -1, -1

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, annotations, current_class
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            image = param['original_image'].copy()
            for annotation in annotations:
                cls, x1, y1, x2, y2 = annotation
                cv2.rectangle(image, (x1, y1), (x2, y2), colors[cls], 2)
            cv2.rectangle(image, (ix, iy), (x, y), colors[current_class], 2)
            cv2.imshow(window_name, image)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        annotations.append((current_class, ix, iy, x, y))
        image = param['original_image'].copy()
        for annotation in annotations:
            cls, x1, y1, x2, y2 = annotation
            cv2.rectangle(image, (x1, y1), (x2, y2), colors[cls], 2)
        cv2.imshow(window_name, image)
    elif event == cv2.EVENT_RBUTTONDOWN:  # Удаление последней рамки
        if annotations:
            removed_annotation = annotations.pop()
            image = param['original_image'].copy()  # Вернемся к оригинальному изображению
            for annotation in annotations:
                cls, x1, y1, x2, y2 = annotation
                cv2.rectangle(image, (x1, y1), (x2, y2), colors[cls], 2)
            cv2.imshow(window_name, image)
        else:
            image = param['original_image'].copy()
            cv2.imshow(window_name, image)

# Обновление файла data.yaml
def update_data_yaml():
    data_yaml_path = 'data.yaml'
    data = {
        'train': 'dataset/train/images',
        'val': 'dataset/valid/images',
        'test': 'dataset/test/images',
        'nc': 4,
        'names': ['class0', 'class1', 'class2', 'class3']
    }
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=None, sort_keys=False)
    print(f"Updated {data_yaml_path}")

# Загрузка и обработка изображений
for filename in os.listdir(full_images_path):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(full_images_path, filename)
        image = cv2.imread(image_path)
        image = resize_image(image)
        original_image = image.copy()
        annotations = []

        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, draw_rectangle, param={'original_image': original_image})

        while True:
            image_with_annotations = original_image.copy()
            for annotation in annotations:
                cls, x1, y1, x2, y2 = annotation
                cv2.rectangle(image_with_annotations, (x1, y1), (x2, y2), colors[cls], 2)
            cv2.imshow(window_name, image_with_annotations)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):  # Нажатие пробела для сохранения
                # Сохранение изображения
                output_image_path = os.path.join(train_images_path, filename)
                cv2.imwrite(output_image_path, original_image)
                print(f"Saved image to {output_image_path}")

                # Сохранение текстовых данных
                label_filename = os.path.splitext(filename)[0] + '.txt'
                label_file_path = os.path.join(train_labels_path, label_filename)
                with open(label_file_path, 'w') as f:
                    for annotation in annotations:
                        cls, x1, y1, x2, y2 = annotation
                        x_center = (x1 + x2) / 2 / 640
                        y_center = (y1 + y2) / 2 / 640
                        width = abs(x2 - x1) / 640
                        height = abs(y2 - y1) / 640
                        f.write(f"{cls} {x_center} {y_center} {width} {height}\n")
                print(f"Saved labels to {label_file_path}")

                # Обновление data.yaml
                update_data_yaml()

                # Перемещение обработанного изображения
                ready_image_path = os.path.join(ready_images_path, filename)
                shutil.move(image_path, ready_image_path)
                print(f"Moved image to {ready_image_path}")
                break
            elif key == 27:  # Нажатие Esc для пропуска изображения
                print("Skipped image")
                break
            elif key in [ord(str(i)) for i in range(10)]:  # Выбор класса
                current_class = int(chr(key))
                print(f"Selected class: {current_class}")

        cv2.destroyAllWindows()

