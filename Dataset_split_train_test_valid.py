import os
import shutil
import random

# Параметры для разделения данных
test_percent = 0.2  # Процент данных для тестирования
valid_percent = 0.1  # Процент данных для валидации

# Путь к папке с данными
dataset_path = 'dataset'
train_images_path = os.path.join(dataset_path, 'train', 'images')
train_labels_path = os.path.join(dataset_path, 'train', 'labels')
valid_images_path = os.path.join(dataset_path, 'valid', 'images')
valid_labels_path = os.path.join(dataset_path, 'valid', 'labels')
test_images_path = os.path.join(dataset_path, 'test', 'images')
test_labels_path = os.path.join(dataset_path, 'test', 'labels')

os.makedirs(valid_images_path, exist_ok=True)
os.makedirs(valid_labels_path, exist_ok=True)
os.makedirs(test_images_path, exist_ok=True)
os.makedirs(test_labels_path, exist_ok=True)

# Получение всех файлов изображений и соответствующих меток
images = [f for f in os.listdir(train_images_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
labels = [f for f in os.listdir(train_labels_path) if f.endswith('.txt')]

# Убедимся, что количество изображений и меток совпадает
images.sort()
labels.sort()

# Проверка на соответствие количества изображений и меток
if len(images) != len(labels):
    print("Количество изображений и меток не совпадает.")
    exit()

# Перемешивание данных
data = list(zip(images, labels))
random.shuffle(data)
images, labels = zip(*data)

# Разделение данных
num_images = len(images)
num_test = int(num_images * test_percent)
num_valid = int(num_images * valid_percent)
num_train = num_images - num_test - num_valid

# Перемещение данных в соответствующие папки
def move_files(file_list, source_image_dir, source_label_dir, dest_image_dir, dest_label_dir):
    for file in file_list:
        image_path = os.path.join(source_image_dir, file)
        label_path = os.path.join(source_label_dir, os.path.splitext(file)[0] + '.txt')
        shutil.move(image_path, os.path.join(dest_image_dir, file))
        shutil.move(label_path, os.path.join(dest_label_dir, os.path.splitext(file)[0] + '.txt'))

# Перемещение тестовых данных
move_files(images[:num_test], train_images_path, train_labels_path, test_images_path, test_labels_path)

# Перемещение валидационных данных
move_files(images[num_test:num_test + num_valid], train_images_path, train_labels_path, valid_images_path, valid_labels_path)

# Оставшиеся данные остаются в папке train

print(f"Перемещено {num_test} изображений в папку test.")
print(f"Перемещено {num_valid} изображений в папку valid.")
print(f"Осталось {num_train} изображений в папке train.")
