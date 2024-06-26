import os
import shutil
import random
from sklearn.model_selection import train_test_split


def create_dataset(
        path_list: list,  # Путь к файлам с изображениями классов
        dataset_path: str,  # Путь к папке с выборками
):
    for path in path_list:
        label, file_name = path.split('/')[-2:]
        class_path = os.path.join(dataset_path, label)
        if not os.path.exists(class_path):
            os.mkdir(class_path)
        new_path = os.path.join(class_path, file_name)
        shutil.copyfile(path, new_path)


def prepare_dataset(src_path, dist_path, k_random_labels=None) -> tuple:
    num_skipped = 0  # счетчик поврежденных файлов
    for folder_name in os.listdir(src_path):  # перебираем папки
        folder_path = os.path.join(src_path, folder_name)  # склеиваем путь
        for fname in os.listdir(folder_path):  # получаем список файлов в папке
            fpath = os.path.join(folder_path, fname)  # получаем путь до файла
            try:
                fobj = open(fpath, "rb")  # пытаемся открыть файл для бинарного чтения (rb)
                is_jfif = b"JFIF" in fobj.peek(10)  # получаем первые 10 байт из файла и ищем в них бинарный вариант строки JFIF
            finally:
                fobj.close()  # Закрываем файл

            if not is_jfif:  # Если не нашли JFIF строку
                # Увеличиваем счетчик
                num_skipped += 1
                # Удаляем поврежденное изображение
                os.remove(fpath)

    print(f"Удалено изображений: {num_skipped}")

    labels_str = sorted(os.listdir(src_path))
    if k_random_labels:
        labels_str = random.choices(labels_str, k=10)

    labels = []
    data = []
    for label in range(len(labels_str)):
        label_path = src_path + labels_str[label]
        data += [f'{label_path}/{file_path}' for file_path in os.listdir(label_path)]
        labels += [label] * len(os.listdir(label_path))

    train_img, test_img, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, stratify=labels,
                                                                      random_state=0)

    train_img, val_img, train_labels, val_labels = train_test_split(train_img, train_labels, test_size=0.25,
                                                                    stratify=train_labels, random_state=0)

    # Папка с папками картинок, рассортированных по категориям
    if os.path.exists(dist_path):
        shutil.rmtree(dist_path)

    os.mkdir(dist_path)
    train_dir = os.path.join(dist_path, 'train')
    os.mkdir(train_dir)

    val_dir = os.path.join(dist_path, 'validation')
    os.mkdir(val_dir)

    test_dir = os.path.join(dist_path, 'test')
    os.mkdir(test_dir)

    create_dataset(path_list=train_img, dataset_path=train_dir)
    create_dataset(path_list=val_img, dataset_path=val_dir)
    create_dataset(path_list=test_img, dataset_path=test_dir)

    return train_dir, val_dir, test_dir, len(labels_str)
