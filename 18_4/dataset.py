import os
import shutil


# Функция создания подвыборок (папок с файлами)
def create_dataset(
        img_path: str,  # Путь к файлам с изображениями классов
        new_path: str,  # Путь к папке с выборками
        class_name: str,  # Имя класса (оно же и имя папки)
        start_index: int,  # Стартовый индекс изображения, с которого начинаем подвыборку
        end_index: int  # Конечный индекс изображения, до которого создаем подвыборку

):
    src_path = os.path.join(img_path, class_name)  # Полный путь к папке с изображениями класса
    dst_path = os.path.join(new_path, class_name)  # Полный путь к папке с новым датасетом класса

    # Получение списка имен файлов с изображениями текущего класса
    class_files = os.listdir(src_path)

    # Создаем подпапку, используя путь
    os.mkdir(dst_path)

    # Перебираем элементы, отобранного списка с начального по конечный индекс
    for fname in class_files[start_index:end_index]:
        # Путь к файлу (источник)
        src = os.path.join(src_path, fname)
        # Новый путь расположения файла (назначение)
        dst = os.path.join(dst_path, fname)
        # Копируем файл из источника в новое место (назначение)
        shutil.copyfile(src, dst)


def prepare_dataset(img_path, last_train, last_validation, last_test):
    # Папка с папками картинок, рассортированных по категориям
    IMAGE_PATH = img_path

    # Папка в которой будем создавать выборки
    BASE_DIR = '18_4/dataset/'

    # Определение списка имен классов
    CLASS_LIST = sorted(os.listdir(IMAGE_PATH))

    # Определение количества классов
    CLASS_COUNT = len(CLASS_LIST)

    # При повторном запуске пересоздаим структуру каталогов
    # Если папка существует, то удаляем ее со всеми вложенными каталогами и файлами
    if os.path.exists(BASE_DIR):
        shutil.rmtree(BASE_DIR)

    # Создаем папку по пути BASE_DIR
    os.mkdir(BASE_DIR)

    # Сцепляем путь до папки с именем вложенной папки. Аналогично BASE_DIR + '/train'
    train_dir = os.path.join(BASE_DIR, 'train')

    # Создаем подпапку, используя путь
    os.mkdir(train_dir)

    # Сцепляем путь до папки с именем вложенной папки. Аналогично BASE_DIR + '/validation'
    validation_dir = os.path.join(BASE_DIR, 'validation')

    # Создаем подпапку, используя путь
    os.mkdir(validation_dir)

    # Сцепляем путь до папки с именем вложенной папки. Аналогично BASE_DIR + '/test'
    test_dir = os.path.join(BASE_DIR, 'test')

    # Создаем подпапку, используя путь
    os.mkdir(test_dir)

    for class_label in range(CLASS_COUNT):  # Перебор по всем классам по порядку номеров (их меток)
        class_name = CLASS_LIST[class_label]  # Выборка имени класса из списка имен

        # Создаем обучающую выборки для заданных классов
        create_dataset(IMAGE_PATH, train_dir, class_name, 0, last_train)
        create_dataset(IMAGE_PATH, validation_dir, class_name, last_train, last_validation)
        create_dataset(IMAGE_PATH, test_dir, class_name, last_validation, last_test)

    return train_dir, validation_dir, test_dir
