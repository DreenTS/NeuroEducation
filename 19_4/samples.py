from keras.utils import image_dataset_from_directory
from keras import layers
from keras.ops import one_hot
from tensorflow import data as tf_data


def img_augmentation(images, img_aug_layers):
    # Слои - это функции, которые мы последовательно применяем к входным данным
    for layer in img_aug_layers:
        images = layer(images)
    return images


def make_samples(img_size, batch_size, len_labels, train_dir, val_dir, test_dir) -> tuple:
    train_ds = image_dataset_from_directory(
        train_dir,  # путь к папке с данными
        seed=0,  # воспроизводимость результата генерации (результаты с одинаковым числом - одинаковы),
        shuffle=True,  # перемешиваем датасет
        image_size=img_size,  # размер генерируемых изображений
        batch_size=batch_size,  # размер мини-батча
    )

    val_ds = image_dataset_from_directory(
        val_dir,  # путь к папке с данными
        seed=0,  # воспроизводимость результата генерации (результаты с одинаковым числом - одинаковы),
        shuffle=False,
        image_size=img_size,  # размер генерируемых изображений
        batch_size=batch_size,  # размер мини-батча
    )

    test_ds = image_dataset_from_directory(
        test_dir,  # путь к папке с данными
        seed=0,  # воспроизводимость результата генерации (результаты с одинаковым числом - одинаковы),
        shuffle=False,
        image_size=img_size,  # размер генерируемых изображений
        batch_size=batch_size,  # размер мини-батча
    )

    img_augmentation_layers = [
        layers.RandomRotation(factor=0.15),  # Вращаем изображение в пределах 15%
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),  # Сдвиг на 10% по вертикали и горизонтали
        layers.RandomFlip(),  # Отражение по вертикали и горизонтали
        layers.RandomContrast(factor=0.1),  # Изменяем контрастность на 10%
    ]

    # Применяем `img_augmentation` к обучающей выборке
    train_ds = train_ds.map(
        lambda img, label: (img_augmentation(img, img_augmentation_layers), one_hot(label, len_labels)),  # One-hot кодирование
        num_parallel_calls=tf_data.AUTOTUNE,
        # число потоков для обработки в map (автонастройка зависит от возможностей процессора)
    )

    val_ds = val_ds.map(
        lambda img, label: (img, one_hot(label, len_labels)),  # One-hot кодирование
        num_parallel_calls=tf_data.AUTOTUNE,
        # число потоков для обработки в map (автонастройка зависит от возможностей процессора)
    )

    test_ds = test_ds.map(
        lambda img, label: (img, one_hot(label, len_labels)),  # One-hot кодирование
        num_parallel_calls=tf_data.AUTOTUNE,
        # число потоков для обработки в map (автонастройка зависит от возможностей процессора)
    )

    # Предварительная выборка примеров в память GPU или оперативную память
    # Помогает максимально эффективно использовать графический процессор

    train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf_data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf_data.AUTOTUNE)

    return train_ds, val_ds, test_ds
