from model import model_maker
from dataset import prepare_dataset
from graphs import show_history

from keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator


if __name__ == '__main__':
    model = model_maker()
    # print(model.summary())

    # Подготавливаем датасет для обучения:
    # разделяем на тренировочную, валидационную и констрольную выборки
    img_path = './cat-and-dog/training_set/training_set'
    train_last_ind, validation_last_ind, test_last_ind = 2000, 3000, 4000
    # получаем пути до директорий с файлами
    train_dir, validation_dir, test_dir = prepare_dataset(img_path, train_last_ind, validation_last_ind, test_last_ind)

    # генератор для обучающей выборки
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,  # нормализация данных
        rotation_range=40,  # поворот 40 градусов
        width_shift_range=0.2,  # смещенние изображения по горизонтали
        height_shift_range=0.2,  # смещенние изображения по вертикали
        shear_range=0.2,  # случайный сдвиг
        zoom_range=0.2,  # случайное масштабирование
        horizontal_flip=True,  # отражение по горизонтали
        fill_mode='nearest'  # стратегия заполнения пустых пикселей при трансформации
    )

    # генератор для проверочной и контрольной выборок
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # генерация картинок из папки для обучающей выборки
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(128, 128),
        batch_size=64,
        class_mode='categorical'
    )

    # генерация картинок из папки для проверочной выборки
    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(128, 128),
        batch_size=64,
        class_mode='categorical'
    )

    # генерация картинок из папки для тестовой выборки
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(128, 128),
        batch_size=64,
        class_mode='categorical'
    )

    # компиляция модели
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(learning_rate=2e-5),
                  metrics=['accuracy'])

    # обучаем модель
    history = model.fit(
        train_generator,
        epochs=30,
        validation_data=validation_generator
    )

    # Рисуем графики точности и потерь
    show_history(history)

    # Проверяем модель на тестовой выборке
    model.evaluate(test_generator)
