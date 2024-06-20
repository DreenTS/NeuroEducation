from keras.applications import MobileNet
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from dataset import prepare_dataset


def model_maker():
    base_model = MobileNet(include_top=False, input_shape=(128, 128, 3))

    for layer in base_model.layers[:]:
        layer.trainable = False

    _model = Sequential()
    _model.add(base_model)
    _model.add(GlobalAveragePooling2D())
    _model.add(Dense(64, activation='relu'))
    _model.add(Dropout(0.5))
    _model.add(Dense(2, activation='softmax'))

    return _model


if __name__ == '__main__':
    model = model_maker()
    # print(model.summary())

    img_path = 'cat-and-dog/training_set/training_set'
    train_last_ind, validation_last_ind, test_last_ind = 2000, 3000, 4000
    train_dir, validation_dir, test_dir = prepare_dataset(img_path, train_last_ind, validation_last_ind, test_last_ind)

    # генератор для обучающей выборки
    train_datagen = ImageDataGenerator(
        rescale=1./255,  # нормализация данных
        rotation_range=40,  # поворот 40 градусов
        width_shift_range=0.2,  # смещенние изображения по горизонтали
        height_shift_range=0.2,  # смещенние изображения по вертикали
        shear_range=0.2,  # случайный сдвиг
        zoom_range=0.2,  # случайное масштабирование
        horizontal_flip=True,  # отражение по горизонтали
        fill_mode='nearest'  # стратегия заполнения пустых пикселей при трансформации
    )
    # генератор для проверочной выборки
    test_datagen = ImageDataGenerator(rescale=1./255)

    # генерация картинок из папки для обучающей выборки
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(128, 128),
        batch_size=20,
        class_mode='categorical'
    )

    # генерация картинок из папки для проверочной выборки
    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(128, 128),
        batch_size=20,
        class_mode='categorical'
    )

    # компиляция модели
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(learning_rate=2e-5),
                  metrics=['accuracy'])

    # обучаем модель fit_generator и fit в данном контексте аналогичные функции
    history = model.fit(
        train_generator,
        steps_per_epoch=100,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=50
    )
