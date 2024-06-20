from keras.applications import MobileNet
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Dense, Dropout


def model_maker():
    # базовая модель - предобученная MobileNet
    base_model = MobileNet(include_top=False, input_shape=(128, 128, 3))

    # запрет на изменение предобученных слоев модели
    for layer in base_model.layers[:]:
        layer.trainable = False

    # строим нашу НС
    _model = Sequential()
    _model.add(base_model)
    _model.add(GlobalAveragePooling2D())
    _model.add(Dense(64, activation='relu'))
    # прореживание
    _model.add(Dropout(0.5))
    # output
    _model.add(Dense(2, activation='softmax'))

    return _model
