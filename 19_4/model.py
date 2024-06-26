from keras import Model
from keras.optimizers import Adam
from keras import layers
from keras.applications import EfficientNetV2B2


def build_model(img_size, num_classes):
    inputs = layers.Input(shape=(img_size[0], img_size[1], 3))
    model = EfficientNetV2B2(include_top=False, input_tensor=inputs, weights="imagenet")

    # Заморозка предобученных весов
    model.trainable = False

    # Замена верхних слоев
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    # Компилируем
    model = Model(inputs, outputs, name="EfficientNetV2")
    optimizer = Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["acc"]
    )
    return model
