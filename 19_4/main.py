from keras.callbacks import ModelCheckpoint
from dataset import prepare_dataset
from graphs import show_plot
from model import build_model
from samples import make_samples


if __name__ == '__main__':
    IMAGE_SIZE = (380, 380)
    BATCH_SIZE = 128
    CALLBACKS = [
        ModelCheckpoint(filepath='19_4_best_model_pretrain.keras',
                        monitor='val_acc',
                        save_best_only=True,
                        mode='max',
                        verbose=0)
    ]

    # TODO: Основное задание, используются все классы датасета

    # # Папка с папками картинок, рассортированных по категориям
    base_dataset_path = './stanford_dogs/'

    # Папка с папками картинок, рассортированных по категориям
    src_dataset_path = './dataset/'

    train_dir, val_dir, test_dir, len_labels = prepare_dataset(base_dataset_path, src_dataset_path)

    train_ds, val_ds, test_ds = make_samples(IMAGE_SIZE, BATCH_SIZE, len_labels,
                                             train_dir, val_dir, test_dir)

    big_model = build_model(img_size=IMAGE_SIZE, num_classes=len_labels)

    epochs = 5

    print(f'\n{IMAGE_SIZE = }\n{BATCH_SIZE = }\n{epochs = }\n')
    history = big_model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=CALLBACKS)

    show_plot(history)

    big_model.evaluate(test_ds)

    big_model.save('model_before_timeout.keras')

    # TODO: Дополнительное задание, используется 10 случайно выбранных классов датасета

    # Папка с папками картинок, рассортированных по категориям
    base_dataset_path = './stanford_dogs/'

    # Папка с папками картинок, рассортированных по категориям
    src_dataset_path = './small_dataset/'

    train_dir, val_dir, test_dir, len_labels = prepare_dataset(base_dataset_path,
                                                               src_dataset_path,
                                                               k_random_labels=10)

    train_ds, val_ds, test_ds = make_samples(IMAGE_SIZE, BATCH_SIZE, len_labels,
                                             train_dir, val_dir, test_dir)

    small_model = build_model(img_size=IMAGE_SIZE, num_classes=len_labels)

    epochs = 4

    print(f'\n{IMAGE_SIZE = }\n{BATCH_SIZE = }\n{epochs = }\n')
    history = small_model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=CALLBACKS)

    show_plot(history)

    small_model.evaluate(test_ds)
