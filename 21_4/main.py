from keras.src.callbacks import ModelCheckpoint
from keras.models import load_model

from dataset import get_base_dataset, tokenize_dataset, get_sequences
from samples import seq_vectorize
from model import get_model
from graphs import show_plot, show_confusion_matrix


if __name__ == '__main__':
    # Формируем датасет
    base_dict = get_base_dataset(dataset_dir_name='rus_lit')
    labels = list(base_dict.keys())

    # Токенизируем данные + выравниваем кол-тво по медианному значению
    tokenize_dataset(base_dict)

    # Выводим сводку
    total = sum(len(i) for i in base_dict.values())
    print(f'Датасет состоит из {total} символов')
    print('Общая выборка по писателям:')
    for author in labels:
        print(f'{author} - {len(base_dict[author])} символов, доля в общей базе: '
              f'{len(base_dict[author]) / total * 100:.2f}%')

    # Получаем словари частотности и последовательностей токенов
    index_dict, sequences = get_sequences(base_dict)
    print("\nТокены:", list(base_dict.values())[0][:10])
    print("Последовательности:", list(sequences.values())[0][:10])

    # Формируем выборки
    win_size, win_step = 1000, 100
    x_train, y_train, x_val, y_val, x_test, y_test = seq_vectorize(seq_list=list(sequences.values()),
                                                                   val_split=0.1,
                                                                   test_split=0.1,
                                                                   class_list=sequences.keys(),
                                                                   win_size=win_size,
                                                                   step=win_step)

    print(f'\nФорма входных данных для обучающей выборки: {x_train.shape}')
    print(f'Форма выходных данных (меток) для обучающей выборки: {y_train.shape}')
    print(f'Форма входных данных для валидационной выборки: {x_val.shape}')
    print(f'Форма выходных данных (меток) для валидационной выборки: {y_val.shape}')
    print(f'Форма входных данных для тестовой выборки: {x_test.shape}')
    print(f'Форма выходных данных (меток) для тестовой выборки: {y_test.shape}\n')

    # Создаем модель
    model = get_model(index_dict=index_dict, win_size=win_size, labels_len=len(labels))
    CALLBACKS = [
        ModelCheckpoint(filepath='pretrain/21_4_best_model_pretrain.weights.h5',
                        monitor='val_acc',
                        save_weights_only=True,
                        save_best_only=True,
                        mode='max',
                        verbose=0),
        ModelCheckpoint(filepath='pretrain/21_4_best_model_pretrain.keras',
                        monitor='val_acc',
                        save_best_only=True,
                        mode='max',
                        verbose=0)
    ]

    # Компилируем модель
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    # Запускаем процесс обучения
    history = model.fit(x_train, y_train, epochs=35, batch_size=128, validation_data=(x_val, y_val), callbacks=CALLBACKS)

    # Выводим историю обучения
    show_plot(history)

    # Загрузка предобученной модели
    model = load_model('pretrain/21_4_best_model_pretrain.keras')

    # Предсказываем на тестовых данных
    y_pred = model.predict(x_test)
    show_confusion_matrix(y_test, y_pred, list(sequences.keys()))

