import numpy as np
from keras import utils


def seq_split(sequence, win_size, step):
    # Делим последовательность на отрезки
    result = []
    for i in range(0, len(sequence) - win_size + 1, step):
        result.append(sequence[i:i + win_size])
    return result


def seq_vectorize(
        seq_list,  # Последовательность
        val_split,  # Доля на валидационную выборку
        test_split,  # Доля на тестовую выборку
        class_list,  # Список классов
        win_size,  # Ширина скользящего окна
        step  # Шаг скользящего окна
):
    # Списки для результирующих данных
    x_train, y_train, x_val, y_val, x_test, y_test = [], [], [], [], [], []

    # Пробежимся по всем классам:
    for cls, class_item in enumerate(class_list):

        # Пороговое значение индекса для разбивки на тестовую и обучающую выборки
        test_gate_split = int(len(seq_list[cls]) * (1 - test_split))
        val_gate_split = int(len(seq_list[cls][:test_gate_split]) * (1 - val_split))

        # Разбиваем последовательность токенов класса на отрезки
        # последовательность до порога попадет в обучающую выборку
        vectors_train = seq_split(seq_list[cls][:val_gate_split], win_size, step)
        # последовательность после порога попадет в валидационную выборку
        vectors_val = seq_split(seq_list[cls][val_gate_split:test_gate_split], win_size, step)
        # последовательность после порога попадет в тестовую выборку
        vectors_test = seq_split(seq_list[cls][test_gate_split:], win_size, step)

        # Добавляем отрезки в выборку
        x_train += vectors_train
        x_val += vectors_val
        x_test += vectors_test

        # Для всех отрезков класса добавляем метки класса в виде one-hot-encoding
        # Каждую метку берем len(vectors) раз, так она одинакова для всех выборок одного класса
        y_train += [utils.to_categorical(cls, len(class_list))] * len(vectors_train)
        y_val += [utils.to_categorical(cls, len(class_list))] * len(vectors_val)
        y_test += [utils.to_categorical(cls, len(class_list))] * len(vectors_test)

    # Возвращаем результатов как numpy-массивов
    return np.array(x_train), np.array(y_train), np.array(x_val), np.array(y_val), np.array(x_test), np.array(y_test)
