import os
import numpy as np
from razdel import tokenize
import pandas as pd


def get_base_dataset(dataset_dir_name) -> dict:
    result_dict = {}
    base_path = dataset_dir_name
    # Проходим циклом по всем папкам, в том числе вложенным
    for dir in os.listdir(base_path):
        genre_path = os.path.join(base_path, dir)
        for author in os.listdir(genre_path):
            author_path = os.path.join(genre_path, author)
            for file_name in os.listdir(author_path):
                if 'info.csv' not in file_name:
                    path = os.path.join(author_path, file_name)
                    # Сохраняем текст
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read().replace("\n", " ")
                        try:
                            result_dict[author] += text
                        except KeyError:
                            result_dict[author] = text[1:]

    # Возвращаем словарь с 5 авторами, у которых наибольшее кол-тво символов в датасете
    return {k: v for k, v in sorted(result_dict.items(), key=lambda x: x[1], reverse=True)[:5]}


def tokenize_dataset(dataset) -> None:
    # Токенизируем слова
    for k, v in dataset.items():
        dataset[k] = [token.text for token in tokenize(v)]

    # Процесс выравнивания по медианному значению
    mean_list = np.array([])
    for tokens in dataset.values():
        mean_list = np.append(mean_list, len(tokens))

    median = int(np.median(mean_list))  # Зафиксировали медианное значение
    for author, tokens in dataset.items():
        if len(tokens) > median:
            temp = tokens[:median]
        else:
            multiplier = (median - len(tokens)) // len(tokens)
            extra = tokens * multiplier
            temp = tokens + extra
            ending = median - len(tokens) - len(extra)
            temp += tokens[:ending]
        dataset[author] = temp


def get_sequences(dataset) -> tuple:
    # Формируем словарь частотности
    index_dict = {}
    for author, tokens in dataset.items():
        a = pd.Series(tokens).value_counts().index.to_list()
        b = dict.fromkeys(a, 0)
        index_dict.update(b)
    number = 1
    for i in index_dict:
        index_dict[i] = number
        number += 1

    # Создаем словарь последовательностей токенов
    sequences = dict()
    for author, tokens in dataset.items():
        sequences[author] = []
        # Преобразование токенов в последовательности
        for token in tokens:
            sequences[author].append(index_dict[token])

    return index_dict, sequences
