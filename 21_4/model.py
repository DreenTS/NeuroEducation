from navec import Navec
from keras.models import Sequential
from keras.layers import Embedding, Dense, SpatialDropout1D, BatchNormalization, Dropout, Input, GRU, LSTM, Bidirectional
import numpy as np


def get_embedding_matrix(max_w, emb_dim, index_dict) -> np.ndarray:
    navec = Navec.load('navec_hudlit_v1_12B_500K_300d_100q.tar')
    embeddings_index = navec

    # Заполняем матрицу embedding по словарю
    embedding_matrix = np.zeros((max_w, emb_dim))
    for word, i in index_dict.items():
        if i < max_w:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

    return embedding_matrix


def get_model(index_dict, win_size, labels_len) -> Sequential:
    max_words = len(index_dict) + 1
    embedding_dim = 300  # обусловлено предобученной моделью Navec

    model = Sequential()
    model.add(Input(shape=(win_size,)))
    model.add(Embedding(max_words, embedding_dim))
    model.add(SpatialDropout1D(0.3))
    model.add(BatchNormalization())

    # Два двунаправленных рекуррентных слоя LSTM
    model.add(Bidirectional(LSTM(8, return_sequences=True)))
    model.add(Bidirectional(LSTM(8, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    # Двауррентных слоя GRU
    model.add(GRU(16, return_sequences=True, reset_after=True))
    model.add(GRU(16, reset_after=True))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    # Дополнительный полносвязный слой
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(labels_len, activation='softmax'))

    embedding_matrix = get_embedding_matrix(max_words, embedding_dim, index_dict)

    # Загрузка подготовленной матрицы в слой Embedding
    model.layers[0].set_weights([embedding_matrix])
    # Заморозка слоя Embedding
    model.layers[0].trainable = False

    return model
