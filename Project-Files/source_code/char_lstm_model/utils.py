# utils.py
# Description: Helper functions for processing text input
# Source: https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
from __future__ import print_function
import numpy as np

maxlen = 50
step = 1

def get_data(filename):
    print('Loading data...')
    text = open(filename).read().lower()
    chars = sorted(list(set(text)))
    char_indices = dict((c, i) for i, c in enumerate(chars))

    # cut the text in semi-redundant sequences of maxlen characters
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])

    X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1
    return X, y, chars


def get_sample_data(filename):
    print('Loading data...')
    text = open(filename).read().lower()
    chars = sorted(list(set(text)))

    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    # cut the text in semi-redundant sequences of maxlen characters
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])

    X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1
    return X, y, chars, char_indices, indices_char, text

