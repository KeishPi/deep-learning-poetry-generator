# generate_char_lstm.py
# Description: Generates 14 lines of text from a trained model
# Adapted from: https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
from __future__ import print_function
import numpy as np
import random
import sys
from keras.models import model_from_json
import h5py


filename = 'data/sonnets.txt'
maxlen = 50
step = 1
diversity = 1.0

# get text data
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

# load json and create model
json_file = open('data/model_char_lstm.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights("data/model_char_lstm.h5")

print('Generating poem...this could take a few minutes')
# helper function to sample an index from a probability array
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# generate seed for sample data
start_index = random.randint(0, len(text) - maxlen - 1)
generated = ''
sentence = text[start_index: start_index + maxlen]
generated += sentence
next_char = ''
print()

# generate remaining chars in seed line
while next_char != '\n':
    x = np.zeros((1, maxlen, len(chars)))
    for t, char in enumerate(sentence):
        x[0, t, char_indices[char]] = 1.

    preds = model.predict(x, verbose=0)[0]
    next_index = sample(preds, diversity)
    next_char = indices_char[next_index]

    generated += next_char
    sentence = sentence[1:] + next_char

# write 14 lines of text
lines = 0
while lines < 14:
    x = np.zeros((1, maxlen, len(chars)))
    for t, char in enumerate(sentence):
        x[0, t, char_indices[char]] = 1.

    preds = model.predict(x, verbose=0)[0]
    next_index = sample(preds, diversity)
    next_char = indices_char[next_index]

    generated += next_char
    sentence = sentence[1:] + next_char
    if next_char == '\n':
        lines += 1

    sys.stdout.write(next_char)
    sys.stdout.flush()

print()

