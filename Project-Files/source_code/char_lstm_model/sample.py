# sample.py
# Description: Generates 14 lines of text from a trained model
# Run as: sample.py <path to trained model file>
# Adapted from: https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
from __future__ import print_function
import numpy as np
import random
from keras.models import load_model
import sys
from sys import argv
from utils import get_sample_data


# get arguments
if len(argv) != 2:
    print("Error: Number of arguments")
    exit(0)

model_file = argv[1]
input_text = 'data/sonnets.txt'
maxlen = 50
diversity = 1.0
# get text data
X, y, chars, char_indices, indices_char, text = get_sample_data(input_text)

# load model
model = load_model(model_file)

# save results to txt file
f = open('results/' + model_file + '_sample.txt', 'w')
f.write('Sample Data\n')
f.write('-' * 50)
f.write('\n')

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
    f.write(next_char)
    sys.stdout.flush()

print()

f.write('\n')
f.close()

