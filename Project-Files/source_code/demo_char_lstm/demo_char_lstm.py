#  demo_char_lstm.py
#  Description: A customizable text generating LSTM to train and generate text samples.
#  Adapted from: https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
from __future__ import print_function
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Bidirectional, GRU, Dropout, LSTM
import numpy as np
from random import randint
from sys import stdout, exit
import argparse
from keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint
import os


# Get arguments and set variables
parser=argparse.ArgumentParser()
parser.add_argument('--epochs', help='Number of epochs', type=int, default=1)
parser.add_argument('--layers', help='Number of hidden layers', type=int, default=1)
parser.add_argument('--nodes', help='Number of nodes per hidden layer', type=int, default=128)
parser.add_argument('--batch_size', help='Num streams processed at once', type=int, default=128)
parser.add_argument('--seq_len', help='Length of each data stream', type=int, default=50)
parser.add_argument('--optimizer', help='Optimizer: adam, sgd, rmsprop', type=str, default='adam')
parser.add_argument('--model_type', help='Type of model: lstm, gru, bidirectional', type=str, default='lstm')
parser.add_argument('--dropout', help='Dropout value between 0 and 1', type=float, default=0.0)
parser.add_argument('--input_text', help='Path to input text', type=str, default='data/sonnets.txt')
parser.add_argument('--sample_len', help='Length of generated text in chars', type=int, default=500)
parser.add_argument('--diversity', help='Diversity of output results between 0 and 1.5', type=float, default=0.5)
parser.add_argument('--load', help='Path to load model from file', type=str, default=None)
parser.add_argument('--save', help='Path to save model to file', type=str, default=None)
parser.add_argument('--checkpoints', help='Interval for saving checkpoint files', type=int, default=1)
parser.add_argument('--logs', help='Save log file for use with TensorBoard (0 is false, 1 is true)', type=int, default=1)
parser.add_argument('--reduce_lr', help='Reduce learning rate on plateau (0 is false, 1 is true)', type=int, default=0)
args=parser.parse_args()

if args.optimizer != 'adam' and args.optimizer != 'sgd' and args.optimizer != 'rmsprop':
    print('Error in optimizer value. Must be adam, sgd, or rmsprop.')
    exit(0)

if args.model_type != 'lstm' and args.model_type != 'gru' and args.model_type != 'bidirectional':
    print('Error in model_type value. Must be lstm, gru, or bidirectional.')
    exit(0)

if args.dropout < 0 or args.dropout > 1:
    print('Error in dropout value. Dropout must be between 0 and 1.')
    exit(0)

input_text = args.input_text   # Path to input text
seq_len = args.seq_len         # Length of each data stream
batch_size = args.batch_size   # Num streams processed at once
model_type = args.model_type   # Type of model: lstm, gru, bidirectional
optimizer =args.optimizer      # Optimizer: adam, sgd, rmsprop
dropout = args.dropout         # Dropout value between 0 and 1 (0 is none)
epochs = args.epochs           # Number of epochs
layers = args.layers           # Number of hidden layers
nodes = args.nodes             # Number of nodes per hidden layer
diversity = args.diversity     # Diversity of output results between 0 and 1.5
sample_len = args.sample_len   # Length of generated text in chars
load = args.load               # Path to load model from file
save = args.save               # Path to save model to file
checkpoints = args.checkpoints # Interval for saving checkpoint files (0 is none)
logs = args.logs               # Save log file for use with TensorBoard (0 is false, 1 is true)
reduce_lr = args.reduce_lr     # Reduce learning rate on plateau (0 is false, 1 is true)


# Load text from file
print('Loading text from file...')
text = open(input_text).read().lower()
chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# create text sequences of seq_len length
step = 1
sentences = []
next_chars = []
for i in range(0, len(text) - seq_len, step):
    sentences.append(text[i: i + seq_len])
    next_chars.append(text[i + seq_len])

# create char vectors for training
x_train = np.zeros((len(sentences), seq_len, len(chars)), dtype=np.bool)
y_train = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x_train[i, t, char_indices[char]] = 1
    y_train[i, char_indices[next_chars[i]]] = 1


# Create new model or load existing model from file
if load:
    print("Loading model from file...")
    model = load_model(load)
else:
    print("Building model...")
    model = Sequential()
    if layers > 1:
        for i in range(layers - 1):
            if model_type == 'lstm':
                model.add(LSTM(nodes,
                               return_sequences=True,
                               input_shape=(seq_len, len(chars))))
                print('LSTM layer added...')
            elif model_type == 'gru':
                model.add(GRU(nodes,
                              return_sequences=True,
                              input_shape=(seq_len, len(chars))))
                print('GRU layer added...')
            elif model_type == 'bidirectional':
                model.add(Bidirectional(LSTM(nodes,
                                             return_sequences=True),
                                             input_shape=(seq_len, len(chars))))
                print('Bidirectional layer added...')

            if dropout:
                model.add(Dropout(dropout))
                print('Dropout added...')

    if model_type == 'lstm':
        model.add(LSTM(nodes, input_shape=(seq_len, len(chars))))
        print('LSTM layer added...')
    elif model_type == 'gru':
        model.add(GRU(nodes, input_shape=(seq_len, len(chars))))
        print('GRU layer added...')
    elif model_type == 'bidirectional':
        model.add(Bidirectional(LSTM(nodes), input_shape=(seq_len, len(chars))))
        print('Bidirectional layer added...')

    if dropout:
        model.add(Dropout(dropout))
        print('Dropout added...')

    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    print('Optimizer ' + optimizer + ' added...')


# Prepare callbacks
callbacks = []

# tensorboard logs:
if logs:
    if not os.path.exists('logs'):
        os.makedirs('logs')
    tb = TensorBoard(log_dir='./logs',
                     histogram_freq=0,
                     batch_size=batch_size,
                     write_graph=True,
                     write_grads=False,
                     write_images=False,
                     embeddings_freq=0,
                     embeddings_layer_names=None,
                     embeddings_metadata=None)
    callbacks.append(tb)
    print('Adding TensorBoard logs...')

# save checkpoint file after period (number of epochs)
if checkpoints:
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    checkpoint_path = 'checkpoints/' + model_type + '_' + str(layers) + '_' \
                       + str(nodes) + '_' + '{epoch:02d}-{loss:.4f}.hdf5'
    mc = ModelCheckpoint(checkpoint_path,
                         monitor='loss',
                         verbose=0,
                         save_best_only=True,
                         save_weights_only=False,
                         mode='auto',
                         period=checkpoints)
    callbacks.append(mc)
    print('Adding checkpoints...')

# reduce learning rate on plateau
if reduce_lr:
    lr = ReduceLROnPlateau(monitor='loss',
                                  factor=0.2,
                                  patience=5,
                                  min_lr=0.001)
    callbacks.append(lr)
    print('Adding learning rate plateau...')


# Train model
print("Training...")
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks)


# Generate text from trained model
print("Generating text...")

# helper function to sample an index from a probability array
def sample(preds, temperature):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# generate starting text for sample data
start_index = randint(0, len(text) - seq_len - 1)
generated = ''
sentence = text[start_index: start_index + seq_len]
generated += sentence
print('Using seed text: "' + sentence + '"')
stdout.write(generated)

# generate sample data
for i in range(sample_len):
    x = np.zeros((1, seq_len, len(chars)))
    for t, char in enumerate(sentence):
        x[0, t, char_indices[char]] = 1.

    preds = model.predict(x, verbose=0)[0]
    next_index = sample(preds, diversity)
    next_char = indices_char[next_index]
    generated += next_char
    sentence = sentence[1:] + next_char
    stdout.write(next_char)
    stdout.flush()
print()


# Save trained model to file
if save:
    model.save(save + '.h5')
    print('Model saved to file: ' + save + '.h5')

