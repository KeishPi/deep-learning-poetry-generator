# train_BI_3_512_ND_adam.py
# Description: training file for baseline char level bidirectional lstm model
# Run as: python train_BI_3_512_ND_adam.py
from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Bidirectional, GRU, Dropout, LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras.models import load_model
from keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint
import numpy as np
from sys import argv
from utils import get_data
import math


model_type = 'Bidirectional'
layers = 3
size = 512
epochs = 30

maxlen = 50
batch_size = 64
input_file = 'data/train_sonnets.txt' # training file
val_input = 'data/val_sonnets.txt' #validation data
drop = 0  # dropout - 0 is none
reduce_lr_on_plateau = False  # reduce learning rate on plateau
learning_rate = 0.01
optimizer = 'adam' # sgd, adagrad, RMSprop, adam, etc.
save_file = 'Bi_3_512_ND_adam' # filename for saved model


# get data
X, y, chars = get_data(input_file)
X_val, y_val, junk = get_data(val_input)


# build model
model = Sequential()
model.add(Bidirectional(LSTM(size, return_sequences=True), input_shape=(maxlen, len(chars))))
model.add(Bidirectional(LSTM(size, return_sequences=True), input_shape=(maxlen, len(chars))))
model.add(Bidirectional(LSTM(size), input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


# train model
print('Training...')
print('-' * 50)

# prepare tensorboard:
tb = TensorBoard(log_dir='./logs',
                 histogram_freq=0,
                 batch_size=batch_size,
                 write_graph=True,
                 write_grads=False,
                 write_images=False,
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None)


# save checkpoint file after period (number of epochs)
checkpoint_path = 'checkpoints/' + save_file + '_' + '{epoch:02d}-{loss:.4f}.hdf5'
mc = ModelCheckpoint(checkpoint_path,
                     monitor='loss',
                     verbose=0,
                     save_best_only=True,
                     save_weights_only=False,
                     mode='auto',
                     period=10)


if reduce_lr_on_plateau:
    # reduce learning rate on plateau
    reduce_lr = ReduceLROnPlateau(monitor='loss',
                                  factor=0.2,
                                  patience=5,
                                  min_lr=0.001)

    hist = model.fit(X, y,
                     batch_size=batch_size,
                     epochs=epochs,
                     validation_data=(X_val, y_val),
                     callbacks=[reduce_lr, tb, mc])
else:
    # run without reduce learning rate on plateau
    hist = model.fit(X, y,
                     batch_size=batch_size,
                     epochs=epochs,
                     validation_data=(X_val, y_val),
                     callbacks=[tb, mc])


# calculate perplexity - NOT SURE IF THIS IS CORRECT
train_loss = hist.history['loss'][epochs-1]
val_loss = hist.history['val_loss'][epochs-1]
perplexity = math.pow(2.0, train_loss)
print('Train Perplexity:', perplexity)


# save trained model
model.save('models/' + save_file + '.h5')

# save model details and results to txt file
f = open('results/' + save_file + '.txt', 'w')
f.write('Training Data\n')
f.write('-' * 50)
f.write('\n')
f.write('Model Type: ' + model_type + '\n')
f.write('Layers: ' + str(layers) + '\n')
f.write('Layer Size: ' + str(size) + '\n')
f.write('Epochs: ' + str(epochs) + '\n')
f.write('Batch Size: ' + str(batch_size) + '\n')
f.write('Dropout: ' + str(drop) + '\n')
f.write('Learning Rate: ' + str(learning_rate) + '\n')
f.write('Reduce Learning Rate On Plateau: ' + str(reduce_lr_on_plateau) + '\n')
f.write('Optimizer: ' + str(optimizer) + '\n')
f.write('Training Loss: ' + str(train_loss) + '\n')
f.write('Validation Loss: ' + str(val_loss) + '\n')
f.write('Training Perplexity: ' + str(perplexity) + '\n')
f.write('Model Filename: ' + save_file + '.h5' + '\n')
f.write('\n')
f.close()
