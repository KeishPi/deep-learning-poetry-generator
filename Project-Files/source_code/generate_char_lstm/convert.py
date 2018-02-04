# convert.py
# Description: converts large hdf5 model files to smaller weight and architecture files
from __future__ import print_function
import numpy as np
import random
from keras.models import load_model
import sys
from keras.models import model_from_json
from sys import argv

model_file = argv[1]
new_model_file = argv[2]
model = load_model(model_file)

# save model to json
model_json = model.to_json()
with open(new_model_file + '.json', "w") as json_file:
    json_file.write(model_json)

# save weights to h5
model.save_weights(new_model_file + '.h5')
print("Saved model")





