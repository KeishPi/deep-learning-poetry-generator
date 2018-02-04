# train_BI_3_512_ND_adam.py
# Description: Testing file for baseline char level bidirectional lstm model.
#              Evaluates on test data, outputs testing loss and perplexity.
# Run as: test.py <path to trained model file>
from __future__ import print_function
import numpy as np
from keras.models import load_model
from sys import argv
from utils import get_data
import math

# get arguments and set variables
if len(argv) != 2:
    print("Error: Number of arguments")
    exit(0)

model_file = argv[1]
input_text = 'data/test_sonnets.txt'
batch_size = 64

# load data
X, y, chars = get_data(input_text)

# load model
model = load_model(model_file)

# evaluate model
print("Testing...")
loss = model.evaluate(X, y, batch_size=batch_size)
print()
print ("Test Loss", loss)

# calculate perplexity
perplexity = math.pow(2.0, loss)
print('Test Perplexity:', perplexity)


# save results to txt file
f = open('results/' + model_file + '_test.txt', 'w')
f.write('Testing Data\n')
f.write('-' * 50)
f.write('\n')
f.write('Testing Loss: ' + str(loss) + '\n')
f.write('Testing Perplexity: ' + str(perplexity) + '\n')
f.write('\n')
f.close()
