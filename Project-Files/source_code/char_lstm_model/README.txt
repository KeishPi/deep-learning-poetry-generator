
char_lstm_model

This folder contains all files needed to train, test, and sample
from our char-level bidirectional lstm model. Some supplementary 
files were not included because of their large file size. An 
explanation of each file in this directory is listed below. 


train_BI_3_512_ND_adam.py:

This is the training file for the basic model we created during our initial 
phase of testing. It trains a 3 layer Bidirectional LSTM with 512 nodes 
per layer, using the Adam optimizer. It uses a batch size of 64 and a
sequence length of 50. The training text file path is hardcoded as
data/train_sonnets.txt. The validation text path is hardcoded as 
data/val_sonnets.txt. Because of the model’s large size it’s not recommended
to run the training on a CPU.

How to run:

	python train_BI_3_512_ND_adam.py

After running, this will output a few files. The first file is a .h5 file of 
the trained model. It's saved in the 'models' directory. The second file is 
a text file in the 'results' directory and includes a list of all training 
variables and results. Both files are named with the model type and a timestamp 
(ex: LSTM_1234.txt and LSTM_1234.h5). 

It will also output a file for use with TensorBoard to the ‘logs’ directory. 
To use TensorBoard with the log file run: tensorboard --logdir=logs then go to
localhost:6006 in a browser.

It will also save checkpoint files to the ‘checkpoints’ directory. Each 
checkpoint file has the epoch number and loss value appended to it. These files
can be used to load the model as it existed at that point in time.


test.py

After the model is trained run test.py on the test text. The test text file path
is hardcoded as data/test_sonnets.txt. The results for loss and test perplexity 
are saved to the results directory in a text file with the model name and _test 
appended to it.

How to run: 

    python test.py <path to trained model>


sample.py

This program generates a 14 line sonnet from a trained model. The output is 
saved to the results directory in a text file with the model name and _sample
appended to it. Because of it’s large size, it can take up to a few minutes to 
generate the poem depending on the speed of the CPU. 

How to run: 

    python sample.py <path to trained model>


utils.py

This file contains helper functions to process text from file. 
