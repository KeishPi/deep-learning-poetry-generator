generate_yoon_kim.py

This program generates a sonnet from a Yoon Kim model that we trained but did not write.
It is included only as an example of what we hoped to achieve with the output of the 
gated-rlm Tensorflow model and to compare output with the char lstm model. Because of it’s 
large size, it can take up to a few minutes to generate the poem depending on the speed of 
the CPU.

How to run: python generate_yoon_kim.py

The source code we used for the Yoon Kim character aware word-level CNN-LSTM model
can be found here: https://github.com/mkroutikov/tf-lstm-char-cnn

In order to create the executable we made the following changes to the source code:
- changed the seed to a random word so the poem would be different each time
- changed the output format to only write single spaces
- changed the output to run for around 14 lines (sometimes it predicts extra newlines so 
length varies)
- combined several functions into one file including functions from generate.py,
model.py and data_reader.py from the original source code.

