generate_char_lstm.py

This program generates a 14 line sonnet from our trained baseline model.
Because of it’s large size, it can take up to a few minutes to generate 
the poem depending on the speed of the CPU. 

How to run:

	python generate_char_lstm.py

The code is nearly identical to source_code/char_lstm_model/sample.py 
except that it does not write to text file. The other difference is that 
it uses separate weight and model files to load the trained model. This 
was done in order to keep the overall file size to a minimum. The large 
hdf5 model was converted into separate smaller json and h5 files using 
the convert.py script. The files necessary to run this program can 
be found in the root level data directory: model_char_lstm.json,
model_char_lstm.h5, and sonnets.txt.

To run convert.py: 
	
	python convert.py <original file name> <new file name>

This will create two new files from the original file. The new files 
will have the same name but one will be a .json file and the other
a .h5 file.