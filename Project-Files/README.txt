
This folder contains three Windows executable files: demo_char_lstm.exe,
generate_char_lstm.exe and generate_yoon_kim.exe. To run the programs, open 
Command Prompt in Windows and navigate to the folder that contains the 
executable files. The programs rely on information stored in the data directory
so they shouldn’t be moved from this location. Below are instructions for 
running each program.

We also created a website with more information about our work and a poem generator.
URL: http://ec2-18-217-70-169.us-east-2.compute.amazonaws.com/


Instructions For Running Executables

* Note: Each time a program runs it displays a short message about tensorflow CPU support. 
This is normal and can be ignored. 

Program 1: demo_char_lstm.exe

A customizable text generating LSTM to train and generate text samples.
Because our models are far too large to run on a regular CPU, we created 
this program to demonstrate how the training and tuning process works. 
The program can be used with the default input text file of Shakespeare’s 
sonnets or with any text file specified in the input_text argument. 

How to run:

Run with defaults in the Windows Command Prompt window type: demo_char_lstm

Run with optional arguments: demo_char_lstm —-epochs=5 —-file=model.h5 —-layers=2


Optional arguments:

--epochs: Number of epochs, type=int, default=1
--layers: Number of hidden layers, type=int, default=1
--nodes: Number of nodes per hidden layer, type=int, default=128
--batch_size: Number of streams processed at once, type=int, default=128
--seq_len: Length of each data stream, type=int, default=50
--optimizer: Optimizer: adam, sgd, rmsprop, type=str, default=adam
--model_type: Type of model: lstm, gru, bidirectional, type=str, default=lstm
--dropout: Dropout value between 0 and 1, type=float, default=0.0
--input_text: Path to input text, type=str, default=data/sonnets.txt
--sample_len: Length of generated text in chars, type=int, default=500
--diversity: Diversity of output between 0 and 1.5, type=float, default=0.5
--load: Path to load model from file, type=str, default=None
--save: Path to save model to file, type=str, default=None
--checkpoints: Interval for saving checkpoint files. They can be used
               to load the model from a specific epoch. type=int, default=1
--logs: Save log file for use with TensorBoard (0 is false, 1 is true), type=int, default=1
--reduce_lr: Reduce learning rate on plateau (0 is false, 1 is true), type=int, default=0


Suggested configurations:

The default arguments create a small model designed to be run on any CPU. As a 
result, the text output will not be very interesting. To create better results, 
test out different configurations using the optional arguments. For example:

- Increase the number of epochs
- Increase the number of hidden layers to 2 or 3
- Increase the number of nodes to 256 or 512
- Decrease batch_size to 64 or 32.
- Increase or decrease seq_len
- Save the model and reload it using —-save and —-load. This is helpful for 
  saving and resuming training progress, especially when using slower CPUs.


The source code can be found in source_code/demo_char_lstm/demo_char_lstm.py


Program 2: generate_char_lstm.exe

This program generates a sonnet from our trained baseline model. It typically generates
a 14 line poem but the model sometimes predicts an extra newline so length varies. The 
baseline model is a character level Bidirectional LSTM. Because of it’s large size, it can 
take up to a few minutes to generate the poem depending on the speed of the CPU.

How to run:

In the windows Command Prompt window type: generate_char_lstm

The source code can be found in source_code/generate_char_lstm/generate_char_lstm.py

The source code for our baseline model can be found in source_code/char_lstm_model


Program 3: generate_yoon_kim.exe

This program generates a sonnet from a Yoon Kim character aware word-level CNN-LSTM
model that we trained but did not write. It is included only as an example of what 
we hoped to achieve with the output of the gated-rlm TensorFlow model and to compare 
output with the char lstm model. It typically generates a 14 line poem but sometimes 
the model predicts an extra newline so length varies. Because of it’s large size, it 
can take up to a few minutes to generate the poem depending on the speed of the CPU.

How to run:

In the windows Command Prompt window type: generate_yoon_kim

The source code we used for the Yoon Kim character aware word-level CNN-LSTM model
can be found here: https://github.com/mkroutikov/tf-lstm-char-cnn

The source code for this executable is in source_code/generate_yoon_kim

The source code for our TensorFlow Gated Recurrent Language Model can be found in 
source_code/gated-rlm

