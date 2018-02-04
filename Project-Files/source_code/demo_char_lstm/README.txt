demo_char_lstm.py

A customizable text generating LSTM to train and generate text samples.
The program can be used with the default input text file of Shakespeare’s 
sonnets located in the data directory or with any text file specified in
the input_text argument.

How to run:

Run with defaults: 
	python demo_char_lstm.py

Run with optional arguments: 
	python demo_char_lstm.py —-epochs=5 —-file=model.h5 —-layers=2


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
--checkpoints: Interval for saving checkpoint files, type=int, default=1
--logs: Save log file for use with TensorBoard (0 is false, 1 is true), type=int, default=1)
--reduce_lr: Reduce learning rate on plateau (0 is false, 1 is true), type=int, default=0)


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




