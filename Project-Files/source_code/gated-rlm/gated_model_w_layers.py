
#  Gated_LSTM_model.py
#
#
#  Created by Jake K on 11/1/17.
#   Based on Gated RLM by Miyamoto and Cho
#       ==> The model description is here: https://arxiv.org/abs/1606.01700
#       ==> The base code is here: https://github.com/nyu-dl/dl4mt-tutorial


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from datetime import datetime
import logging
import sys #Progbar
import os
from os.path import join as pjoin


import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq

import numpy as np

from base import Model, Progressbar
from data_reader import DataReader
from wordChar_prep import *
from data_preprocess import *
from layers import     bidirectional_lstm, GloVe, gate, lstm_lm

import random
random.seed(12345)

""" gated word-char LSTM ""

    actual longest token length is: 20
    size of word vocabulary: 4663
    size of char vocabulary: 60
    number of tokens in train: 15912
    number of tokens in valid: 1939
    number of tokens in test: 1956

    structure:

                    word -> [[biLSTM] + [GloVe]] -> gate -> [LSTM-LM] -> [SoftMax] -> word++

    mini-bach size = 32 (w SGD)

    word dimenstionality: [32, 200, 200]
    word vocab: 10000 (them), 4663 (us)
    char dimensionality: [32, 200, 200]
    char vocab: 51 (them) 60 (us)

    LSTM dimensions: n_hidden = 200,



    [bi-LSTM_fwd]  # ex: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/bidirectional_rnn.py
    [bi-LSTM_bkwd] # doc: https://www.tensorflow.org/api_docs/python/tf/nn/bidirectional_dynamic_rnn
    [fully connected layer]
    {run char LSTM}
        char_outputs, char_lstm_state = tf.nn.dynamic_rnn(nn_var, input_embedding, sequence_length=source_sequence_length, time_major=True)

    [GloVe Lookup]  # https://www.tensorflow.org/programmers_guide/embedding
                    # https://www.tensorflow.org/tutorials/word2vec
                    # https://www.tensorflow.org/api_docs/python/tf/nn/
    { compute word lookup }


    [   gate   ] # hard way: https://www.tensorflow.org/extend/adding_an_op
                 # easy way? unroll V & b and implement a python function?
                 # something like this highway network:
                    https://medium.com/jim-fleming/highway-networks-with-tensorflow-1e6dfa667daa
    {get Xwt using the output of the char and word embeddings. then pass Xwt to the LSTM}

    [LSTM layer 1 ] # https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/LSTMCell
    [LSTM layer 2 ] # stack w https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/MultiRNNCell
    [fully connected layer] # https://www.tensorflow.org/api_docs/python/tf/contrib/layers/fully_connected

    [ SoftMax (w dropout)]

    [ Attention ] # https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/AttentionCellWrapper

        tf seq2seq tutorial:
            https://github.com/tensorflow/nmt

    hepful models:
        https://github.com/spiglerg/RNN_Text_Generation_Tensorflow/blob/master/rnn_tf.py

        https://github.com/carpedm20/lstm-char-cnn-tensorflow/blob/master/models/LSTMTDNN.py

        stanford QA (encoder decoder w attention)
        https://web.stanford.edu/class/cs224n/reports/2761057.pdf
        https://gitlab.com/ksebov/su-cs224n/tree/master/assignment4

    resources
        https://r2rt.com/recurrent-neural-networks-in-tensorflow-ii.html ( i to iii )

    """
#tf.set_random_seed(seed)

class Gated_RNN_LM(object): #Model):   #MAXLEN COMES FROM
    def __init__(self, sess,
        # input data
        word_vocab, char_vocab, word_tensors, char_tensors, pretrained_embeddings, max_word_length,
        #hyperparameters
        n_char=51, n_words=10000,           #vocab size
        dim_char=200, dim_word=200,         #tensor dimensionality (seq len)
        bi_lstm_size=200, lstm_lm_size=200, #nn dimensionality
        maxlen=None, gradclip=5., use_dropout=False,
        #optimization
        optimizer='sgd', batch_size=32,
        lrate=1, lr_decay=2.1, lr_start=7,
        patience=3, pretrain=2,
        #display/save
        max_epochs=100, reload_=False, #not sure about reload :/
        dispFreq=100, saveFreq=1000, validFreq=1000,
        dropout_prob=0.5,
        #model paths
          #saveto=
          #savebestto=
        #data paths
        word_dictionary='./data/word_dict.pkl', char_dictionary='./data/char_dict.pkl',
        train_text='./data/train.txt', valid_text='./data/eval.txt', test_text='./data/test.txt',

        #might use form github model
        hsm=0, max_grad_norm=5,
        use_batch_norm=True,
        checkpoint_dir="checkpoint", forward_only=False,
        data_dir="data", dataset_name="pdb", use_progressbar=False):
        """
        Initialize the parameters for gated rnlm
        Args:
        {From Miyamoto and Cho}
        ### model hyperparameters ###
        model_type: gate
        n_char:  51                             # num of unique characters (dict size + 1) {55 I think for shakespeare}
        n_words: 10000                          # vocabulary size {not this for shakespeare}
        dim_char: 200                           # char vector dimensionality (becomes 400 after concatenating fw/bw output)
        dim_word: 200                           # word vector dimensionality
        bi_lstm_size (was dim_bi_lstm): 200     # num of hidden units in the bidirectional LSTMs
        lstm_lm_size (was dim_lm_lstm): 200     # num of hidden units in the language modeling part
        maxlen: None                            # maximum length (char-based) of the sentence
        gradclip: 5.                            # gradient clipping
        use_dropout: False                      # dropout: True or False
        bos: "|"                                # the BOS symbol

        ### optimization ###
        optimizer: sgd
        batch_size: 32
        lrate: 1.                               # learning rate
        lr_decay: 2.1                           # decay factor, must be float
        lr_decay_start: 7                       # decay starts after this epoch
        patience: 3                             # early stopping
        pretrain: 2                             # the first m epochs: word only,
                                                    # the next m epochs: char only
                                                    # set 0 to disable

        ### display / save ###
        max_epochs: 100
        dispFreq: 100                           # display NLL after every x updates
        saveFreq: 1000                          # save params after every y updates
        validFreq: 1000                         # perform validation afer every z updates
        reload_: False

        ### model paths ###
        saveto: gate_word_char_p_shkspr.npz             # the newest model
        savebestto: gate_word_char_p_shkspr_best.npz    # the best model


        dropout_prob: the probability of dropout
        use_batch_norm: whether to use batch normalization or not
        hsm: whether to use hierarchical softmax
        """
        self.sess = sess

        self.batch_size = batch_size
        self.seq_length = seq_length = dim_char
        #self.maxlen = maxlen
        self.maxlen = max_word_length
        self.bos="|"

        #vocab & GloVe
        self.words = word_vocab
        self.chars = char_vocab
        self.pretrained_embeddings = pretrained_embeddings

        # Bidirectional Char-LSTM
        self.dim_char = dim_char
        #self.n_char = n_char
        self.n_char = len(char_vocab)
        self.bi_lstm_size = bi_lstm_size

        # GloVe Lookup table
        self.dim_word = dim_word
        #self.n_words = n_words
        self.n_words = len(word_vocab)
        print("num words is ".format(self.n_words))

        # LSTM Language Model
        self.lstm_lm_size = lstm_lm_size
        self.use_dropout = use_dropout
        self.dropout_prob = dropout_prob
        self.use_batch_norm = use_batch_norm

        # Training
        self.gradclip = gradclip
        self.lr = lrate
        self.lr_decay  = lr_decay
        self.lr_start = lr_start
        self.patience = patience
        self.pretrain = pretrain # used in gate functon calculation of X_wt
        """={ DOUBLE AND TRIPPLE CHECK EVERYTHING BELLOW THIS!!! }="""
        #self.max_grad_norm = max_grad_norm
        #self.max_word_length = max_word_length
        #self.hsm = hsm

        # dir paths
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir

        #self.forward_only = forward_only
        self.use_progressbar = use_progressbar
        """
                self.loader = BatchLoader(self.data_dir, self.dataset_name, self.batch_size, self.seq_length, self.max_word_length)
                print('Word vocab size: %d, Char vocab size: %d, Max word length (incl. padding): %d' % \
                (len(self.loader.idx2word), len(self.loader.idx2char), self.loader.max_word_length))

                self.max_word_length = self.loader.max_word_length
                self.char_vocab_size = len(self.loader.idx2char)
                self.word_vocab_size = len(self.loader.idx2word)
        """
        # build Gated Recurrent Neural Network Language Model
        #self.build_model()


        # load checkpoints
        """
        if self.forward_only == True:
            if self.load(self.checkpoint_dir, self.dataset_name):
                print("[*] SUCCESS to load model for %s." % self.dataset_name)
            else:
                print("[!] Failed to load model for %s." % self.dataset_name)
                sys.exit(1)
        """


    def build_model(self):
    #with tf.variable_scope("Gated_RNN-LM"):
        print ("building model...")
        print ("word vocab: ")
        print (self.words)
        print ("char vocab: ")
        print (self.chars)

        # ==== set up placeholder tokens ========
        #char in
        self.X_char = tf.placeholder(tf.float32, [self.batch_size, self.maxlen, self.dim_char], "X_char") #[batch, seq, max_word_len]
        print ("char_input dims: ", self.X_char.get_shape().as_list()) #[32, None, 200]

        #word in
        self.X_word = tf.placeholder(tf.int64, [self.batch_size, self.maxlen], "X_word") #[batch, seq, max_word_len]
        #self.one_hot_word=tf.one_hot(self.word_input, self.n_words)
        print ("word_input dims: ", self.X_word.get_shape().as_list()) #[32, None, 200]

        self.seq_len_placeholder = tf.placeholder(tf.int64, [self.maxlen], "seq_len_placeholder")
        self.word_labels_placeholder = tf.placeholder(dtype=tf.int32, shape=[None,2], name='word_labels')

        """
        self.char_input = tf.placeholder(dtype=tf.int64, name="X_char")
        self.n_timesteps = 201 #self.char_input.get_shape()[1] #bc tf is batch major (not time-major) by default
        self.n_samples = 36 #self.char_input.get_shape()[0]
        #print ("char_input dims: ", self.char_input.get_shape().as_list())

        self.word_input = tf.placeholder(dtype=tf.int64, name="X_word")
        #print ("word_input dims: ", self.word_input.get_shape().as_list()) #[32, None, 200]
        """
        """can I use ints or must they be shared tensor variables? (don't forget to do reuse=T in scope!)"""
        self.is_train = tf.Variable(np.float32(0.))        # why
        self.pretrain_mode = tf.Variable(np.float32(0.)) # line 348 of word_char_lm.py

        #self.label_words = tf.placeholder(tf.int64, [self.batch_size, self.sequence_length, self.dim_word], "Y_hat")


        #char_W = tf.get_variable("char_embed", [self.n_char, self.dim_char]) #[self.char_vocab_size, self.char_embed_dim])
        #print ("char_W dims: ", char_W.get_shape().as_list()) #[51, 200]
        #word_W = tf.get_variable("word_embed", [self.n_words, self.dim_word])
        #print ("word_W dims: ", word_W.get_shape().as_list()) #[10,000, 200]

        ###################
        """ {char LSTM} """ #https://github.com/tensorflow/tensorflow/issues/799
        ###################
        #char_batch_vector = tf.placeholder(tf.int64, [self.dim_char], "batch_vector")
        print ("char_batch_vector dims: ", self.seq_len_placeholder.get_shape().as_list()) #[10,000, 200]
        self.Cemb = bidirectional_lstm(self=self, char_dict=self.chars, batch_tensor=self.seq_len_placeholder, n_hidden=200, X_char=self.X_char)

        ######################
        """ {Word Look Up} """
        #####################
        self.Wemb = GloVe(self=self, X_word=self.X_word, pretrained_embeds=self.pretrained_embeddings )

        ##############
        """{ GATE }"""
        ##############
        #try simply concatenating Wemb and Cemb first
        self.X_wt = gate(self=self, word_emb=self.Wemb, char_emb=self.Cemb, pretrain_mode=self.pretrain_mode)

        ##################
        """ {LSTM LM } """
        ##################
        self.final_output, self.final_state = lstm_lm(self=self, input_X=self.X_wt, n_hidden=self.lstm_lm_size, num_layers=2)
        #test crap        self.word = lstm_lm(self=self, lm_input=self.Wemb, n_hidden=self.lstm_lm_size, num_layers=2)

        ###################
        """ { Softmax } """
        ###################
        with tf.variable_scope('softmax') as scope:
            W = tf.get_variable('W', [state_size, num_classes])
            b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
        self.logits = tf.reshape(tf.matmul(tf.reshape(self.final_output, [-1, state_size]), W) + b, [batch_size, num_steps, num_classes])
        self.predictions = tf.nn.softmax(self.logits)

        #########################
        """ { Optimization } """
        ########################
        # TODO: word_labels!
        """
        self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.word_labels, logits=self.logits)
        self.total_loss = tf.reduce_mean(self.losses)

        # SGD #https://www.tensorflow.org/api_docs/python/tf/train/GradientDescentOptimizer
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss)

        # Adagrad #https://www.tensorflow.org/versions/r1.1/api_docs/python/tf/train/AdagradOptimizer
        #self.train_step = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.total_loss)

        # Adam #https://www.tensorflow.org/versions/r1.1/api_docs/python/tf/train/AdamOptimizer
        #self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss) #add lr decay?
        """

        """ [To Do from M & C] """
        # mask for final loss: 1 if the last char of a word, 0 otherwise
        final_loss_mask = 1 - x_last_chars
        final_loss_mask = final_loss_mask.flatten()

        # cost
        x_flat = label_words.flatten()
        x_flat_idx = tensor.arange(x_flat.shape[0]) * options['n_words'] + x_flat
        cost = -tensor.log(probs.flatten()[x_flat_idx] + 1e-8) * final_loss_mask # only last chars of words
        cost = cost.reshape([x_f.shape[0], x_f.shape[1]]) # reshape to n_steps x n_samples
        cost = cost.sum(0)                                # sum up NLL of words in a sentence
        cost = cost.mean()                                # take mean of sentences
        """ [END To Do from M & C] """

def main(_):
    """ move word/char preprocessing to preprocess or other function! """

    """ Tokenize inputs!
    --------------------"""
    #corpus_tokenizer('../data/test.txt', '../data/test_tokenized.txt' )
    #corpus_tokenizer('../data/train.txt', '../data/train_tokenized.txt' )
    #corpus_tokenizer('../data/valid.txt', '../data/valid_tokenized.txt' )

    """ GloVE preprocessing
    (ensure glove embdeddings exist)
    -------------------------------------"""
    #trim_GloVe(create_vocab=False)


    """ Word-Char CNN-LSTM preprocess (Yoon Kim)
    ---------------------------------"
    word_vocab, char_vocab, word_tensors, char_tensors, max_word_length = \
        load_data(data_dir="../data", max_word_length=65, eos='+')
    word_vocab.save('word_vocab.pkl')
    char_vocab.save('char_vocab.pkl')"""


    """ Gated RLM preprocess
    -------------------------"""
    with open( '../data/MnC_dicts/char_dict.pkl' , 'rb') as chars:
        char_dict = pkl.load(chars)
    with open( '../data/MnC_dicts/word_dict.pkl' , 'rb') as words:
        word_dict = pkl.load(words)
    input_txt = load_file('../data/train_tokenized.txt')

    X_char, X_char_trash, X_mask, spaces, last_chars = prepare_char_data(text_to_char_index(input_txt, char_dict, '|'), text_to_char_index(input_txt, char_dict, '|'))
    X_word, x_mask = prepare_word_data(text_to_word_index(input_txt, word_dict))
    """
    print ('X_char: ')
    print (X_char)
    print ('X_word: ')
    print (X_word)
    print ('X_char_trash: ')
    print (X_char_trash)
    print ('X_mask: ')
    print (X_mask)
    print ('spaces: ')
    print (spaces)
    print ('last_chars: ')
    print (last_chars)
    print ('x_mask: ')
    print (x_mask)
    """
    embed_path = "../data/GloVe_vectors.trimmed.200d.npz"
    embeddingz = np.load(embed_path)
    embeddings = embeddingz['glove']
    embeddingz.close()
    assert embeddings.shape[1] == 200 #(embedding size)

    vocab_len = embeddings.shape[0]
    print ("word vocab from embeddings shape[0] is {}".format(vocab_len))

    with tf.Session() as sess:
        x = tf.placeholder(tf.int64, shape=[None, 200])
        model = Gated_RNN_LM(sess, word_dict, char_dict, pretrained_embeddings=embeddings, word_tensors=X_word, char_tensors=X_char, max_word_length=20 )
        sess.run(tf.global_variables_initializer())
#        x_f_, x_r_, x_spaces_, x_last_chars_, x_word_input_, label_words_ \
#                                                      = txt_to_inps(x, char_dict, word_dict, opts=[{'bos': 155}, {'maxlen': 200}, {'n_char': 65}, {'n_words': 7000}])
        model.build_model()
    """
    if not FLAGS.forward_only:
        model.run(FLAGS.epoch, FLAGS.learning_rate, FLAGS.decay)
    else:
        test_loss = model.test(2)
        print(" [*] Test loss: %2.6f, perplexity: %2.6f" % (test_loss, np.exp(test_loss)))
    """

if __name__ == '__main__':
  tf.app.run()
