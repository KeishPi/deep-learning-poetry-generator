# Source: https://github.com/mkroutikov/tf-lstm-char-cnn
# Note: Code altered to reformat generated text
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


import tensorflow as tf
import tensorflow.contrib #
import os
import time
import random
import codecs
import collections
import numpy as np


flags = tf.flags

# data
flags.DEFINE_string('load_model',   'data/epoch024_6.1626.model',    'filename of the model to load')
# we need data only to compute vocabulary
flags.DEFINE_string('data_dir',   'data',    'data directory')
flags.DEFINE_integer('num_samples', 300, 'how many words to generate')
flags.DEFINE_float('temperature', 1.0, 'sampling temperature')

# model params
flags.DEFINE_integer('rnn_size',        650,                            'size of LSTM internal state')
flags.DEFINE_integer('highway_layers',  2,                              'number of highway layers')
flags.DEFINE_integer('char_embed_size', 15,                             'dimensionality of character embeddings')
flags.DEFINE_string ('kernels',         '[1,2,3,4,5,6,7]',              'CNN kernel widths')
flags.DEFINE_string ('kernel_features', '[50,100,150,200,200,200,200]', 'number of features in the CNN kernel')
flags.DEFINE_integer('rnn_layers',      2,                              'number of layers in the LSTM')
flags.DEFINE_float  ('dropout',         0.5,                            'dropout. 0 = no dropout')

# optimization
flags.DEFINE_integer('max_word_length',     65,   'maximum word length')

# bookkeeping
flags.DEFINE_integer('seed',           3435, 'random number generator seed')
flags.DEFINE_string ('EOS',            '+',  '<EOS> symbol. should be a single unused character (like +) for PTB and blank for others')

FLAGS = flags.FLAGS





class Vocab:
    
    def __init__(self, token2index=None, index2token=None):
        self._token2index = token2index or {}
        self._index2token = index2token or []
    
    def feed(self, token):
        if token not in self._token2index:
            # allocate new index for this token
            index = len(self._token2index)
            self._token2index[token] = index
            self._index2token.append(token)
        
        return self._token2index[token]
    
    @property
    def size(self):
        return len(self._token2index)
    
    def token(self, index):
        return self._index2token[index]
    
    def __getitem__(self, token):
        index = self.get(token)
        if index is None:
            raise KeyError(token)
        return index
    
    def get(self, token, default=None):
        return self._token2index.get(token, default)
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self._token2index, self._index2token), f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            token2index, index2token = pickle.load(f)
        
        return cls(token2index, index2token)


def load_data(data_dir, max_word_length, eos='+'):
    
    char_vocab = Vocab()
    char_vocab.feed(' ')  # blank is at index 0 in char vocab
    char_vocab.feed('{')  # start is at index 1 in char vocab
    char_vocab.feed('}')  # end   is at index 2 in char vocab
    
    word_vocab = Vocab()
    word_vocab.feed('|')  # <unk> is at index 0 in word vocab
    
    actual_max_word_length = 0
    
    word_tokens = collections.defaultdict(list)
    char_tokens = collections.defaultdict(list)
    
    for fname in ('train', 'valid', 'test'):
        #print('reading', fname)
        with codecs.open(os.path.join(data_dir, fname + '.txt'), 'r', 'utf-8') as f:
            for line in f:
                line = line.strip()
                line = line.replace('}', '').replace('{', '').replace('|', '')
                line = line.replace('<unk>', ' | ')
                if eos:
                    line = line.replace(eos, '')
            
                for word in line.split():
                    if len(word) > max_word_length - 2:  # space for 'start' and 'end' chars
                        word = word[:max_word_length-2]
                
                    word_tokens[fname].append(word_vocab.feed(word))
                    
                    char_array = [char_vocab.feed(c) for c in '{' + word + '}']
                    char_tokens[fname].append(char_array)
        
                    actual_max_word_length = max(actual_max_word_length, len(char_array))
                
                if eos:
                    word_tokens[fname].append(word_vocab.feed(eos))
                    
                    char_array = [char_vocab.feed(c) for c in '{' + eos + '}']
                    char_tokens[fname].append(char_array)

    assert actual_max_word_length <= max_word_length
    """
        print()
        print('actual longest token length is:', actual_max_word_length)
        print('size of word vocabulary:', word_vocab.size)
        print('size of char vocabulary:', char_vocab.size)
        print('number of tokens in train:', len(word_tokens['train']))
        print('number of tokens in valid:', len(word_tokens['valid']))
        print('number of tokens in test:', len(word_tokens['test']))
        """
    # now we know the sizes, create tensors
    word_tensors = {}
    char_tensors = {}
    for fname in ('train', 'valid', 'test'):
        assert len(char_tokens[fname]) == len(word_tokens[fname])
        
        word_tensors[fname] = np.array(word_tokens[fname], dtype=np.int32)
        char_tensors[fname] = np.zeros([len(char_tokens[fname]), actual_max_word_length], dtype=np.int32)
        
        for i, char_array in enumerate(char_tokens[fname]):
            char_tensors[fname] [i,:len(char_array)] = char_array

    return word_vocab, char_vocab, word_tensors, char_tensors, actual_max_word_length


class DataReader:
    
    def __init__(self, word_tensor, char_tensor, batch_size, num_unroll_steps):
        
        length = word_tensor.shape[0]
        assert char_tensor.shape[0] == length
        
        max_word_length = char_tensor.shape[1]
        
        # round down length to whole number of slices
        reduced_length = (length // (batch_size * num_unroll_steps)) * batch_size * num_unroll_steps
        word_tensor = word_tensor[:reduced_length]
        char_tensor = char_tensor[:reduced_length, :]
        
        ydata = np.zeros_like(word_tensor)
        ydata[:-1] = word_tensor[1:].copy()
        ydata[-1] = word_tensor[0].copy()
        
        x_batches = char_tensor.reshape([batch_size, -1, num_unroll_steps, max_word_length])
        y_batches = ydata.reshape([batch_size, -1, num_unroll_steps])
        
        x_batches = np.transpose(x_batches, axes=(1, 0, 2, 3))
        y_batches = np.transpose(y_batches, axes=(1, 0, 2))
        
        self._x_batches = list(x_batches)
        self._y_batches = list(y_batches)
        assert len(self._x_batches) == len(self._y_batches)
        self.length = len(self._y_batches)
        self.batch_size = batch_size
        self.num_unroll_steps = num_unroll_steps
    
    def iter(self):
        
        for x, y in zip(self._x_batches, self._y_batches):
            yield x, y


class adict(dict):
    ''' Attribute dictionary - a convenience data structure, similar to SimpleNamespace in python 3.3
        One can use attributes to read/write dictionary content.
        '''
    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self


def conv2d(input_, output_dim, k_h, k_w, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim])
        b = tf.get_variable('b', [output_dim])
    
    return tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], padding='VALID') + b


def linear(input_, output_size, scope=None):
    '''
        Linear map: output[k] = sum_i(Matrix[k, i] * args[i] ) + Bias[k]
        
        Args:
        args: a tensor or a list of 2D, batch x n, Tensors.
        output_size: int, second dimension of W[i].
        scope: VariableScope for the created subgraph; defaults to "Linear".
        Returns:
        A 2D Tensor with shape [batch x output_size] equal to
        sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
        Raises:
        ValueError: if some of the arguments has unspecified or wrong shape.
        '''
    
    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "SimpleLinear"):
        matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
        bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)
    
    return tf.matmul(input_, tf.transpose(matrix)) + bias_term


def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).
        
        t = sigmoid(Wy + b)
        z = t * g(Wy + b) + (1 - t) * y
        where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
        """
    
    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = f(linear(input_, size, scope='highway_lin_%d' % idx))
            
            t = tf.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias)
            
            output = t * g + (1. - t) * input_
            input_ = output

    return output


def tdnn(input_, kernels, kernel_features, scope='TDNN'):
    '''
        
        :input:           input float tensor of shape [(batch_size*num_unroll_steps) x max_word_length x embed_size]
        :kernels:         array of kernel sizes
        :kernel_features: array of kernel feature sizes (parallel to kernels)
        '''
    assert len(kernels) == len(kernel_features), 'Kernel and Features must have the same size'
    
    max_word_length = input_.get_shape()[1]
    embed_size = input_.get_shape()[-1]
    
    # input_: [batch_size*num_unroll_steps, 1, max_word_length, embed_size]
    input_ = tf.expand_dims(input_, 1)
    
    layers = []
    with tf.variable_scope(scope):
        for kernel_size, kernel_feature_size in zip(kernels, kernel_features):
            reduced_length = max_word_length - kernel_size + 1
            
            # [batch_size x max_word_length x embed_size x kernel_feature_size]
            conv = conv2d(input_, kernel_feature_size, 1, kernel_size, name="kernel_%d" % kernel_size)
            
            # [batch_size x 1 x 1 x kernel_feature_size]
            pool = tf.nn.max_pool(tf.tanh(conv), [1, 1, reduced_length, 1], [1, 1, 1, 1], 'VALID')
            
            layers.append(tf.squeeze(pool, [1, 2]))
        
        if len(kernels) > 1:
            output = tf.concat(layers, 1)
        else:
            output = layers[0]

    return output


def inference_graph(char_vocab_size, word_vocab_size,
                    char_embed_size=15,
                    batch_size=20,
                    num_highway_layers=2,
                    num_rnn_layers=2,
                    rnn_size=650,
                    max_word_length=65,
                    kernels         = [ 1,   2,   3,   4,   5,   6,   7],
                    kernel_features = [50, 100, 150, 200, 200, 200, 200],
                    num_unroll_steps=35,
                    dropout=0.0):
    
    assert len(kernels) == len(kernel_features), 'Kernel and Features must have the same size'
    
    input_ = tf.placeholder(tf.int32, shape=[batch_size, num_unroll_steps, max_word_length], name="input")
    
    ''' First, embed characters '''
    with tf.variable_scope('Embedding'):
        char_embedding = tf.get_variable('char_embedding', [char_vocab_size, char_embed_size])
        
        ''' this op clears embedding vector of first symbol (symbol at position 0, which is by convention the position
            of the padding symbol). It can be used to mimic Torch7 embedding operator that keeps padding mapped to
            zero embedding vector and ignores gradient updates. For that do the following in TF:
            1. after parameter initialization, apply this op to zero out padding embedding vector
            2. after each gradient update, apply this op to keep padding at zero'''
        clear_char_embedding_padding = tf.scatter_update(char_embedding, [0], tf.constant(0.0, shape=[1, char_embed_size]))
        
        # [batch_size x max_word_length, num_unroll_steps, char_embed_size]
        input_embedded = tf.nn.embedding_lookup(char_embedding, input_)
        
        input_embedded = tf.reshape(input_embedded, [-1, max_word_length, char_embed_size])
    
    ''' Second, apply convolutions '''
    # [batch_size x num_unroll_steps, cnn_size]  # where cnn_size=sum(kernel_features)
    input_cnn = tdnn(input_embedded, kernels, kernel_features)
    
    ''' Maybe apply Highway '''
    if num_highway_layers > 0:
        input_cnn = highway(input_cnn, input_cnn.get_shape()[-1], num_layers=num_highway_layers)
    
    ''' Finally, do LSTM '''
    with tf.variable_scope('LSTM'):
        def create_rnn_cell():
            cell = tf.contrib.rnn.BasicLSTMCell(rnn_size, state_is_tuple=True, forget_bias=0.0, reuse=False)
            if dropout > 0.0:
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.-dropout)
            return cell
        
        if num_rnn_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([create_rnn_cell() for _ in range(num_rnn_layers)], state_is_tuple=True)
        else:
            cell = create_rnn_cell()
    
        initial_rnn_state = cell.zero_state(batch_size, dtype=tf.float32)
        
        input_cnn = tf.reshape(input_cnn, [batch_size, num_unroll_steps, -1])
        input_cnn2 = [tf.squeeze(x, [1]) for x in tf.split(input_cnn, num_unroll_steps, 1)]
        
        outputs, final_rnn_state = tf.contrib.rnn.static_rnn(cell, input_cnn2,
                                                             initial_state=initial_rnn_state, dtype=tf.float32)
            
        # linear projection onto output (word) vocab
        logits = []
        with tf.variable_scope('WordEmbedding') as scope:
            for idx, output in enumerate(outputs):
                if idx > 0:
                    scope.reuse_variables()
                logits.append(linear(output, word_vocab_size))

    return adict(
             input = input_,
             clear_char_embedding_padding=clear_char_embedding_padding,
             input_embedded=input_embedded,
             input_cnn=input_cnn,
             initial_rnn_state=initial_rnn_state,
             final_rnn_state=final_rnn_state,
             rnn_outputs=outputs,
             logits = logits
             )


def loss_graph(logits, batch_size, num_unroll_steps):
    
    with tf.variable_scope('Loss'):
        targets = tf.placeholder(tf.int64, [batch_size, num_unroll_steps], name='targets')
        target_list = [tf.squeeze(x, [1]) for x in tf.split(targets, num_unroll_steps, 1)]
        
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = target_list), name='loss')
    
    return adict(
        targets=targets,
        loss=loss
        )


def training_graph(loss, learning_rate=1.0, max_grad_norm=5.0):
    ''' Builds training graph. '''
    global_step = tf.Variable(0, name='global_step', trainable=False)
    
    with tf.variable_scope('SGD_Training'):
        # SGD learning parameter
        learning_rate = tf.Variable(learning_rate, trainable=False, name='learning_rate')
        
        # collect all trainable variables
        tvars = tf.trainable_variables()
        grads, global_norm = tf.clip_by_global_norm(tf.gradients(loss, tvars), max_grad_norm)
        
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
    
    return adict(
        learning_rate=learning_rate,
        global_step=global_step,
        global_norm=global_norm,
        train_op=train_op)


def model_size():
    
    params = tf.trainable_variables()
    size = 0
    for x in params:
        sz = 1
        for dim in x.get_shape():
            sz *= dim.value
        size += sz
    return size



def main(_):
    ''' Loads trained model and evaluates it on test split '''
    
    if FLAGS.load_model is None:
        print('Please specify checkpoint file to load model from')
        return -1
    
    if not os.path.exists(FLAGS.load_model + '.meta'):
        print('Checkpoint file not found', FLAGS.load_model)
        return -1
    
    word_vocab, char_vocab, word_tensors, char_tensors, max_word_length = \
        load_data(FLAGS.data_dir, FLAGS.max_word_length, eos=FLAGS.EOS)

    with tf.Graph().as_default(), tf.Session() as session:
        
        # tensorflow seed must be inside graph
        rand_seed = random.randrange(0, FLAGS.seed + 1)
        #tf.set_random_seed(FLAGS.seed)
        tf.set_random_seed(rand_seed)
        #np.random.seed(seed=FLAGS.seed)
        np.random.seed(seed=rand_seed)
        
        ''' build inference graph '''
        with tf.variable_scope("Model"):
            m = inference_graph(
                                      char_vocab_size=char_vocab.size,
                                      word_vocab_size=word_vocab.size,
                                      char_embed_size=FLAGS.char_embed_size,
                                      batch_size=1,
                                      num_highway_layers=FLAGS.highway_layers,
                                      num_rnn_layers=FLAGS.rnn_layers,
                                      rnn_size=FLAGS.rnn_size,
                                      max_word_length=max_word_length,
                                      kernels=eval(FLAGS.kernels),
                                      kernel_features=eval(FLAGS.kernel_features),
                                      num_unroll_steps=1,
                                      dropout=0)
                                      
                                      # we need global step only because we want to read it from the model
            global_step = tf.Variable(0, dtype=tf.int32, name='global_step')

        saver = tf.train.Saver()
        saver.restore(session, FLAGS.load_model)

        ''' training starts here '''
        rnn_state = session.run(m.initial_rnn_state)
        logits = np.ones((word_vocab.size,))
        rnn_state = session.run(m.initial_rnn_state)
        newline = 0
        print('')
        print('Generating poem...this could take a few minutes')
        while newline < 16:
            logits = logits / FLAGS.temperature
            prob = np.exp(logits)
            prob /= np.sum(prob)
            prob = prob.ravel()
            ix = np.random.choice(range(len(prob)), p=prob)
            
            word = word_vocab.token(ix)
            
            if newline < 1:
                print('')
                newline += 1
            else:
                if word == '|':  # EOS
                   print('<unk>', end=' ')
                elif word == '+':
                    print('')
                    newline += 1
                else:
                    print(word, end=' ')
        
            
            char_input = np.zeros((1, 1, max_word_length))
            for i,c in enumerate('{' + word + '}'):
                char_input[0,0,i] = char_vocab[c]
        
            logits, rnn_state = session.run([m.logits, m.final_rnn_state],
                                            {m.input: char_input,
                                            m.initial_rnn_state: rnn_state})
            logits = np.array(logits)
        print('')

if __name__ == "__main__":
    tf.app.run()



