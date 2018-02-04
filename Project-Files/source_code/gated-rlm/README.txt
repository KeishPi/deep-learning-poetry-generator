/gated-rlm  (Gated Recurrent Language Model)

  base.py  (basic helper functions & Keras’ Progbar object to visualize training)

  data_preprocess.py  (from Miyamoto and Cho’s source code, some functions were
                       slightly modified to accommodate our model and/or TensorFlow)

  layers.py  (the layers for the model)

  gated_model_w_layers.py  (an initial, messy implementation of the model,
                            but the main method still has lines commented out for
                            some of the data preprocessing steps)

  gated_rlm.py	(a cleaner implementation of the model using TensorFlows app.flags
                  function to initialize, share, and update variables through the
                  TensorFlow session)

  train.py  (the training script for the gated rlm model, though most of the
              training logic still resides in gated_rlm.py)
  wordChar_prep.py (some basic utils)
  /data

    /MnC_dicts
      char_dict.pkl  (character vocab for the model from Miyamoto and Cho’s script)
      word_dict.pkl  (word vocab for the model from Miyamoto and Cho’s script)

    /Yoon_dicts
      char_vocab.pkl  (character vocab for the model from Yoon Kim’s script)
      Word_vocab.pkl  (word vocab for the model from Yoon Kim’s script)

    /Basic_dicts
      char_dict.txt  (simple char vocab without indices, one char per line)
      word_dict.txt  (word vocab with occurrences, for GloVe)

    /shakespeare_corpus
      /raw
        citation-n-legal_stuff.txt  (legal information from Project Gutenberg)
        shakespeare_complete.txt  (complete works of Shakespeare)
        Shakespeare_sonnets.txt  (the sonnets)
      /tokenized
        shakespeare_tokenized.txt  (complete works of shakespeare tokenized)

    /sonnets
      /raw
        sonnets.txt  - all sonnets, from Carrie’s model
        train.txt - Carrie’s training set
        test.txt - Carrie’s test set
        valid.txt - Carrie’s development set
      /tokenized
        sonnets_tokenized.txt - all sonnets, from Carrie’s model tokenized
        train_tokenized.txt - Carrie’s training set tokenized
        test_tokenized.txt - Carrie’s test set tokenized
        valid_tokenized.txt  - Carrie’s development set tokenized

      GloVe_vectors_trimmed.200d.npz - 200-dimensional word embeddings trained on
                                      entire corpus, and trimmed to
                                      only include words in sonnet files

/tools
  build_dictionary_char.py - build a char dictionary (in cPickle), by Miyamoto et al
  build_dictionary_word.py - build a word dictionary (in cPickle), by Miyamoto et al
  tokenize_file.py - preprocess files fed to the model, dictionary scripts, and GloVe

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To Tokenize Files:
  $ python tokenize_file.py <File Names>

To Train Model:
  (tensorflow)$ python train.py
  * assumes default settings and file hierarchy above, options yet to be set

  To Sample from Model:
    (tensorflow)$ python sample.py
    * NOT YET IMPLEMENTED
