from keras.models import Sequential
# Below: Dense was unused
from keras.layers.core import Activation, Dropout, Merge, TimeDistributedDense
from keras.layers.recurrent import LSTM, GRU
# from keras.optimizers import RMSprop, SGD, Adagrad # unused
from keras.regularizers import l2

import h5py

import math
import shutil

class OneLayerLSTM:

    def __init__(self, hidden_size, vocab_size, dropin, optimiser,
                 l2reg, hsn_size=512, weights=None):
        self.hidden_size = hidden_size  # number of units in first LSTM
        self.dropin = dropin  # prob. of dropping input units
        self.vocab_size = vocab_size  # size of word vocabulary
        self.optimiser = optimiser  # optimisation method
        self.l2reg = l2reg  # weight regularisation penalty
        self.hsn_size = hsn_size # size of the source hidden vector
        self.weights = weights  # initialise with checkpointed weights?

    def buildKerasModel(self, hsn=False):
        '''
        Define the exact structure of your model here. We create an image
        description generation model by merging the VGG image features with
        a word embedding model, with an LSTM over the sequences.

        The order in which these appear below (text, image) is _IMMUTABLE_.
        '''

        print('Building Keras model...')

        # We will learn word representations for each word
        text = Sequential()
        text.add(TimeDistributedDense(self.vocab_size, self.hidden_size,
                                      W_regularizer=l2(self.l2reg)))
        text.add(Dropout(self.dropin))
        #text.add(Activation('tanh'))

        if hsn:
            print("... with hsn")
            source_hidden = Sequential()
            source_hidden.add(TimeDistributedDense(self.hsn_size, self.hidden_size,
                                                   W_regularizer=l2(self.l2reg)))
            source_hidden.add(Dropout(self.dropin))
            #source_hidden.add(Activation('tanh'))

        # Compress the 4096D VGG FC_15 features into hidden_size
        visual = Sequential()
        visual.add(TimeDistributedDense(4096, self.hidden_size,
                                        W_regularizer=l2(self.l2reg)))
        visual.add(Dropout(self.dropin))
        #visual.add(Activation('tanh'))

        # Model is a merge of the VGG features and the Word Embedding vectors
        model = Sequential()
        if hsn:
          model.add(Merge([text, source_hidden, visual], mode='sum'))
        else:
          model.add(Merge([text, visual], mode='sum'))

        model.add(LSTM(self.hidden_size, self.hidden_size,  # 1st LSTM layer
                       return_sequences=True))
        model.add(TimeDistributedDense(self.hidden_size, self.vocab_size,
                                       W_regularizer=l2(self.l2reg)))
        model.add(Activation('time_distributed_softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer=self.optimiser)

        if self.weights is not None:
            # Initialise the weights of the model
            shutil.copyfile("%s/weights.hdf5" % self.weights, 
                           "%s/weights.hdf5.bak" % self.weights)
            model.load_weights("%s/weights.hdf5" % self.weights)

        return model

    def buildHSNActivations(self):
        '''
        Define the exact structure of your model here. We create an image
        description generation model by merging the VGG image features with
        a word embedding model, with an LSTM over the sequences.

        The order in which these appear below (text, image) is _IMMUTABLE_.
        '''

        print('Building Keras model...')

        # We will learn word representations for each word
        text = Sequential()
        text.add(TimeDistributedDense(self.vocab_size, self.hidden_size,
                                      W_regularizer=l2(self.l2reg)))
        text.add(Dropout(self.dropin))

        # Compress the 4096D VGG FC_15 features into hidden_size
        visual = Sequential()
        visual.add(TimeDistributedDense(4096, self.hidden_size,
                                        W_regularizer=l2(self.l2reg)))
        text.add(Dropout(self.dropin))

        # Model is a merge of the VGG features and the Word Embedding vectors
        model = Sequential()
        model.add(Merge([text, visual], mode='sum'))
        model.add(LSTM(self.hidden_size, self.hidden_size,  # 1st GRU layer
                       return_sequences=True))

        model.compile(loss='categorical_crossentropy',
                      optimizer=self.optimiser)

        if self.weights is not None:
            # Initialise the weights of the model
            shutil.copyfile("%s/weights.hdf5" % self.weights, 
                           "%s/weights.hdf5.bak" % self.weights)
            f = h5py.File("%s/weights.hdf5" % self.weights)
            for k in range(f.attrs['nb_layers']-2):
                g = f['layer_{}'.format(k)]
                weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
                model.layers[k].set_weights(weights)
            f.close()

        return model

class TwoLayerLSTM:

    def __init__(self, hidden_size, vocab_size, dropin, droph, optimiser,
                 l2reg, hsn=False, weights=None):
        self.hidden_size = hidden_size  # number of units in first LSTM
        self.dropin = dropin  # prob. of dropping input units
        self.droph = droph  # prob. of dropping hidden->hidden units
        self.vocab_size = vocab_size  # size of word vocabulary
        self.optimiser = optimiser  # optimisation method
        self.l2reg = l2reg  # weight regularisation penalty
        self.hsn_size = 409 # size of the source hidden vector
        self.weights = weights  # initialise with checkpointed weights?

    def buildKerasModel(self, hsn=False):
        '''
        Define the exact structure of your model here. We create an image
        description generation model by merging the VGG image features with
        a word embedding model, with an LSTM over the sequences.

        The order in which these appear below (text, image) is _IMMUTABLE_.
        '''

        print('Building Keras model...')

        # We will learn word representations for each word
        text = Sequential()
        text.add(TimeDistributedDense(self.vocab_size, self.hidden_size,
                                      W_regularizer=l2(self.l2reg)))
        text.add(Dropout(self.dropin))

        if hsn:
            print("... with hsn")
            source_hidden = Sequential()
            source_hidden.add(TimeDistributedDense(self.hsn_size, self.hidden_size,
                                                   W_regularizer=l2(self.l2reg),
                                                   activation='tanh'))
            source_hidden.add(Dropout(self.dropin))

        # Compress the 4096D VGG FC_15 features into hidden_size
        visual = Sequential()
        visual.add(TimeDistributedDense(4096, self.hidden_size,
                                        W_regularizer=l2(self.l2reg),
                                        activation='tanh'))
        text.add(Dropout(self.dropin))

        # Model is a merge of the VGG features and the Word Embedding vectors
        model = Sequential()
        if hsn:
          model.add(Merge([text, source_hidden, visual], mode='sum'))
        else:
          model.add(Merge([text, visual], mode='sum'))

        model.add(LSTM(self.hidden_size, self.hidden_size,  # 1st LSTM layer
                       return_sequences=True))

        # The second layer has 80% of the units of the first layer
        stacked_LSTM_size = int(math.floor(self.hidden_size * 0.8))
        model.add(Dropout(self.droph))
        model.add(LSTM(self.hidden_size, stacked_LSTM_size,  # 2nd LSTM layer
                       return_sequences=True))
        model.add(TimeDistributedDense(stacked_LSTM_size, self.vocab_size,
                                       W_regularizer=l2(self.l2reg)))
        model.add(Activation('time_distributed_softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer=self.optimiser)

        if self.weights is not None:
            # Initialise the weights of the model
            shutil.copyfile("%s/weights.hdf5" % self.weights, 
                           "%s/weights.hdf5.bak" % self.weights)
            model.load_weights("%s/weights.hdf5" % self.weights)

        return model

    def buildHSNActivations(self):
        '''
        Define the exact structure of your model here. We create an image
        description generation model by merging the VGG image features with
        a word embedding model, with an LSTM over the sequences.

        The order in which these appear below (text, image) is _IMMUTABLE_.
        '''

        print('Building Keras model...')

        # We will learn word representations for each word
        text = Sequential()
        text.add(TimeDistributedDense(self.vocab_size, self.hidden_size,
                                      W_regularizer=l2(self.l2reg)))
        text.add(Dropout(self.dropin))

        # Compress the 4096D VGG FC_15 features into hidden_size
        visual = Sequential()
        visual.add(TimeDistributedDense(4096, self.hidden_size,
                                        W_regularizer=l2(self.l2reg)))
        text.add(Dropout(self.dropin))

        # Model is a merge of the VGG features and the Word Embedding vectors
        model = Sequential()
        model.add(Merge([text, visual], mode='sum'))
        model.add(LSTM(self.hidden_size, self.hidden_size,  # 1st LSTM layer
                       return_sequences=True))

        # The second layer has 80% of the units of the first layer
        stacked_LSTM_size = int(math.floor(self.hidden_size * 0.8))
        model.add(Dropout(self.droph))
        model.add(LSTM(self.hidden_size, stacked_LSTM_size,  # 2nd LSTM layer
                       return_sequences=True))

        model.compile(loss='categorical_crossentropy',
                      optimizer=self.optimiser)

        if self.weights is not None:
            # Initialise the weights of the model
            shutil.copyfile("%s/weights.hdf5" % self.weights, 
                           "%s/weights.hdf5.bak" % self.weights)
            f = h5py.File("%s/weights.hdf5" % self.weights)
            for k in range(f.attrs['nb_layers']-2):
                g = f['layer_{}'.format(k)]
                weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
                model.layers[k].set_weights(weights)
            f.close()

        return model
