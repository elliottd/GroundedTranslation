from keras.models import Sequential
# Below: Dense was unused
from keras.layers.core import Activation, Dropout, Merge, TimeDistributedDense
from keras.layers.recurrent import LSTM
# from keras.optimizers import RMSprop, SGD, Adagrad # unused
from keras.regularizers import l2

import math
import shutil

class TwoLayerLSTM:

    def __init__(self, hidden_size, vocab_size, dropin, droph, optimiser,
                 l2reg, weights=None):
        self.hidden_size = hidden_size  # number of units in first LSTM
        self.dropin = dropin  # prob. of dropping input units
        self.droph = droph  # prob. of dropping hidden->hidden units
        self.vocab_size = vocab_size  # size of word vocabulary
        self.optimiser = optimiser  # optimisation method
        self.l2reg = l2reg  # weight regularisation penalty
        self.weights = weights  # initialise with checkpointed weights?

    def buildKerasModel(self):
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

        # Compress the 4096D VGG FC_7 features into hidden_size
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
