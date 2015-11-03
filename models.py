from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Merge, TimeDistributedDense
from keras.layers.recurrent import LSTM, GRU
from keras.regularizers import l2
from keras.optimizers import Adam

import h5py
import shutil
import logging

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OneLayerLSTM:

    def __init__(self, hidden_size, vocab_size, dropin, optimiser,
                 l2reg, hsn_size=512, weights=None, gru=False):
        self.hidden_size = hidden_size  # number of units in first LSTM
        self.dropin = dropin  # prob. of dropping input units
        self.vocab_size = vocab_size  # size of word vocabulary
        self.optimiser = optimiser  # optimisation method
        self.l2reg = l2reg  # weight regularisation penalty
        self.hsn_size = hsn_size  # size of the source hidden vector
        self.weights = weights  # initialise with checkpointed weights?
        self.beta1 = None
        self.beta2 = None
        self.epsilon = None
        self.lr = None
        self.gru = gru  # gru recurrent layer? (false = lstm)

    def buildKerasModel(self, use_sourcelang=False, use_image=True):
        '''
        Define the exact structure of your model here. We create an image
        description generation model by merging the VGG image features with
        a word embedding model, with an LSTM over the sequences.

        The order in which these appear below (text, image) is _IMMUTABLE_.
        (Needs to match up with input to model.fit.)
        '''
        logger.info('Building Keras model...')
        logger.info('Using image features: %s', use_image)
        logger.info('Using source language features: %s', use_sourcelang)

        # We will learn word representations for each word
        text = Sequential()
        text.add(TimeDistributedDense(output_dim=self.hidden_size,
                                      input_dim=self.vocab_size,
                                      W_regularizer=l2(self.l2reg)))
        text.add(Dropout(self.dropin))

        if use_sourcelang:
            logger.info("... hsn: adding source features (%d dim)", self.hsn_size)
            source_hidden = Sequential()
            source_hidden.add(TimeDistributedDense(output_dim=self.hidden_size,
                                                   input_dim=self.hsn_size,
                                                   W_regularizer=l2(self.l2reg)))
            source_hidden.add(Dropout(self.dropin))

        if use_image:
            # Compress the 4096D VGG FC_15 features into hidden_size
            logger.info("... visual: adding image features as input features")
            visual = Sequential()
            visual.add(TimeDistributedDense(output_dim=self.hidden_size,
                                            input_dim=4096,
                                            W_regularizer=l2(self.l2reg)))
            visual.add(Dropout(self.dropin))

        # Model is a merge of the VGG features and the Word Embedding vectors
        model = Sequential()
        if use_sourcelang and use_image:
            model.add(Merge([text, source_hidden, visual], mode='sum'))
        else:
            if use_image:
                model.add(Merge([text, visual], mode='sum'))
            elif use_sourcelang:
                model.add(Merge([text, source_hidden], mode='sum'))
            else:  # text sequence model (e.g. source encoder in MT_baseline)
                assert not use_sourcelang and not use_image
                model.add(text)

        if self.gru:
            model.add(GRU(output_dim=self.hidden_size,
                          input_dim=self.hidden_size,
                          return_sequences=True))
        else:
            model.add(LSTM(output_dim=self.hidden_size,
                           input_dim=self.hidden_size,
                           return_sequences=True))

        model.add(TimeDistributedDense(output_dim=self.vocab_size,
                                       input_dim=self.hidden_size,
                                       W_regularizer=l2(self.l2reg)))
        model.add(Activation('time_distributed_softmax'))

#        if self.optimiser == 'adam':
#            # allow user-defined hyper-parameters for ADAM because it is
#            # our preferred optimiser
#            lr = self.lr if self.lr is not None else 0.001
#            beta1 = self.beta1 if self.beta1 is not None else 0.9
#            beta2 = self.beta2 if self.beta2 is not None else 0.999
#            epsilon = self.epsilon if self.epsilon is not None else 1e-8
#            optimiser = Adam(lr=lr, beta1=beta1,
#                             beta2=beta2, epsilon=epsilon)
#            model.compile(loss='categorical_crossentropy',
#                          optimizer=optimiser)
#        else:
        model.compile(loss='categorical_crossentropy',
                      optimizer=self.optimiser)

        if self.weights is not None:
            logger.info("... with weights defined in %s", self.weights)
            # Initialise the weights of the model
            shutil.copyfile("%s/weights.hdf5" % self.weights,
                            "%s/weights.hdf5.bak" % self.weights)
            model.load_weights("%s/weights.hdf5" % self.weights)

        return model

    def buildHSNActivations(self, use_image=True):
        '''
        Define the exact structure of your model here. We create an image
        description generation model by merging the VGG image features with
        a word embedding model, with an LSTM over the sequences.

        The order in which these appear below (text, image) is _IMMUTABLE_.
        '''

        logger.info('Building Keras model...')
        logger.info('Using image features: %s', use_image)

        # We will learn word representations for each word
        text = Sequential()
        text.add(TimeDistributedDense(self.vocab_size, self.hidden_size,
                                      W_regularizer=l2(self.l2reg)))
        text.add(Dropout(self.dropin))

        if use_image:
            # Compress the 4096D VGG FC_15 features into hidden_size
            visual = Sequential()
            visual.add(TimeDistributedDense(4096, self.hidden_size,
                                            W_regularizer=l2(self.l2reg)))
            visual.add(Dropout(self.dropin))

        # Model is a merge of the VGG features and the Word Embedding vectors
        model = Sequential()
        if use_image:
            model.add(Merge([text, visual], mode='sum'))
        else:
            model.add(text)
        model.add(LSTM(self.hidden_size, self.hidden_size,  # 1st GRU layer
                       return_sequences=True))

        if self.weights is not None:
            logger.info("... with weights defined in %s", self.weights)
            # Initialise the weights of the model
            shutil.copyfile("%s/weights.hdf5" % self.weights,
                            "%s/weights.hdf5.bak" % self.weights)
            f = h5py.File("%s/weights.hdf5" % self.weights)
            for k in range(f.attrs['nb_layers']-2):
                g = f['layer_{}'.format(k)]
                weights = [g['param_{}'.format(p)]
                           for p in range(g.attrs['nb_params'])]
                model.layers[k].set_weights(weights)
            f.close()

        model.compile(loss='categorical_crossentropy',
                      optimizer=self.optimiser)
        return model

    def buildMergeActivations(self, use_image=True, use_sourcelang=False):
        '''
        Define the exact structure of your model here. We create an image
        description generation model by merging the VGG image features with
        a word embedding model, with an LSTM over the sequences.

        The order in which these appear below (text, image) is _IMMUTABLE_.
        '''

        logger.info('Building Keras model...')

        # We will learn word representations for each word
        text = Sequential()
        text.add(TimeDistributedDense(self.vocab_size, self.hidden_size,
                                      W_regularizer=l2(self.l2reg)))
        text.add(Dropout(self.dropin))

        if use_sourcelang:
            logger.info("... hsn: adding source language vector as input features")
            source_hidden = Sequential()
            source_hidden.add(TimeDistributedDense(self.hsn_size, self.hidden_size,
                                                   W_regularizer=l2(self.l2reg)))
            source_hidden.add(Dropout(self.dropin))

        if use_image:
            # Compress the 4096D VGG FC_15 features into hidden_size
            logger.info("... visual: adding image features as input features")
            visual = Sequential()
            visual.add(TimeDistributedDense(4096, self.hidden_size,
                                            W_regularizer=l2(self.l2reg)))
            visual.add(Dropout(self.dropin))

        # Model is a merge of the VGG features and the Word Embedding vectors
        model = Sequential()
        if use_sourcelang and use_image:
            model.add(Merge([text, source_hidden, visual], mode='sum'))
        else:
            if use_image:
                model.add(Merge([text, visual], mode='sum'))
            elif use_sourcelang:
                model.add(Merge([text, source_hidden], mode='sum'))
            else: # text sequence model (e.g. source encoder in MT_baseline)
                assert not use_sourcelang and not use_image
                model.add(text)

        model.compile(loss='categorical_crossentropy',
                      optimizer=self.optimiser)

        if self.weights is not None:
            logger.info("... with weights defined in %s", self.weights)
            # Initialise the weights of the model
            shutil.copyfile("%s/weights.hdf5" % self.weights,
                            "%s/weights.hdf5.bak" % self.weights)
            f = h5py.File("%s/weights.hdf5" % self.weights)
            for k in range(f.attrs['nb_layers']-3):
                g = f['layer_{}'.format(k)]
                weights = [g['param_{}'.format(p)]
                           for p in range(g.attrs['nb_params'])]
                model.layers[k].set_weights(weights)
            f.close()

        return model
