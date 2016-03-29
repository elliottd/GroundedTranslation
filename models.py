from keras.models import Sequential, Graph
from keras.layers.core import Activation, Dropout, Merge, TimeDistributedDense
from keras.layers.recurrent import LSTM, GRU
from keras.regularizers import l2
from keras.optimizers import Adam

import h5py
import shutil
import logging
import sys

from memory_profiler import profile

# Set up logger
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

class GraphLSTM:

    def __init__(self, hidden_size, vocab_size, dropin, optimiser,
                 l2reg, hsn_size=512, weights=None, gru=False,
                 clipnorm=-1, batch_size=None, t=None):
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
        self.clipnorm = clipnorm
        self.max_t = t
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
        model = Graph()
        model.add_input('text', input_shape=(self.max_t, self.vocab_size))
        model.add_node(TimeDistributedDense(output_dim=self.hidden_size,
                                      input_dim=self.vocab_size,
                                      W_regularizer=l2(self.l2reg)),
                                      name='w_embed', input='text')
        model.add_node(Dropout(self.dropin), name='w_embed_drop', input='w_embed')
        model.add_node(LSTM(output_dim=self.hidden_size,
                       input_dim=self.hidden_size,
                       return_sequences=True), name='lstm',
                       input='w_embed_drop')
        model.add_input('img', input_shape=(self.max_t, 4096))
        model.add_node(TimeDistributedDense(output_dim=self.hidden_size,
                                            input_dim=4096,
                                            W_regularizer=l2(self.l2reg)), name='i_embed', input='img')
        model.add_node(Dropout(self.dropin), name='i_embed_drop', input='i_embed')
        model.add_node(TimeDistributedDense(output_dim=2*self.hidden_size,
                                       input_dim=2*self.hidden_size,
                                       W_regularizer=l2(self.l2reg)),
                                       name='m', inputs=['lstm',
                                       'i_embed_drop'])
        model.add_node(TimeDistributedDense(output_dim=self.vocab_size,
                                            input_dim=2*self.hidden_size,
                                            W_regularizer=l2(self.l2reg),
                                            activation='softmax'),
                                            name='output',
                                            input='m',
                                            create_output=True)

        if self.optimiser == 'adam':
            # allow user-defined hyper-parameters for ADAM because it is
            # our preferred optimiser
            lr = self.lr if self.lr is not None else 0.001
            beta1 = self.beta1 if self.beta1 is not None else 0.9
            beta2 = self.beta2 if self.beta2 is not None else 0.999
            epsilon = self.epsilon if self.epsilon is not None else 1e-8
            optimiser = Adam(lr=lr, beta1=beta1,
                             beta2=beta2, epsilon=epsilon,
                             clipnorm=self.clipnorm)
            model.compile(optimiser, {'output': 'categorical_crossentropy'})
        else:
            model.compile(loss='categorical_crossentropy',
                          optimizer=self.optimiser)

        if self.weights is not None:
            logger.info("... with weights defined in %s", self.weights)
            # Initialise the weights of the model
            shutil.copyfile("%s/weights.hdf5" % self.weights,
                            "%s/weights.hdf5.bak" % self.weights)
            model.load_weights("%s/weights.hdf5" % self.weights)

        #plot(model, to_file="model.png")

        return model


class OneLayerLSTM:

    def __init__(self, hidden_size, vocab_size, dropin, optimiser,
                 l2reg, hsn_size=512, weights=None, gru=False,
                 clipnorm=-1):
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
        self.clipnorm = clipnorm
        self.gru = gru  # gru recurrent layer? (false = lstm)

    @profile
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
        model.add(Activation('softmax'))

        if self.optimiser == 'adam':
            # allow user-defined hyper-parameters for ADAM because it is
            # our preferred optimiser
            lr = self.lr if self.lr is not None else 0.001
            beta1 = self.beta1 if self.beta1 is not None else 0.9
            beta2 = self.beta2 if self.beta2 is not None else 0.999
            epsilon = self.epsilon if self.epsilon is not None else 1e-8
            optimiser = Adam(lr=lr, beta1=beta1,
                             beta2=beta2, epsilon=epsilon,
                             clipnorm=self.clipnorm)
            model.compile(loss='categorical_crossentropy',
                          optimizer=optimiser)
        else:
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
