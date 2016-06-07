from keras.models import Sequential, Graph
from keras.layers.core import Activation, Dropout, Merge, TimeDistributedDense, Masking
from keras.layers.recurrent import LSTM, GRU
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.optimizers import Adam

import h5py
import shutil
import logging
import sys

# Set up logger
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

class NIC:

    def __init__(self, embed_size, hidden_size, vocab_size, dropin, optimiser,
                 l2reg, hsn_size=512, weights=None, gru=False,
                 clipnorm=-1, batch_size=None, t=None, lr=0.001):

        self.max_t = t  # Expected timesteps. Needed to build the Theano graph

        # Model hyperparameters
        self.vocab_size = vocab_size  # size of word vocabulary
        self.embed_size = embed_size  # number of units in a word embedding
        self.hsn_size = hsn_size  # size of the source hidden vector
        self.hidden_size = hidden_size  # number of units in first LSTM
        self.gru = gru  # gru recurrent layer? (false = lstm)
        self.dropin = dropin  # prob. of dropping input units
        self.l2reg = l2reg  # weight regularisation penalty

        # Optimiser hyperparameters
        self.optimiser = optimiser  # optimisation method
        self.lr = lr
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.clipnorm = clipnorm

        self.weights = weights  # initialise with checkpointed weights?

    def buildKerasModel(self, use_sourcelang=False, use_image=True):
        '''
        Define the exact structure of your model here. We create an image
        description generation model by merging the VGG image features with
        a word embedding model, with an LSTM over the sequences.
        '''
        logger.info('Building Keras model...')

        model = Graph()
        model.add_input('text', input_shape=(self.max_t, self.vocab_size))
        model.add_node(Masking(mask_value=0.), input='text', name='text_mask')

        # Word embeddings
        model.add_node(TimeDistributedDense(output_dim=self.embed_size,
                                            input_dim=self.vocab_size,
                                            W_regularizer=l2(self.l2reg)),
                                            name="w_embed", input='text_mask')

        # Embed -> Hidden
        model.add_node(TimeDistributedDense(output_dim=self.hidden_size,
                                      input_dim=self.vocab_size,
                                      W_regularizer=l2(self.l2reg)),
                                      name='embed_to_hidden',
                                      input='w_embed')

        if use_image:
            # Image 'embedding'
            logger.info('Using image features: %s', use_image)
            model.add_input('img', input_shape=(self.max_t, 4096))
            model.add_node(Masking(mask_value=0.),
                           input='img', name='img_mask')
            model.add_node(TimeDistributedDense(output_dim=self.hidden_size,
                                                input_dim=4096,
                                                W_regularizer=l2(self.l2reg)),
                                                name='i_embed',
                                                input='img_mask')
            model.add_node(Dropout(self.dropin), name='i_embed_drop', input='i_embed')


        if use_sourcelang:
            logger.info('Using source features: %s', use_sourcelang)
            logger.info('Size of source feature vectors: %d', self.hsn_size)
            model.add_input('source', input_shape=(self.max_t, self.hsn_size))
            model.add_node(Masking(mask_value=0.),
                           input='source',
                           name='source_mask')
	    model.add_node(Activation('relu'), name='s_relu', input='source_mask')
            model.add_node(TimeDistributedDense(output_dim=self.hidden_size,
                                                input_dim=self.hsn_size,
                                                W_regularizer=l2(self.l2reg)),
                                                name="s_embed",
                                                input="s_relu")
            model.add_node(Dropout(self.dropin), 
                           name="s_embed_drop",
                           input="s_embed")

	rnn_input_dim = self.hidden_size
        # Input nodes for the recurrent layer
        if use_image and use_sourcelang:
            recurrent_inputs = ['embed_to_hidden',
                                'i_embed_drop',
                                's_embed_drop']
	    rnn_input_dim *= 1
        elif use_image:
            recurrent_inputs = ['embed_to_hidden', 'i_embed_drop']
	    rnn_input_dim *= 2
        elif use_sourcelang:
            recurrent_inputs = ['embed_to_hidden', 's_embed_drop']
	    rnn_input_dim *= 2

        # Recurrent layer
        if self.gru:
            logger.info("Building a GRU with recurrent inputs %s", recurrent_inputs)
            model.add_node(GRU(output_dim=self.hidden_size,
                           input_dim=rnn_input_dim,
                           return_sequences=True,
                           W_regularizer=l2(self.l2reg),
                           U_regularizer=l2(self.l2reg)),
                           name='rnn',
                           inputs=recurrent_inputs,
                           merge_mode='concat')

        else:
            logger.info("Building an LSTM with recurrent inputs %s", recurrent_inputs)
            model.add_node(LSTM(output_dim=self.hidden_size,
                           input_dim=rnn_input_dim,
                           return_sequences=True,
                           W_regularizer=l2(self.l2reg),
                           U_regularizer=l2(self.l2reg)),
                           name='rnn',
                           inputs=recurrent_inputs,
                           merge_mode='concat')

        model.add_node(TimeDistributedDense(output_dim=self.vocab_size,
                                            input_dim=self.hidden_size,
                                            W_regularizer=l2(self.l2reg),
                                            activation='softmax'),
                                            name='output',
                                            input='rnn',
                                            create_output=True)

        if self.optimiser == 'adam':
            # allow user-defined hyper-parameters for ADAM because it is
            # our preferred optimiser
            optimiser = Adam(lr=self.lr, beta1=self.beta1,
                             beta2=self.beta2, epsilon=self.epsilon,
                             clipnorm=self.clipnorm)
            model.compile(optimiser, {'output': 'categorical_crossentropy'})
        else:
            model.compile(self.optimiser, {'output': 'categorical_crossentropy'})

        if self.weights is not None:
            logger.info("... with weights defined in %s", self.weights)
            # Initialise the weights of the model
            shutil.copyfile("%s/weights.hdf5" % self.weights,
                            "%s/weights.hdf5.bak" % self.weights)
            model.load_weights("%s/weights.hdf5" % self.weights)

        #plot(model, to_file="model.png")

        return model

    def buildHSNActivations(self, use_image=True):
        '''
        Define the exact structure of your model here. We create an image
        description generation model by merging the VGG image features with
        a word embedding model, with an LSTM over the sequences.
        '''

        logger.info('Building Keras model...')

        model = Graph()
        model.add_input('text', input_shape=(self.max_t, self.vocab_size))
        model.add_node(Masking(mask_value=0.), input='text', name='text_mask')

        # Word embeddings
        model.add_node(TimeDistributedDense(output_dim=self.embed_size,
                                            input_dim=self.vocab_size,
                                            W_regularizer=l2(self.l2reg)),
                                            name="w_embed", input='text_mask')

        # Embed -> Hidden
        model.add_node(TimeDistributedDense(output_dim=self.hidden_size,
                                      input_dim=self.vocab_size,
                                      W_regularizer=l2(self.l2reg)),
                                      name='embed_to_hidden',
                                      input='w_embed')

        if use_image:
            # Image 'embedding'
            logger.info('Using image features: %s', use_image)
            model.add_input('img', input_shape=(self.max_t, 4096))
            model.add_node(Masking(mask_value=0.),
                           input='img', name='img_mask')
            model.add_node(TimeDistributedDense(output_dim=self.hidden_size,
                                                input_dim=4096,
                                                W_regularizer=l2(self.l2reg)),
                                                name='i_embed',
                                                input='img_mask')
            model.add_node(Dropout(self.dropin), name='i_embed_drop', input='i_embed')


        if use_image:
            recurrent_inputs = ['embed_to_hidden', 'i_embed_drop']

        # Recurrent layer
        if self.gru:
            logger.info("Building a GRU with recurrent inputs %s", recurrent_inputs)
            model.add_node(GRU(output_dim=self.hidden_size,
                           input_dim=2*self.hidden_size,
                           return_sequences=True,
                           W_regularizer=l2(self.l2reg),
                           U_regularizer=l2(self.l2reg)),
                           name='rnn',
                           inputs=recurrent_inputs,
                           merge_mode='concat',
                           create_output=True)

        else:
            logger.info("Building an LSTM with recurrent inputs %s", recurrent_inputs)
            model.add_node(LSTM(output_dim=self.hidden_size,
                           input_dim=2*self.hidden_size,
                           return_sequences=True,
                           W_regularizer=l2(self.l2reg),
                           U_regularizer=l2(self.l2reg)),
                           name='rnn',
                           inputs=recurrent_inputs,
                           merge_mode='concat',
                           create_output=True)

        if self.optimiser == 'adam':
            # allow user-defined hyper-parameters for ADAM because it is
            # our preferred optimiser
            optimiser = Adam(lr=self.lr, beta1=self.beta1,
                             beta2=self.beta2, epsilon=self.epsilon,
                             clipnorm=self.clipnorm)
            model.compile(optimiser, {'rnn': 'categorical_crossentropy'})
        else:
            model.compile(self.optimiser, {'rnn': 'categorical_crossentropy'})

        if self.weights is not None:
            logger.info("... with weights defined in %s", self.weights)
            # Initialise the weights of the model
            shutil.copyfile("%s/weights.hdf5" % self.weights,
                            "%s/weights.hdf5.bak" % self.weights)
            model.load_weights("%s/weights.hdf5" % self.weights)

        #plot(model, to_file="model.png")

        return model

class MRNN:

    def __init__(self, embed_size, hidden_size, vocab_size, dropin, optimiser,
                 l2reg, hsn_size=512, weights=None, gru=False,
                 clipnorm=-1, batch_size=None, t=None, lr=0.001):

        self.max_t = t  # Expected timesteps. Needed to build the Theano graph

        # Model hyperparameters
        self.vocab_size = vocab_size  # size of word vocabulary
        self.embed_size = embed_size  # number of units in a word embedding
        self.hsn_size = hsn_size  # size of the source hidden vector
        self.hidden_size = hidden_size  # number of units in first LSTM
        self.gru = gru  # gru recurrent layer? (false = lstm)
        self.dropin = dropin  # prob. of dropping input units
        self.l2reg = l2reg  # weight regularisation penalty

        # Optimiser hyperparameters
        self.optimiser = optimiser  # optimisation method
        self.lr = lr
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.clipnorm = clipnorm

        self.weights = weights  # initialise with checkpointed weights?

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

        model = Graph()
        model.add_input('text', input_shape=(self.max_t, self.vocab_size))
        model.add_node(Masking(mask_value=0.), input='text', name='text_mask')

        # Word embeddings
        model.add_node(TimeDistributedDense(output_dim=self.embed_size,
                                            input_dim=self.vocab_size,
                                            W_regularizer=l2(self.l2reg)),
                                            name="w_embed", input='text_mask')
        model.add_node(Dropout(self.dropin),
                       name="w_embed_drop",
                       input="w_embed")

        # Embed -> Hidden
        model.add_node(TimeDistributedDense(output_dim=self.hidden_size,
                                      input_dim=self.embed_size,
                                      W_regularizer=l2(self.l2reg)),
                                      name='embed_to_hidden', input='w_embed_drop')
        recurrent_inputs = 'embed_to_hidden'

        # Source language input
        if use_sourcelang:
            model.add_input('source', input_shape=(self.max_t, self.hsn_size))
            model.add_node(Masking(mask_value=0.),
                           input='source',
                           name='source_mask')

            model.add_node(TimeDistributedDense(output_dim=self.hidden_size,
                                                input_dim=self.hsn_size,
                                                W_regularizer=l2(self.l2reg)),
                                                name="s_embed",
                                                input="source_mask")
            model.add_node(Dropout(self.dropin),
                           name="s_embed_drop",
                           input="s_embed")
            recurrent_inputs = ['embed_to_hidden', 's_embed_drop']

        # Recurrent layer
        if self.gru:
            model.add_node(GRU(output_dim=self.hidden_size,
                           input_dim=self.hidden_size,
                           return_sequences=True), name='rnn',
                           input=recurrent_inputs)

        else:
            model.add_node(LSTM(output_dim=self.hidden_size,
                           input_dim=self.hidden_size,
                           return_sequences=True), name='rnn',
                           input=recurrent_inputs)

        # Image 'embedding'
        model.add_input('img', input_shape=(self.max_t, 4096))
        model.add_node(Masking(mask_value=0.),
                       input='img', name='img_mask')

        model.add_node(TimeDistributedDense(output_dim=self.hidden_size,
                                            input_dim=4096,
                                            W_regularizer=l2(self.l2reg)),
                                            name='i_embed', input='img_mask')
        model.add_node(Dropout(self.dropin), name='i_embed_drop', input='i_embed')

        # Multimodal layer outside the recurrent layer
        model.add_node(TimeDistributedDense(output_dim=self.hidden_size,
                                       input_dim=self.hidden_size,
                                       W_regularizer=l2(self.l2reg)),
                                       name='m_layer',
                                       inputs=['rnn','i_embed_drop', 'embed_to_hidden'],
                                       merge_mode='sum')

        model.add_node(TimeDistributedDense(output_dim=self.vocab_size,
                                            input_dim=self.hidden_size,
                                            W_regularizer=l2(self.l2reg),
                                            activation='softmax'),
                                            name='output',
                                            input='m_layer',
                                            create_output=True)

        if self.optimiser == 'adam':
            # allow user-defined hyper-parameters for ADAM because it is
            # our preferred optimiser
            optimiser = Adam(lr=self.lr, beta1=self.beta1,
                             beta2=self.beta2, epsilon=self.epsilon,
                             clipnorm=self.clipnorm)
            model.compile(optimiser, {'output': 'categorical_crossentropy'})
        else:
            model.compile(self.optimiser, {'output': 'categorical_crossentropy'})

        if self.weights is not None:
            logger.info("... with weights defined in %s", self.weights)
            # Initialise the weights of the model
            shutil.copyfile("%s/weights.hdf5" % self.weights,
                            "%s/weights.hdf5.bak" % self.weights)
            model.load_weights("%s/weights.hdf5" % self.weights)

        #plot(model, to_file="model.png")

        return model

