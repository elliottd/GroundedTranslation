from keras.models import Model
from keras.layers import Input, Activation, Dropout, Merge, TimeDistributed, Masking, Dense
from keras.layers.recurrent import LSTM, GRU
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.optimizers import Adam
from keras import backend as K


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

        text_input = Input(shape=(self.max_t, self.vocab_size), name='text')
        text_mask = Masking(mask_value=0., name='text_mask')(text_input)

        # Word embeddings
        wemb = TimeDistributed(Dense(output_dim=self.embed_size,
                                      input_dim=self.vocab_size,
                                      W_regularizer=l2(self.l2reg)),
                                      name="w_embed")(text_mask)
        drop_wemb = Dropout(self.dropin, name="wemb_drop")(wemb)

        # Embed -> Hidden
        emb_to_hidden = TimeDistributed(Dense(output_dim=self.hidden_size,
                                      input_dim=self.vocab_size,
                                      W_regularizer=l2(self.l2reg)),
                                      name='wemb_to_hidden')(drop_wemb)

        if use_image:
            # Image 'embedding'
            logger.info('Using image features: %s', use_image)
            img_input = Input(shape=(self.max_t, 4096), name='img')
            img_emb = TimeDistributed(Dense(output_dim=self.hidden_size,
                                            input_dim=4096,
                                            W_regularizer=l2(self.l2reg)),
                                            name='img_emb')(img_input)
            img_drop = Dropout(self.dropin, name='img_embed_drop')(img_emb)

        if use_sourcelang:
            logger.info('Using source features: %s', use_sourcelang)
            logger.info('Size of source feature vectors: %d', self.hsn_size)
            src_input = Input(shape=(self.max_t, self.hsn_size), name='src')
            src_relu = Activation('relu', name='src_relu')(src_input)
            src_embed = TimeDistributed(Dense(output_dim=self.hidden_size,
                                              input_dim=self.hsn_size,
                                              W_regularizer=l2(self.l2reg)),
                                              name="src_embed")(src_relu)
            src_drop = Dropout(self.dropin, name="src_drop")(src_embed)

        # Input nodes for the recurrent layer
        rnn_input_dim = self.hidden_size
        if use_image and use_sourcelang:
            recurrent_inputs = [emb_to_hidden, img_drop, src_drop]
            recurrent_inputs_names = ['emb_to_hidden', 'img_drop', 'src_drop']
            inputs = [text_input, img_input, src_input]
        elif use_image:
            recurrent_inputs = [emb_to_hidden, img_drop]
            recurrent_inputs_names = ['emb_to_hidden', 'img_drop']
            inputs = [text_input, img_input]
        elif use_sourcelang:
            recurrent_inputs = [emb_to_hidden, src_drop]
            recurrent_inputs_names = ['emb_to_hidden', 'src_drop']
            inputs = [text_input, src_input]
        merged_input = Merge(mode='sum')(recurrent_inputs)

        # Recurrent layer
        if self.gru:
            logger.info("Building a GRU with recurrent inputs %s", recurrent_inputs_names)
            rnn = GRU(output_dim=self.hidden_size,
                      input_dim=rnn_input_dim,
                      return_sequences=True,
                      W_regularizer=l2(self.l2reg),
                      U_regularizer=l2(self.l2reg),
                      name='rnn')(merged_input)

        else:
            logger.info("Building an LSTM with recurrent inputs %s", recurrent_inputs_names)
            rnn = LSTM(output_dim=self.hidden_size,
                      input_dim=rnn_input_dim,
                      return_sequences=True,
                      W_regularizer=l2(self.l2reg),
                      U_regularizer=l2(self.l2reg),
                      name='rnn')(merged_input)

        output = TimeDistributed(Dense(output_dim=self.vocab_size,
                                       input_dim=self.hidden_size,
                                       W_regularizer=l2(self.l2reg),
                                       activation='softmax'),
                                       name='output')(rnn)

        if self.optimiser == 'adam':
            # allow user-defined hyper-parameters for ADAM because it is
            # our preferred optimiser
            optimiser = Adam(lr=self.lr, beta_1=self.beta1,
                             beta_2=self.beta2,  epsilon=self.epsilon,
                             clipnorm=self.clipnorm)
            model = Model(input=inputs, output=output)
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
        text_input = Input(shape=(self.max_t, self.vocab_size), name='text')
        text_mask = Masking(mask_value=0., name='text_mask')(text_input)

        # Word embeddings
        wemb = TimeDistributed(Dense(output_dim=self.embed_size,
                                      input_dim=self.vocab_size,
                                      W_regularizer=l2(self.l2reg)),
                                      name="w_embed")(text_mask)
        drop_wemb = Dropout(self.dropin, name="wemb_drop")(wemb)

        # Embed -> Hidden
        emb_to_hidden = TimeDistributed(Dense(output_dim=self.hidden_size,
                                      input_dim=self.vocab_size,
                                      W_regularizer=l2(self.l2reg)),
                                      name='wemb_to_hidden')(drop_wemb)

        if use_image:
            # Image 'embedding'
            logger.info('Using image features: %s', use_image)
            img_input = Input(shape=(self.max_t, 4096), name='img')
            img_emb = TimeDistributed(Dense(output_dim=self.hidden_size,
                                            input_dim=4096,
                                            W_regularizer=l2(self.l2reg)),
                                            name='img_emb')(img_input)
            img_drop = Dropout(self.dropin, name='img_embed_drop')(img_emb)

        # Input nodes for the recurrent layer
        rnn_input_dim = self.hidden_size
        if use_image:
            recurrent_inputs = [emb_to_hidden, img_drop]
            recurrent_inputs_names = ['emb_to_hidden', 'img_drop']
            inputs = [text_input, img_input]
        merged_input = Merge(mode='sum')(recurrent_inputs)

        # Recurrent layer
        if self.gru:
            logger.info("Building a GRU with recurrent inputs %s", recurrent_inputs_names)
            rnn = GRU(output_dim=self.hidden_size,
                      input_dim=rnn_input_dim,
                      return_sequences=True,
                      W_regularizer=l2(self.l2reg),
                      U_regularizer=l2(self.l2reg),
                      name='rnn')(merged_input)

        else:
            logger.info("Building an LSTM with recurrent inputs %s", recurrent_inputs_names)
            rnn = LSTM(output_dim=self.hidden_size,
                      input_dim=rnn_input_dim,
                      return_sequences=True,
                      W_regularizer=l2(self.l2reg),
                      U_regularizer=l2(self.l2reg),
                      name='rnn')(merged_input)

        if self.optimiser == 'adam':
            # allow user-defined hyper-parameters for ADAM because it is
            # our preferred optimiser
            optimiser = Adam(lr=self.lr, beta_1=self.beta1,
                             beta_2=self.beta2, epsilon=self.epsilon,
                             clipnorm=self.clipnorm)
            model = Model(input=[text_input, img_input], output=rnn)
            print(model.get_config())
            model.compile(optimiser, {'rnn': 'categorical_crossentropy'})
        else:
            model.compile(self.optimiser, {'rnn': 'categorical_crossentropy'})

        if self.weights is not None:
            logger.info("... with weights defined in %s", self.weights)
            # Initialise the weights of the model
            shutil.copyfile("%s/weights.hdf5" % self.weights,
                            "%s/weights.hdf5.bak" % self.weights)
            f = h5py.File("%s/weights.hdf5" % self.weights)
            self.partial_load_weights(model, f)
            f.close()

        #plot(model, to_file="model.png")

        return model

    def partial_load_weights(self, model, f):
        '''
        Keras does not seem to support partially loading weights from one
        model into another model. This function achieves the same purpose so
        we can serialise the final RNN hidden state to disk.

        TODO: find / engineer a more elegant and general approach
        '''

        flattened_layers = model.layers

        # new file format
        filtered_layers = []
        for layer in flattened_layers:
            weights = layer.weights
            if weights:
                filtered_layers.append(layer)
        flattened_layers = filtered_layers

        layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]
        filtered_layer_names = []
        for name in layer_names[:-1]: # -1 so we clip out the output layer
            g = f[name]
            weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
            if len(weight_names):
                filtered_layer_names.append(name)
        layer_names = filtered_layer_names
        if len(layer_names) != len(flattened_layers):
            raise Exception('You are trying to load a weight file '
                            'containing ' + str(len(layer_names)) +
                            ' layers into a model with ' +
                            str(len(flattened_layers)) + ' layers.')

        # we batch weight value assignments in a single backend call
        # which provides a speedup in TensorFlow.
        weight_value_tuples = []
        for k, name in enumerate(layer_names):
            g = f[name]
            weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
            weight_values = [g[weight_name] for weight_name in weight_names]
            layer = flattened_layers[k]
            symbolic_weights = layer.weights
            if len(weight_values) != len(symbolic_weights):
                raise Exception('Layer #' + str(k) +
                                ' (named "' + layer.name +
                                '" in the current model) was found to '
                                'correspond to layer ' + name +
                                ' in the save file. '
                                'However the new layer ' + layer.name +
                                ' expects ' + str(len(symbolic_weights)) +
                                ' weights, but the saved weights have ' +
                                str(len(weight_values)) +
                                ' elements.')
            weight_value_tuples += zip(symbolic_weights, weight_values)
        K.batch_set_value(weight_value_tuples)

class MRNN:
    '''
    TODO: port this model architecture to Keras 1.0.7
    '''

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
            optimiser = Adam(lr=self.lr, beta_1=self.beta1,
                             beta_2=self.beta2, epsilon=self.epsilon,
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
 
