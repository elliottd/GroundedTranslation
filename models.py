from keras.models import Model
from keras.layers import Input, Activation, Dropout, Merge, TimeDistributed, Masking, Dense, Lambda, LSTM
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.optimizers import Adam
from keras import backend as K
from InitialisableRNN import InitialisableGRU, InitialisableLSTM

import h5py
import shutil
import logging
import sys
import numpy as np

# Set up logger
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


class NIC:

    def __init__(self, options):
        self.max_t = options.max_t  # Expected timesteps. Needed to build the Theano graph

        # Model hyperparameters
        self.vocab_size = options.vocab_size  # size of word vocabulary
        self.embed_size = options.embed_size  # number of units in a word embedding
        self.hsn_size = options.source_size  # size of the source hidden vector
        self.hidden_size = options.hidden_size  # number of units in first LSTM
        self.gru = options.gru  # gru recurrent layer? (false = lstm)
        self.dropin = options.dropin  # prob. of dropping input units
        self.l2reg = options.l2reg  # weight regularisation penalty

        # Optimiser hyperparameters
        self.optimiser = options.optimiser  # optimisation method
        self.lr = options.lr
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.clipnorm = options.clipnorm

        self.weights = options.init_from_checkpoint  # initialise with checkpointed weights?
        self.transfer_img_emb = options.transfer_img_emb

    def buildKerasModel(self, 
                        use_sourcelang=False, 
                        use_image=True, 
                        embeddings=None, 
                        init_output=False,
                        fix_weights=False):
        '''
        Define the exact structure of your model here. We create an image
        description generation model by merging the VGG image features with
        a word embedding model, with an RNN over the sequences.
        '''
        logger.info('Building Keras model...')

        text_input = Input(shape=(self.max_t,), name='text')# self.vocab_size), name='text')
        text_mask = Masking(mask_value=0., name='text_mask')(text_input)

        print(text_input._keras_shape)

        # Word embeddings
        if embeddings is not None:
            wemb = Embedding(output_dim=self.embed_size,
                             input_dim=self.vocab_size,
                             input_length=self.max_t,
                             W_regularizer=l2(self.l2reg),
                             weights=[embeddings],
                             mask_zero=True,
                             trainable=not fix_weights,
                             name="w_embed")(text_input)
        else:
            wemb = Embedding(output_dim=self.embed_size,
                             input_dim=self.vocab_size,
                             input_length=self.max_t,
                             W_regularizer=l2(self.l2reg),
                             mask_zero=True,
                             name="w_embed")(text_input)

        drop_wemb = Dropout(self.dropin, name="wemb_drop")(wemb)

        # Embed -> Hidden
        emb_to_hidden = TimeDistributed(Dense(output_dim=self.hidden_size,
                                              input_dim=self.embed_size,
                                              W_regularizer=l2(self.l2reg)),
                                              name='wemb_to_hidden')(drop_wemb)

        if use_image and use_sourcelang:
            # Concatenated embedding
            logger.info('Using %d dim image features', 4096)
            img_input = Input(shape=(4096,), name='img')
            logger.info('Using %d dim source features', self.hsn_size)
            src_input = Input(shape=(self.hsn_size,), name='src')
            src_relu = Activation('relu')(src_input)
            logger.info("Creating a %d dim concatenated input", (4096+self.hsn_size))
            merged_input = Merge(mode='concat')([img_input, src_relu])
            merge_embed = Dense(output_dim=self.hidden_size,
                                           input_dim=4096+self.hsn_size,
                                           W_regularizer=l2(self.l2reg),
                                           name="merge_embed")(merged_input)
            merge_drop = Dropout(self.dropin, name="merge_drop")(merge_embed)
            rnn_initialisation = merge_drop
            model_inputs = [text_input, img_input, src_input]
        elif use_image:
            logger.info('Using %d dim image features', 4096)
            img_input = Input(shape=(4096,), name='img')
            # Image 'embedding'
            img_emb = Dense(output_dim=self.hidden_size,
                                       input_dim=4096,
                                       W_regularizer=l2(self.l2reg),
                                       name='img_emb')(img_input)
            img_drop = Dropout(self.dropin, name='img_embed_drop')(img_emb)
            rnn_initialisation = img_drop
            model_inputs = [text_input, img_input]
        elif use_sourcelang:
            logger.info('Using %d dim source features', self.hsn_size)
            src_input = Input(shape=(self.hsn_size,), name='src')
            src_relu = Activation('relu')(src_input)
            # Source 'embedding'
            src_embed = Dense(output_dim=self.hidden_size,
                                         input_dim=self.hsn_size,
                                         W_regularizer=l2(self.l2reg),
                                         name="src_embed")(src_relu)
            src_drop = Dropout(self.dropin, name="src_drop")(src_embed)
            rnn_initialisation = src_embed
            model_inputs = [text_input, src_input]

        # Recurrent layer
        if self.gru:
            logger.info("Building a GRU")
            rnn = InitialisableGRU(output_dim=self.hidden_size,
                      input_dim=self.hidden_size,
                      return_sequences=True,
                      W_regularizer=l2(self.l2reg),
                      U_regularizer=l2(self.l2reg),
                      name='rnn')([emb_to_hidden, rnn_initialisation])
        else:
            logger.info("Building an LSTM")
            rnn = InitialisableLSTM(output_dim=self.hidden_size,
                      input_dim=self.hidden_size,
                      return_sequences=True,
                      W_regularizer=l2(self.l2reg),
                      U_regularizer=l2(self.l2reg),
                      name='rnn')([emb_to_hidden, rnn_initialisation])

        rnn_to_output = TimeDistributed(Dense(output_dim=self.embed_size,
                                              input_dim=self.hidden_size,
                                              W_regularizer=l2(self.l2reg)),
                                        name='rnn_to_output')(rnn)
        if init_output:
            output = TimeDistributed(Dense(output_dim=self.vocab_size,
                                           input_dim=self.embed_size,
                                           W_regularizer=l2(self.l2reg),
                                           activation='softmax',
                                           weights=[embeddings.T]),
                                           trainable=not fix_weights,
                                     name='output')(rnn_to_output)
        else:
            output = TimeDistributed(Dense(output_dim=self.vocab_size,
                                           input_dim=self.embed_size,
                                           W_regularizer=l2(self.l2reg),
                                           activation='softmax'),
                                     name='output')(rnn_to_output)


        if self.optimiser == 'adam':
            # allow user-defined hyper-parameters for ADAM because it is
            # our preferred optimiser
            optimiser = Adam(lr=self.lr, beta_1=self.beta1,
                             beta_2=self.beta2,  epsilon=self.epsilon,
                             clipnorm=self.clipnorm)
        else:
            optimiser = self.optimiser
        model = Model(input=model_inputs, output=output)
        if self.weights is not None:
            logger.info("... with weights defined in %s", self.weights)
            # Initialise the weights of the model
            shutil.copyfile("%s/weights.hdf5" % self.weights,
                            "%s/weights.hdf5.bak" % self.weights)
            model.load_weights("%s/weights.hdf5" % self.weights)
        model.compile(optimiser, {'output': 'categorical_crossentropy'})

        if self.transfer_img_emb is not None:
            self.load_specific_weight(model, self.transfer_img_emb, 'img_emb')

        #plot(model, to_file="model.png")

        return model

    def buildHSNActivations(self, use_image=True):
        '''
        We cut off the output layer of the model because we're only interested
        in the activations in the hidden states.
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

        # Image 'embedding'
        logger.info('Using image features: %s', use_image)
        img_input = Input(shape=(4096,), name='img')
        img_emb = Dense(output_dim=self.hidden_size,
                                   input_dim=4096,
                                   W_regularizer=l2(self.l2reg),
                                   name='img_emb')(img_input)
        img_drop = Dropout(self.dropin, name='img_embed_drop')(img_emb)

        # Input nodes for the recurrent layer
        model_inputs = [text_input, img_input]
        rnn_initialisation = img_drop

        # Recurrent layer
        if self.gru:
            logger.info("Building a GRU")
            rnn = InitialisableGRU(output_dim=self.hidden_size,
                                   input_dim=self.hidden_size,
                                   return_sequences=True,
                                   W_regularizer=l2(self.l2reg),
                                   U_regularizer=l2(self.l2reg),
                                   name='rnn')([emb_to_hidden,
                                       rnn_initialisation])
        else:
            logger.info("Building an LSTM")
            rnn = InitialisableLSTM(output_dim=self.hidden_size,
                                    input_dim=self.hidden_size,
                                    return_sequences=True,
                                    W_regularizer=l2(self.l2reg),
                                    U_regularizer=l2(self.l2reg),
                                    name='rnn')([emb_to_hidden, rnn_initialisation])

        if self.optimiser == 'adam':
            # allow user-defined hyper-parameters for ADAM because it is
            # our preferred optimiser
            optimiser = Adam(lr=self.lr, beta_1=self.beta1,
                             beta_2=self.beta2, epsilon=self.epsilon,
                             clipnorm=self.clipnorm)
        else:
            optimiser = self.optimiser
        model = Model(input=model_inputs, output=rnn)
        model.compile(optimiser, {'rnn': 'categorical_crossentropy'})

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

    def load_specific_weight(self, model, pretrained, target_layer_name):
        '''
        Keras does not seem to support partially loading weights from one
        model into another model. This function makes it possible to load a
        specific weight into the model using the target_layer_name variable.

        TODO: find / engineer a more elegant and general approach
        '''

        w_file = h5py.File("%s/weights.hdf5" % pretrained)
        logger.info("Trying to load %s from %s", target_layer_name, pretrained)
        # new file format

        flattened_layers = model.layers
        target_layer = None
        for layer in model.layers:
            weights = layer.weights
            if layer.name == target_layer_name:
                target_layer = layer
                break

        # Get a list of all the weights in the pre-trained weights file
        pretrained_layer_names = [n.decode('utf8') for n in w_file.attrs['layer_names']]
        filtered_layer_names = []
        for name in pretrained_layer_names:
            g = w_file[name]
            weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
            if len(weight_names):
                filtered_layer_names.append(name)
        layer_names = filtered_layer_names

        for k, name in enumerate(layer_names):
            if name == target_layer_name:
                g = w_file[name]
                weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
                weight_values = [g[weight_name] for weight_name in weight_names]
                target_layer.set_weights(weight_values)

    def full_load_weights(self, model, f):
        '''
        Keras does not seem to support partially loading weights from one
        model into another model. This function achieves the same purpose so
        we can serialise the final RNN hidden state to disk.

        TODO: find / engineer a more elegant and general approach
        '''

        f = h5py.File("%s" % f)
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
        for name in layer_names: # -1 so we clip out the output layer
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
        f.close()

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
        print(layer_names)
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

 
