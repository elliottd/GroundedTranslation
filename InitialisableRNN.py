from keras import backend as K
from keras.layers.recurrent import GRU, LSTM

class InitialisableGRU(GRU):
    """A GRU that allows its initial hidden state to be
       connected to different layer in the model.

    Inspired by:
        https://gist.github.com/mbollmann/29fd21931820c64095617125824ea246

    See Also:
        https://github.com/fchollet/#keras/issues/2995
    """

    def __init__(self, initial_state=None, **kwargs):
        super(InitialisableGRU, self).__init__(**kwargs)

    def build(self, input_shape):
        if isinstance(input_shape, list) and len(input_shape) > 1:
            input_shape, hidden_shapes = input_shape
        super(InitialisableGRU, self).build(input_shape)

    def call(self, x, mask=None):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        input_shape = self.input_spec[0].shape
        if isinstance(x, (tuple, list)):
            x, initial = x
        else:
            initial = None
        if K._BACKEND == 'tensorflow':
            if not input_shape[1]:
                raise Exception('When using TensorFlow, you should define '
                                'explicitly the number of timesteps of '
                                'your sequences.\n'
                                'If your first layer is an Embedding, '
                                'make sure to pass it an "input_length" '
                                'argument. Otherwise, make sure '
                                'the first layer has '
                                'an "input_shape" or "batch_input_shape" '
                                'argument, including the time axis. '
                                'Found input shape at layer ' + self.name +
                                ': ' + str(input_shape))
        if self.stateful and initial is not None:
            raise Exception(('Initial states should not be specified '
                             'for stateful GRUs, since they would overwrite '
                             'the memorized states.'))
        elif initial is not None:
            initial_states = [initial]
        elif self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(x)
        constants = self.get_constants(x)
        preprocessed_input = self.preprocess_input(x)

        # only use the main input mask
        if isinstance(mask, list):
            mask = mask[0]

        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=mask,
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=input_shape[1])
        if self.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.states[i], states[i]))

        if self.return_sequences:
            return outputs
        else:
            return last_output

    def get_output_shape_for(self, input_shape):
        if isinstance(input_shape, list) and len(input_shape) > 1:
            input_shape = input_shape[0]
        if self.return_sequences:
            output_shape = (input_shape[0], input_shape[1], self.output_dim)
        else:
            output_shape = (input_shape[0], self.output_dim)
        return [output_shape]

    def compute_mask(self, input, mask):
        if isinstance(mask, list) and len(mask) > 1:
            return mask[0]
        elif self.return_sequences:
            return mask[0]
        else:
            return [None]

class InitialisableLSTM(LSTM):
    """An LSTM that allows its initial hidden state to be
       connected to different layer in the model.

    Inspired by:
        https://gist.github.com/mbol#lmann/29fd21931820c64095617125824ea246

    See Also:
        https://github.com/fchollet/#keras/issues/2995
    """

    def __init__(self, initial_state=None, **kwargs):
        super(InitialisableLSTM, self).__init__(**kwargs)

    def build(self, input_shape):
        if isinstance(input_shape, list) and len(input_shape) > 1:
            input_shape, hidden_shapes = input_shape
        super(InitialisableLSTM, self).build(input_shape)

    def call(self, x, mask=None):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        input_shape = self.input_spec[0].shape
        if isinstance(x, (tuple, list)):
            x, initial = x
        else:
            initial = None
        if K._BACKEND == 'tensorflow':
            if not input_shape[1]:
                raise Exception('When using TensorFlow, you should define '
                                'explicitly the number of timesteps of '
                                'your sequences.\n'
                                'If your first layer is an Embedding, '
                                'make sure to pass it an "input_length" '
                                'argument. Otherwise, make sure '
                                'the first layer has '
                                'an "input_shape" or "batch_input_shape" '
                                'argument, including the time axis. '
                                'Found input shape at layer ' + self.name +
                                ': ' + str(input_shape))
        if self.stateful and initial is not None:
            raise Exception(('Initial states should not be specified '
                             'for stateful GRUs, since they would overwrite '
                             'the memorized states.'))
        elif initial is not None:
            # We set the initial memory cell and the initial hidden state to
            # the custom initial vector.
            initial_states = [initial, initial]
        elif self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(x)
        constants = self.get_constants(x)
        preprocessed_input = self.preprocess_input(x)

        # only use the main input mask
        if isinstance(mask, list):
            mask = mask[0]

        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=mask,
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=input_shape[1])
        if self.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.states[i], states[i]))

        if self.return_sequences:
            return outputs
        else:
            return last_output

    def get_output_shape_for(self, input_shape):
        if isinstance(input_shape, list) and len(input_shape) > 1:
            input_shape = input_shape[0]
        if self.return_sequences:
            output_shape = (input_shape[0], input_shape[1], self.output_dim)
        else:
            output_shape = (input_shape[0], self.output_dim)
        return [output_shape]

    def compute_mask(self, input, mask):
        if isinstance(mask, list) and len(mask) > 1:
            return mask[0]
        elif self.return_sequences:
            return mask[0]
        else:
            return [None]

