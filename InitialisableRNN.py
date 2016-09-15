from keras import backend as K
from keras.layers.recurrent import GRU, LSTM

class InitialisableGRU(GRU):
    """A GRU that allows its initial hidden state to be
       connected to different layer in the model.

       Our changes are in the __init__() and call()

    Inspired by:
        https://gist.github.com/mbol#lmann/29fd21931820c64095617125824ea246

    See Also:
        https://github.com/fchollet/#keras/issues/2995
    """

    def __init__(self, initial_state=None, **kwargs):
        self.initial_state = initial_state
        super(InitialisableGRU, self).__init__(**kwargs)

    def build(self, input_shape):
        super(InitialisableGRU, self).build(input_shape)

    def call(self, x, mask=None):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        input_shape = self.input_spec[0].shape
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
        if self.stateful and self.initial_state is not None:
            raise Exception(('Initial states should not be specified '
                             'for stateful GRUs, since they would overwrite '
                             'the memorized states.'))
        elif self.initial_state is not None:
            initial_states = [self.initial_state]
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

class InitialisableLSTM(LSTM):
    """An LSTM that allows its initial hidden state and its initial memory
       to be connected to different layer in the model.

       Our changes are in the __init__() and call()

    Inspired by:
        https://gist.github.com/mbol#lmann/29fd21931820c64095617125824ea246

    See Also:
        https://github.com/fchollet/#keras/issues/2995
    """

    def __init__(self, initial_state=None, **kwargs):
        self.initial_state = initial_state
        super(InitialisableLSTM, self).__init__(**kwargs)

    def build(self, input_shape):
        super(InitialisableLSTM, self).build(input_shape)

    def call(self, x, mask=None):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        input_shape = self.input_spec[0].shape
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
        if self.stateful and self.initial_state is not None:
            raise Exception(('Initial states should not be specified '
                             'for stateful LSTMs, since they would overwrite '
                             'the memorized states.'))
        elif self.initial_state is not None:
            initial_states = [self.initial_state, self.initial_state]
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
