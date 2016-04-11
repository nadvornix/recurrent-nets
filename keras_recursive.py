
from __future__ import absolute_import
import numpy as np

from keras import backend as K
from keras import activations, initializations
from keras.layers.core import MaskedLayer, Layer
from keras.layers.recurrent import Recurrent
import theano

class Recursive(Recurrent):
    pass


class RecTest(Recursive):
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='sigmoid', **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        super(RecTest, self).__init__(**kwargs)

    def build(self):
        input_shape = self.input_shape
        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensor of shape (output_dim)
            self.states = [None]
        
        input_dim = input_shape[2]
        self.input_dim = input_dim

        self.W = self.init((input_dim, self.output_dim))
        self.U = self.inner_init((self.output_dim, self.output_dim))
        self.b = K.zeros((self.output_dim,))
        self.trainable_weights = [self.W, self.U, self.b]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided ' +
                            '(including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.output_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim))]

    def step(self, x, states):
        # states only contains the previous output.
        assert len(states) == 1
        prev_output = states[0]
        h = K.dot(x, self.W) + self.b
        output = self.activation(h + K.dot(prev_output, self.U))
        return output, [output]

    def get_config(self):
        config = {"output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "activation": self.activation.__name__}
        base_config = super(RecTest, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))















class Recursive2(Layer):
    input_ndim = 3

    def __init__(self, comp_model, output_dim, weights=None,
                 return_sequences=True, go_backwards=False, stateful=False,
                 input_dim=None, input_length=None, **kwargs):
        self.com_model = comp_model
        self.return_sequences = return_sequences
        self.initial_weights = weights
        self.go_backwards = go_backwards
        # self.stateful = stateful

        #todo: input_dim = model.input_dim...
        self.input_dim = input_dim
        self.input_length = input_length
        self.output_dim = output_dim
        # self.states = [K.zeros((input_shape[0], self.output_dim))]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(Recursive2, self).__init__(**kwargs)

    # def get_output_mask(self, train=False):
    #     # if self.return_sequences:
    #     #     return super(Recursive, self).get_output_mask(train)
    #     # else:
    #     return None

    def reset_states(self):
        return
        # input_shape = self.input_shape
        # if not input_shape[0]:
        #     raise Exception('If a RNN is stateful, a complete ' +
        #                     'input_shape must be provided ' +
        #                     '(including batch size).')
        # if hasattr(self, 'states'):
        #     K.set_value(self.states[0],
        #                 np.zeros((input_shape[0], self.output_dim)))
        # else:
        #     self.states = [K.zeros((input_shape[0], self.output_dim))]


    def build(self):
        #fixme: this should be got from model
        input_shape = self.input_shape
        input_dim = input_shape[2]
        self.input_dim = input_dim
        self.input = K.placeholder(input_shape)

        self.comp_model.build()
        self.trainable_weights = self.comp_model.trainable_weights

        self.reset_states()


    @property
    def output_shape(self):
        input_shape = self.input_shape
        # if self.return_sequences:
        return (input_shape[0], input_shape[1], self.output_dim)
        # else:
        #     return (input_shape[0], self.output_dim)

    def step(self, children, index, *partials): # are these params correct
        inputs1 = []
        inputs2 = []
        for partial in partials:
            inputs1.append(partial[children[0]])
            inputs2.append(partial[children[1]])

        model_out = self.comp_model.apply(inputs1 + inputs2)
        
        updates = theano.updates.OrderedUpdates()
        if isinstance(model_out, tuple):
            updates += model_out[1]
            model_out = model_out[0]

        new_partials = []
        for p, o in zip(partials, model_out):
            new_partials.append(T.set_subtensor(p[index], o))

        return new_partials, updates



    def get_initial_states(self, X):
        # build an all-zero tensor of shape (samples, output_dim)
        initial_state = K.zeros_like(X)  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=1)  # (samples, input_dim)
        reducer = K.zeros((self.input_dim, self.output_dim))
        initial_state = K.dot(initial_state, reducer)  # (samples, output_dim)
        initial_states = [initial_state for _ in range(len(self.states))]
        return initial_states

    def get_output(self, train=False):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        X = self.get_input(train)
        # mask = self.get_input_mask(train)

        assert K.ndim(X) == 3
        assert K._BACKEND == 'theano'

        # if self.stateful: #TODO: this seems important
        #     initial_states = self.states
        # else:
        #     initial_states = self.get_initial_states(X)
        initial_states = self.states #??

        ## last_output, outputs, states = K.renn(self.step, X,
        ##                                      initial_states,
        ##                                      go_backwards=self.go_backwards)

        
        #todo: ?!?!

        last_output, outputs, states = K.renn(self.step, X,
                                             initial_states,
                                             go_backwards=self.go_backwards)

        if self.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.states[i], states[i]))

        return outputs

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "return_sequences": self.return_sequences,
                  "go_backwards": self.go_backwards,
                  "stateful": self.stateful}
        # if self.stateful:
        #     config['batch_input_shape'] = self.input_shape
        # else:
        # config['input_dim'] = self.input_dim
        # config['input_length'] = self.input_length

        base_config = super(Recursive2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

