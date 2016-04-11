
from __future__ import absolute_import
import numpy as np

from keras import backend as K
from keras import activations, initializations
from keras.layers.core import MaskedLayer

from keras.layers.recurrent import Recurrent

class MyLSTM(Recurrent):
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 forget_bias_init='one', activation='tanh',
                 inner_activation='hard_sigmoid', **kwargs):

        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        super(MyLSTM, self).__init__(**kwargs)

    def build(self):
        input_shape = self.input_shape
        input_dim = input_shape[2] # = |x| # works only for stateful? (todo: try)
        self.input_dim = input_dim
        self.input = K.placeholder(input_shape)
        

        # output dim = |c| = |h| = |output|
        # input dim = |x|

        if self.stateful:
            self.reset_states()
        else:
            # initial states: 2 all-zero tensor of shape (output_dim)
            self.states = [None, None]

        # input_dim x output_dim
        # output dim = 50 = |h|?

        input_dim = self.input_dim
        output_dim = self.output_dim

        self.W_fx = self.init((input_dim, output_dim))
        self.W_fh = self.inner_init((output_dim, output_dim))
        self.b_f = self.forget_bias_init((output_dim, ))

        self.W_ix = self.init((input_dim, output_dim))
        self.W_ih = self.inner_init((output_dim, output_dim))
        self.b_i = K.zeros((output_dim, ))
        
        self.W_cx = self.init((input_dim, output_dim))
        self.W_ch = self.inner_init((output_dim, output_dim))
        self.b_c = K.zeros((output_dim, ))
        
        self.W_ox = self.init((input_dim, output_dim))
        self.W_oh = self.inner_init((output_dim, output_dim))
        self.b_o = K.zeros((output_dim, ))

        self.trainable_weights = [self.W_fx, self.W_fh, self.b_f,
            self.W_ix, self.W_ih, self.b_i,
            self.W_cx, self.W_ch, self.b_c,
            self.W_ox, self.W_oh, self.b_o,
        ]


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
            K.set_value(self.states[1],
                        np.zeros((input_shape[0], self.output_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim)),
                           K.zeros((input_shape[0], self.output_dim))]

    def step(self, x, states):
        assert len(states) == 2
        h_tm1 = states[0]
        c_tm1 = states[1]

        x_f = K.dot(x, self.W_fx) + self.b_f
        x_i = K.dot(x, self.W_ix) + self.b_i
        x_c = K.dot(x, self.W_cx) + self.b_c
        x_o = K.dot(x, self.W_ox) + self.b_o

        f = self.inner_activation(K.dot(h_tm1, self.W_fh) + x_f)
        i = self.inner_activation(K.dot(h_tm1, self.W_ih) + x_i)
        cprime = self.activation(K.dot(h_tm1,self.W_ch) + x_c)
        c = f * c_tm1 + i * cprime
        o = self.activation(K.dot(h_tm1, self.W_oh) + x_o)
        h = o * self.inner_activation(c)

        return h, [h, c]

    def get_config(self):
        config = {"output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "forget_bias_init": self.forget_bias_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__}
        base_config = super(MyLSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

