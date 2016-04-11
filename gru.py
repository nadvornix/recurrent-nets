

from __future__ import absolute_import
import numpy as np

from keras import backend as K
from keras import activations, initializations
from keras.layers.core import MaskedLayer

from keras.layers.recurrent import Recurrent

class MyGRU(Recurrent):
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 forget_bias_init='one', activation='tanh',
                 inner_activation='hard_sigmoid', **kwargs):

        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        # self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation) # tanh
        self.inner_activation = activations.get(inner_activation) # sigmoid
        super(MyGRU, self).__init__(**kwargs)

    def build(self):
        input_shape = self.input_shape
        input_dim = input_shape[2] # = |x| # works only for stateful? (todo: try)
        self.input_dim = input_dim
        self.input = K.placeholder(input_shape)
        
        # from IPython import embed; embed()

        # output dim = |c| = |h| = |output|
        # input dim = |x|

        if self.stateful:
            self.reset_states()
        else:
            # initial states: 2 all-zero tensor of shape (output_dim)
            self.states = [None]

        # input_dim x output_dim
        # output dim = 50 = |h|?

        input_dim = self.input_dim
        output_dim = self.output_dim


        self.W_zx = self.init((input_dim, output_dim))
        self.W_zh = self.inner_init((output_dim, output_dim))
        self.b_z = K.zeros((self.output_dim,))

        self.W_rx = self.init((input_dim, output_dim))
        self.W_rh = self.inner_init((output_dim, output_dim))
        self.b_r = K.zeros((self.output_dim,))

        self.W_Mx = self.init((input_dim, output_dim))
        self.W_Mrh = self.inner_init((output_dim, output_dim))
        self.b_M = K.zeros((self.output_dim,))

        self.trainable_weights = [self.W_zx, self.W_zh, self.b_z,
                                  self.W_rx, self.W_rh, self.b_r,
                                  self.W_Mx, self.W_Mrh, self.b_M]

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
            # K.set_value(self.states[1],
            #             np.zeros((input_shape[0], self.output_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim))]

    def step(self, x, states):
        assert len(states) == 1
        h_tm1 = states[0]

        z_tx = K.dot(x, self.W_zx) + self.b_z
        z_th = K.dot(h_tm1, self.W_zh)

        r_tx = K.dot(x, self.W_rx) + self.b_r
        r_th = K.dot(h_tm1, self.W_rh)

        M_t = K.dot(x,self.W_Mx) + self.b_M

        r = self.inner_activation(r_th + r_tx)
        hprime = self.activation(M_t + K.dot(r*h_tm1, self.W_Mrh))
        # from IPython import embed; embed()

        z = self.inner_activation(z_tx + z_th)

        h = z * h_tm1 + (1-z) * hprime

        return h, [h]

    def get_config(self):
        config = {"output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  # "forget_bias_init": self.forget_bias_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__}
        base_config = super(MyGRU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))