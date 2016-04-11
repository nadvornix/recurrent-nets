

from __future__ import absolute_import
import numpy as np

from keras import backend as K
from keras import activations, initializations
from keras.layers.core import MaskedLayer

from keras.layers.recurrent import Recurrent
import theano
class MyCW(Recurrent):
    def __init__(self, output_dim, periods, 
                 init='glorot_uniform', inner_init='orthogonal',
                 forget_bias_init='one', activation='tanh',
                 inner_activation='hard_sigmoid', **kwargs):

        self.output_dim = output_dim
        # self.periods = periods
        assert output_dim % len(periods) == 0
        self.periods = np.asarray(sorted(periods, reverse=True))

        self.init = initializations.get(init)
        # todo: init of other parts of network
        self.activation = activations.get(activation)

        super(MyCW, self).__init__(**kwargs)

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
            self.states = [None, None,None]

        # input_dim x output_dim
        # output dim = 50 = |h|?

        input_dim = self.input_dim
        output_dim = self.output_dim

        n = self.output_dim // len(self.periods)
        
        mask = np.zeros((self.output_dim, self.output_dim))
        period = np.zeros((self.output_dim, ), 'i')

        for i, T in enumerate(self.periods):
            mask[i*n:(i+1)*n, i*n:] = 1
            period[i*n:(i+1)*n] = T

        # from IPython import embed; embed()
        self.mask = K.zeros((self.output_dim, self.output_dim))
        self.period = K.zeros((self.output_dim, ), 'i')

        K.set_value(self.mask, mask)
        K.set_value(self.period, period)

        ## todo: mask & period are shared
        # n: K.zeros is shared by default (?)

        self.hh = self.init((self.output_dim, self.output_dim))
        self.xh = self.init((self.input_dim, self.output_dim))
        self.b = K.zeros((self.output_dim,), name="b")

        self.trainable_weights = [self.hh, self.xh, self.b]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'


        input_shape = self.input_shape
        
        (batch_size, tsteps, xsize) = input_shape

        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided ' +
                            '(including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((batch_size, self.output_dim)))
            K.set_value(self.states[1],
                        np.zeros((1), dtype="i"))
            K.set_value(self.states[0],
                        np.zeros((batch_size, self.output_dim)))

        else:
            self.states = [K.zeros((batch_size, self.output_dim), name="stateA"),
                        # K.variable(0),
                        # theano.shared(0),
                        K.zeros((1), name="stateB", dtype="int32"),
                        K.zeros((batch_size, self.output_dim), name="stateC"),
                        ]

    # def _step(self, t, x_t, p_tm1, h_tm1):
    #     return [p_t, self.activate(p_t)]
        
    def step(self, x, states):
        # assert len(states) == 3
        h_tm1 = states[0]
        t = states[1]
        p_tm1 = states[2]
        
        x_t = K.dot(x, self.xh) + self.b

        p = x_t + K.dot(h_tm1, self.hh * self.mask)

        p_t = K.switch(K.equal(t[0] % self.period, 0), p, p_tm1)
        
        h = self.activation(p_t)

        return h, [h, t+1, p_t]



    def get_config(self):
        config = {"output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "activation": self.activation.__name__,
                  "periods": self.periods, }
        base_config = super(MyCW, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))