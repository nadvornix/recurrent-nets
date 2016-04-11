


# comp_model = nnb.ConcatenationModel(axis=0)
#             comp_model |= PerceptronLayer(insize=word_dim * 2, outsize=word_dim)
#             options.set('comp_model', comp_model)

import nnb
import numpy as np

#15 word vectors of size 5
word_vecs = np.random.random(size=(15,5))
#15 word matrices of size 5x5
word_mats = np.random.random(size=(15,5,5))

from keras_recursive import Recursive, RecTest
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Merge

ct=[[1,2], [4,3], [0,5]]
tree_size=6

batch_size=1

leaves_len=4

x_len = 5

sentence_vects = np.random.random(size=(tree_size, x_len))


left = Sequential()
left.add(Dense(50, input_shape=(5, 2)))
left.add(Activation('sigmoid'))

right = Sequential()
right.add(Dense(50, input_shape=(15, )))
right.add(Activation('sigmoid'))

model = Sequential()
m = Merge([left, right], mode='sum')
model.add(m)

model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')




# model = Graph()
# model.add_input(name='tree', input_shape=(len(ct),2,))
# model.add_input(name="x", input_shape=(tree_size,x_len, )) # TODO: just one x, or whole tree?

# model.add_node(name='tree_', input='tree',
#                    layer=Dropout(0.0))
# model.add_node(name='x_', input='x',
#                layer=Dropout(0.0))

# re = RecTest(stateful=True, output_dim=x_len)

# model.add_node(name="re", layer=re, merge_mode="join")
# # re = RecTest(name="re", input_dim=x_len, output_dim=x_len, stateful=True, batch_input_shape=(1, 6, 5))

# # model.add_node(name='reOut', input='re', layer=Activation('sigmoid'))

# model.add_node(name='out', input='re', layer=Dense(1))

# model.add_node(name='outS', input='out', layer=Activation('sigmoid'))

# model.add_output(name='score', input='outS')


# model.compile(loss='mse', optimizer='adam')


# print (net(ct,wi))
