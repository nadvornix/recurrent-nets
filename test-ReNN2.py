from __future__ import print_function



# word_vecs = np.random.random(size=(15,5))
# #15 word matrices of size 5x5
# word_mats = np.random.random(size=(15,5,5))
import nnb
import numpy as np
# #15 word vectors of size 5

from keras_recursive import Recursive, RecTest
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Merge
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM

with open("Xs.cache", "rb") as f:
    Xs = np.load(f)

with open("Xs.cache", "rb") as f:
    Ys = np.load(f)

print("data loaded")
x_h, x_w = Xs[0].shape

batch_size=1

# graph model with two inputs and one output
# graph = Graph()
model = Sequential()

# model.add(Dense(x_w))
# model.add(Activation('sigmoid'))

model.add(LSTM(input_dim=x_w, output_dim=x_w, input_shape=(1, x_h, x_w), activation='sigmoid', return_sequences=False))
# model.add(Activation('sigmoid'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(optimizer='rmsprop', loss='mse')

history = model.fit(Xs, Ys, nb_epoch=100, batch_size=batch_size)

# # graph.add_input(name='x', input_shape=(x_h,x_w)) # is this input_shape correct?
# graph.add_input(name='x') # is this input_shape correct?
# graph.add_node(Dropout(0), name='x_', input="x")
# # graph.add_input(name='input2', input_shape=(32,))
# graph.add_node(LSTM(16), name='A', input="x_")
# # graph.add_node(Dense(4), name='dense2', input='input2')
# graph.add_node(Dense(1), name='B', input='A')
# # graph.add_output(name='output', inputs=['dense2', 'dense3'], merge_mode='sum')
# graph.add_output(name='output', input='B')

# print("model pepared")

# graph.compile(optimizer='rmsprop', loss={'output':'mse'})
# print("model compiled")

# history = graph.fit({'input1':X_train, 'input2':X2_train, 'output':y_train}, nb_epoch=10)
# history = graph.fit({'x':Xs, 'output':Ys}, nb_epoch=100, batch_size=batch_size)
# print("training finished")

# predictions = graph.predict({'input1':X_test, 'input2':X2_test}) # {'output':...}

