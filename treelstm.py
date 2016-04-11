
import numpy as np

from theano import tensor as T
import theano
import cPickle as pickle
floatX=theano.config.floatX


# from IPython import embed; embed()

L=60 # max sentence len

# from IPython import embed; embed()

random_seed=5
rng = np.random.RandomState(random_seed) 
params={}
n_classes = 5
l2_regularisation = 0.0001
learningrate = 0.1
######

def make_matrix(name, size):
    """Create a shared variable tensor and save it to self.params"""
    vals = np.asarray(rng.normal(loc=0.0, scale=0.1, size=size), dtype=floatX)
    params[name] = theano.shared(vals, name)
    return params[name]


# input_index = theano.tensor.iscalar('input_index')
# children = theano.tensor.ivector('children')
tree = theano.tensor.imatrix('tree')
embeddings = theano.tensor.fmatrix('embeddings')
target_class = theano.tensor.iscalar('target_class')

n=100

W_i = make_matrix("W_i", (n*2, n))
b_i = make_matrix("b_i", (n, 1))

W_o = make_matrix("W_o", (n*2, n))
b_o = make_matrix("b_o", (n, 1))

W_u = make_matrix("W_u", (n*2, n))
b_u = make_matrix("b_u", (n, 1))

W_f = make_matrix("W_f", (n, n))
U_fa = make_matrix("U_fa", (n*2, n))
U_fb = make_matrix("U_fb", (n*2, n))
b_f = make_matrix("b_f", (n, 1))

initial_index = theano.tensor.alloc(np.array(0, dtype=floatX), 1)

Cs = theano.tensor.fmatrix() # height==2*L-1
hs = theano.tensor.fmatrix() # height==2*L-1
xs = theano.tensor.fmatrix() # todo

def treelstm_step(childrenI, index, h_prev, W_i, b_i, W_o, b_o, W_u, b_u, W_f, U_fa, U_fb, b_f, Cs, hs, xs):
    """h_prev shall be unused here"""

    lI = childrenI[0]
    rI = childrenI[1]

    Ca = Cs[lI]
    Cb = Cs[rI]
    
    h_sum = hs[lI]+hs[rI]

    x = index[xs]
    xh = T.concatenate(x, h_sum)

    i = T.nnet.sigmoid(T.dot(W_i, xh)+b_i)
    o = T.nnet.sigmoid(T.dot(W_o, xh)+b_o)
    u = T.tanh(T.dot(W_u, xh)+b_u)

    fa = T.nnet.sigmoid(T.dot(W_f, x) + b_f + T.dot(U_fa, hs[lI]))
    fb = T.nnet.sigmoid(T.dot(W_f, x) + b_f + T.dot(U_fb, hs[rI]))

    c = i*u + fa*Ca + fb*Cb

    h = o * T.tanh(c)

    # TODO: via set subtensor
    # hs[index] = h
    # Cs[index] = c # TODO

    return index+1, h



hidden_vector, _ = theano.scan(
    treelstm_step,
    sequences=tree,
    outputs_info=initial_index,
    non_sequences=[W_i, b_i, W_o, b_o, W_u, b_u, W_f, U_fa, U_fb, b_f, Cs, hs, xs]
)

hidden_vector = hidden_vector[-1]


W_output = make_matrix('W_output', (n_classes,n))
output = theano.tensor.nnet.softmax([theano.tensor.dot(W_output, hidden_vector)])[0]
predicted_class = theano.tensor.argmax(output)


cost = -1.0 * theano.tensor.log(output[target_class])
for m in params.values():
    cost += l2_regularisation * (theano.tensor.sqr(m).sum())

gradients = theano.tensor.grad(cost, params.values())
updates = [(p, p - (learningrate * g)) for p, g in zip(params.values(), gradients)]


tree = theano.tensor.imatrix('tree')
embeddings = theano.tensor.fmatrix('embeddings')
target_class = theano.tensor.iscalar('target_class')


initial_index = theano.tensor.alloc(np.array(0, dtype=floatX), 1)

Cs = theano.tensor.fmatrix() # height==2*L-1
hs = theano.tensor.fmatrix() # height==2*L-1
xs = theano.tensor.fmatrix() # todo

train = theano.function([tree, embeddings, target_class], [cost, predicted_class], updates=updates, allow_input_downcast = True)

# test = theano.function([input_indices, target_class], [cost, predicted_class], allow_input_downcast = True)

#Prepare data:

def pad_left(l, LEN, elem):
    """ left padding of l into length l with element e"""
    if len(l)>=LEN:
        return l[:LEN]
    return [elem]*(L-len(l)) + l

with open("data.pickle", "r") as f:
    data = pickle.load(f)

#format: [(words, tree, labels, embeddings)...]
xs=[]
ys=[]
N=len(data[0][3][0]) # should be 100 - len of embedding of first word of first sentence

for words, tree, labels, embeddings in data:
    ys.append(labels[-1])
    tree_len=len(tree)
    empty_emb = [0]*N
    xs = pad_left(embeddings + [empty_emb] * tree_len, L, empty_emb)
    xs.append((tree, xs))


