

import nnb
import numpy as np
import cPickle as pickle

parser=nnb.utils.ptb.PTBParser(filename="/storage/ostrava1/home/nadvorj1/trees/dev.txt")

i=0
trees = []

while True:
    tree = parser.parse()
    if tree == None:
        break
    trees.append(tree)
    i+=1




def pad_left(l, LEN, elem):
    """ left padding of l into length l with element e"""
    if len(l)>=LEN:
        return l[:LEN]
    return [elem]*(L-len(l)) + l

glovepath = "/storage/ostrava1/home/nadvorj1/glove.6B.100d.txt"
N=100

g = dict()
with open(glovepath, 'r') as f:
    for line in f:
        l = line.split()
        word = l[0]
        g[word] = np.array(l[1:]).astype(float)

print("GLOVE DONE")
L=60 # max sentence len

data=[]
for t in trees:
    empty_emb = [0,]*N # embedding for words not in glove
    words, tree, labels = t.get_features()
    embedded = [g[w.lower()] if w.lower() in g else empty_emb for w in words]

    data.append((words, tree, labels, embedded))
    
    # tree_padded = pad_left(tree, L, [0,0])
    # h_padded = pad_left(words_emb+[empty_emb]*num_w, L, empty_emb)
    # inputs = pad_left(words_emb+[empty_emb]*tree_len, L, empty_emb)
    # words_padded = pad_left(words_emb+[empty_emb]*num_w, L, empty_emb)
    # words_padded = pad_left(words_emb, L, empty_emb)
    # tree_padded = pad_left([[0,0],]*len(words) + tree, L, [0,0])
    # ys_padded = pad_left(tree, L, ys)
    # x = np.concatenate((np.asarray(tree_padded),np.asarray(words_padded)), axis=1)
    # xs.append(x)
    # ys.append(labels[-1]) # xxx: is this correct place to find sentiment of sentence?

with open("data.pickle", "w") as f:
    pickle.dump(data, f)

# Xs = np.asarray(xs)
# Ys = np.asarray(ys)
# from IPython import embed; embed()

# with open("xs.cache", "w") as f:
#     np.save(f, xs)

# with open("ys.cache", "w") as f:
#     np.save(f, ys)

