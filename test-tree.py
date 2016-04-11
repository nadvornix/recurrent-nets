import nnb

parser=nnb.utils.ptb.PTBParser(filename="/storage/ostrava1/home/nadvorj1/trees/dev.txt")

i=0
trees = []

while True:
    tree = parser.parse()
    if tree == None:
        break
    trees.append(tree)
    i+=1



import numpy as np

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

xs=[]
for t in trees:
    empty_emb = [0,]*N
    words, tree, ys = t.get_features()
    s_len = len(words)
    words_emb = [g[w.lower()] if w.lower() in g else empty_emb for w in words]
    words_padded = pad_left(words_emb, L, empty_emb)
    tree_padded = pad_left([[0,0],]*len(words) + tree, L, [0,0])
    # ys_padded = pad_left(tree, L, ys)
    x = np.concatenate((np.asarray(tree_padded),np.asarray(words_padded)), axis=1)
    xs.append(x)
    ys.append(ys[-1]) # xxx: is this correct place?

Xs = np.asarray(xs)
Ys = np.asarray(ys)
from IPython import embed; embed()

with open("Xs.cache", "w") as f:
    np.save(f, Xs)

with open("Ys.cache", "w") as f:
    np.save(f, Ys)


