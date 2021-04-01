import numpy as np
import argparse
from scipy import io, spatial
import time
from random import shuffle
import random
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import glob
from collections import Counter

from sje import *

mtxs = []
for fname in glob.glob('attribute_embeddings/*'):
    print(fname)
    mtx = np.loadtxt(fname)
    mtxs.append(mtx)
attr_embs = np.concatenate(mtxs,axis=1)


mtxs = []
for fname in glob.glob('attribute_probabilities/*'):
    print(fname)
    mtx = np.loadtxt(fname)
    mtxs.append(mtx)
attr_prob = np.concatenate(mtxs,axis=1)

clf1 = SJE(attr_prob)
clf1.evaluate()

clf1 = SJE(attr_embs)
clf1.evaluate()